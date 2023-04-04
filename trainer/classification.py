import os
import copy
import datetime
import torch
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term, accuracy
from util.net import save_checkpoint, trans_state_dict, print_networks, get_timepc, reduce_tensor
from optim.scheduler import get_scheduler
from data import get_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from timm.data import Mixup

from torch.nn.parallel import DistributedDataParallel as NativeDDP
try:
	from apex import amp
	from apex.parallel import DistributedDataParallel as ApexDDP
	from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
	from timm.models import convert_sync_batchnorm as ApexSyncBN
	print('apex is not available, using ApexSyncBN to replace')
from timm.models import convert_sync_batchnorm as TIMMSyncBN

from timm.utils import dispatch_clip_grad
from util.net import get_loss_scaler, get_autocast, distribute_bn

from . import TRAINER


@TRAINER.register_module
class CLS():
	def __init__(self, cfg):
		self.cfg = cfg
		self.master, self.logger, self.writer = cfg.master, cfg.logger, cfg.writer
		self.local_rank, self.rank, self.world_size = cfg.local_rank, cfg.rank, cfg.world_size
		log_msg(self.logger, '==> Running Trainer: {}'.format(cfg.trainer.name))
		# =========> model <=================================
		log_msg(self.logger, '==> Using GPU: {} for Training'.format(list(range(cfg.world_size))))
		log_msg(self.logger, '==> Building model')
		self.net = get_model(cfg.model)
		self.net.to('cuda:{}'.format(cfg.local_rank))
		self.net.eval()
		self.ema = cfg.trainer.ema
		if self.ema:
			self.net_E = copy.deepcopy(self.net)
			self.net_E.eval()
		else:
			self.net_E = None
		log_msg(self.logger, f"==> Load checkpoint: {cfg.model.model_kwargs['checkpoint_path']}") if cfg.model.model_kwargs['checkpoint_path'] else None
		print_networks([self.net], self.cfg.size, self.logger)
		self.dist_BN = cfg.trainer.dist_BN
		if cfg.dist and cfg.trainer.sync_BN != 'none':
			self.dist_BN = ''
			log_msg(self.logger, f'==> Synchronizing BN by {cfg.trainer.sync_BN}')
			syncbn_dict = {'apex': ApexSyncBN, 'native': torch.nn.SyncBatchNorm.convert_sync_batchnorm, 'timm': TIMMSyncBN}
			self.net = syncbn_dict[cfg.trainer.sync_BN](self.net)
		log_msg(self.logger, '==> Creating optimizer')
		cfg.optim.lr *= cfg.trainer.data.batch_size / 512
		cfg.trainer.scheduler_kwargs['lr_min'] *= cfg.trainer.data.batch_size_per_gpu * self.world_size / 512
		cfg.trainer.scheduler_kwargs['warmup_lr'] *= cfg.trainer.data.batch_size_per_gpu * self.world_size / 512
		self.optim = get_optim(cfg, self.net, lr=cfg.optim.lr)
		self.amp_autocast = get_autocast(cfg.trainer.scaler)
		self.loss_scaler = get_loss_scaler(cfg.trainer.scaler)
		self.loss_terms = get_loss_terms(cfg.loss.loss_terms, device='cuda:{}'.format(cfg.local_rank))
		if cfg.trainer.scaler == 'apex':
			self.net, self.optim = amp.initialize(self.net, self.optim, opt_level='O1')
		if cfg.dist:
			if cfg.trainer.scaler in ['none', 'native']:
				log_msg(self.logger, '==> Native DDP')
				self.net = NativeDDP(self.net, device_ids=[cfg.local_rank], find_unused_parameters=cfg.trainer.find_unused_parameters)
			elif cfg.trainer.scaler in ['apex']:
				log_msg(self.logger, '==> Apex DDP')
				self.net = ApexDDP(self.net, delay_allreduce=True)
			else:
				raise 'Invalid scaler mode: {}'.format(cfg.trainer.scaler)
		# =========> dataset <=================================
		cfg.logdir_train, cfg.logdir_test = f'{cfg.logdir}/show_train', f'{cfg.logdir}/show_test'
		makedirs([cfg.logdir_train, cfg.logdir_test], exist_ok=True)
		log_msg(self.logger, "==> Loading dataset: {}".format(cfg.data.name))
		self.train_loader, self.test_loader = get_loader(cfg)
		cfg.data.train_size, cfg.data.test_size = len(self.train_loader), len(self.test_loader)
		cfg.data.train_length, cfg.data.test_length = self.train_loader.dataset.length, self.test_loader.dataset.length
		self.mixup_fn = Mixup(**cfg.trainer.mixup_kwargs) if cfg.trainer.mixup_kwargs['prob'] > 0 else None
		self.scheduler = get_scheduler(cfg, self.optim)
		self.topk_recorder = cfg.trainer.topk_recorder
		self.iter, self.epoch = cfg.trainer.iter, cfg.trainer.epoch
		self.iter_full, self.epoch_full = cfg.trainer.iter_full, cfg.trainer.epoch_full
		self.nan_or_inf_cnt = 0
		if cfg.trainer.resume_dir:
			state_dict = torch.load(cfg.model.model_kwargs['checkpoint_path'], map_location='cpu')
			self.net_E.load_state_dict(state_dict['net_E'], strict=cfg.model.model_kwargs['strict']) if self.ema and 'net_E' in state_dict else None
			self.optim.load_state_dict(state_dict['optimizer'])
			self.scheduler.load_state_dict(state_dict['scheduler'])
			self.loss_scaler.load_state_dict(state_dict['scaler']) if self.loss_scaler else None
			self.cfg.task_start_time = get_timepc() - state_dict['total_time']
			self.nan_or_inf_cnt = state_dict['nan_or_inf_cnt'] if state_dict.get('nan_or_inf_cnt', None) else 0
		else:
			pass
		self.train_mode = True if self.cfg.mode == 'train' else False
		log_cfg(self.cfg)
		
	def reset(self, isTrain=True, train_mode=True):
		self.isTrain = isTrain
		self.net.train() if train_mode else self.net.eval()
		self.log_terms, self.progress = get_log_terms(able(self.cfg.logging.log_terms_train, isTrain, self.cfg.logging.log_terms_test), default_prefix=('Train' if isTrain else 'Test'))
		
	def scheduler_step(self, step):
		self.scheduler.step(step)
		update_log_term(self.log_terms.get('lr'), self.optim.param_groups[0]["lr"], 1, self.master)
		
	def set_input(self, inputs):
		self.imgs = inputs['img'].cuda()
		self.targets = inputs['target'].cuda()
		self.bs = self.imgs.shape[0]
	
	def forward(self, net=None):
		self.outputs = net(self.imgs) if net is not None else self.net(self.imgs)
		if not isinstance(self.outputs, dict):
			self.outputs = {'out': self.outputs, 'out_kd': self.outputs}
		
	def backward_term(self, loss_term, optim):
		optim.zero_grad()
		if self.loss_scaler:
			self.loss_scaler(loss_term, optim, clip_grad=self.cfg.loss.clip_grad, parameters=self.net.parameters(), create_graph=self.cfg.loss.create_graph)
		else:
			loss_term.backward(retain_graph=self.cfg.loss.retain_graph)
			if self.cfg.loss.clip_grad is not None:
				dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
			optim.step()

	def check_bn(self):
		if hasattr(self.net, 'module'):
			self.net.module.check_bn() if hasattr(self.net.module, 'check_bn') else None
		else:
			self.net.check_bn() if hasattr(self.net, 'check_bn') else None
		
	@torch.no_grad()
	def _update_ema(self):
		if self.ema > 0:
			ema_model, cur_model = self.net_E, self.net
			for ema_params, current_params in zip(ema_model.state_dict().values(), cur_model.state_dict().values()):
				if ema_params.dtype in [torch.int64]:  # bn.num_batches_tracked --> torch.int64
					continue
				ema_params.data.mul_(self.ema).add_(current_params.data, alpha=1. - self.ema)
		
	def optimize_parameters(self):
		if self.mixup_fn is not None:
			self.imgs, self.targets = self.mixup_fn(self.imgs, self.targets)
		with self.amp_autocast():
			self.forward()
			nan_or_inf_out = 1. if torch.any(torch.isnan(self.outputs['out'])) or torch.any(torch.isinf(self.outputs['out'])) else 0.
			nan_or_inf_out = reduce_tensor(nan_or_inf_out, self.world_size, mode='sum', sum_avg=False).clone().detach().item()
			nan_or_inf_out = True if nan_or_inf_out > 0. else False
			if nan_or_inf_out:
				self.nan_or_inf_cnt += 1
				log_msg(self.logger, f'NaN or Inf Found, total {self.nan_or_inf_cnt} times')
				self.check_bn()
			loss_ce = self.loss_terms['CE'](self.outputs['out'], self.targets) if not nan_or_inf_out else 0
			loss_kd = (self.loss_terms['KD'](self.outputs['out_kd'], self.imgs) if self.loss_terms.get('KD', None) else 0) if not nan_or_inf_out else 0
		self.backward_term((loss_ce + loss_kd) if not nan_or_inf_out else (0 * self.outputs['out'][0, 0]), self.optim)
		update_log_term(self.log_terms.get('CE'), reduce_tensor(loss_ce, self.world_size).clone().detach().item(), 1, self.master)
		update_log_term(self.log_terms.get('KD'), reduce_tensor(loss_kd, self.world_size).clone().detach().item(), 1, self.master)
		self._update_ema()
		
	def post_update(self):
		if not self.cfg.trainer.mixup_kwargs or not self.isTrain:
			top15, top15bs_cnt = accuracy(self.outputs['out'], self.targets, topk=(1, 5))
			# top1, top5 = top15
			# update_log_term(self.log_terms.get('top1'), reduce_tensor(top1, self.world_size).clone().detach().item(), self.bs, self.master)
			# update_log_term(self.log_terms.get('top5'), reduce_tensor(top5, self.world_size).clone().detach().item(), self.bs, self.master)
			top1_cnt, top5_cnt, top_all = top15bs_cnt
			update_log_term(self.log_terms.get('top1_cnt'), reduce_tensor(top1_cnt, self.world_size, mode='sum', sum_avg=False).clone().detach().item(), 1, self.master)
			update_log_term(self.log_terms.get('top5_cnt'), reduce_tensor(top5_cnt, self.world_size, mode='sum', sum_avg=False).clone().detach().item(), 1, self.master)
			update_log_term(self.log_terms.get('top_all'), reduce_tensor(top_all, self.world_size, mode='sum', sum_avg=False).clone().detach().item(), 1, self.master)
	
	def _finish(self):
		log_msg(self.logger, 'finish training')
		self.writer.close() if self.master else None
		topk_list = [self.topk_recorder['net_top1'], self.topk_recorder['net_top5']]
		topk_list.extend([self.topk_recorder['net_E_top1'], self.topk_recorder['net_E_top5']]) if self.ema else None
		f = open(f'{self.cfg.logdir}/top.txt', 'w')
		msg = ''
		for i in range(len(topk_list[0])):
			for j in range(len(topk_list)):
				msg += '{:3.5f}\t'.format(topk_list[j][i])
			msg += '\n'
		f.write(msg)
		f.close()
	
	def train(self):
		self.reset(isTrain=True, train_mode=self.train_mode)
		self.check_bn()
		self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
		train_length = self.cfg.data.train_size
		train_loader = iter(self.train_loader)
		while self.epoch < self.epoch_full and self.iter < self.iter_full:
			self.scheduler_step(self.iter)
			# ---------- data ----------
			t1 = get_timepc()
			self.iter += 1
			train_data = next(train_loader)
			self.set_input(train_data)
			t2 = get_timepc()
			update_log_term(self.log_terms.get('data_t'), t2 - t1, 1, self.master)
			# ---------- optimization ----------
			self.optimize_parameters()
			t3 = get_timepc()
			update_log_term(self.log_terms.get('optim_t'), t3 - t2, 1, self.master)
			update_log_term(self.log_terms.get('batch_t'), t3 - t1, 1, self.master)
			# ---------- log ----------
			if self.master:
				if self.iter % self.cfg.logging.train_log_per == 0:
					msg = able(self.progress.get_msg(self.iter, self.iter_full, self.iter / train_length, self.iter_full / train_length), self.master, None)
					log_msg(self.logger, msg)
					if self.writer:
						for k, v in self.log_terms.items():
							self.writer.add_scalar(f'Train/{k}', v.val, self.iter)
						self.writer.flush()
			if self.iter % self.cfg.logging.train_reset_log_per == 0:
				self.reset(isTrain=True, train_mode=self.train_mode)
			# ---------- update train_loader ----------
			if self.iter % train_length == 0:
				self.epoch += 1
				if self.cfg.dist and self.dist_BN != '':
					distribute_bn(self.net, self.world_size, self.dist_BN)
					distribute_bn(self.net_E, self.world_size, self.dist_BN)
				self.optim.sync_lookahead() if hasattr(self.optim, 'sync_lookahead') else None
				if self.epoch >= self.cfg.trainer.start_test_epoch or self.epoch % self.cfg.trainer.every_test_epoch == 0:
					self.test()
				else:
					self.test_ghost()
					self.is_best, self.is_best_ema = False, False
				self.cfg.total_time = get_timepc() - self.cfg.task_start_time
				total_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time)))
				eta_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time / self.epoch * (self.epoch_full - self.epoch))))
				log_msg(self.logger, f'==> Total time: {total_time_str}\t Eta: {eta_time_str} \tLogged in \'{self.cfg.logdir}\'')
				self.save_checkpoint()
				self.reset(isTrain=True, train_mode=self.train_mode)
				self.check_bn()
				self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
				train_loader = iter(self.train_loader)
		self._finish()
	
	@torch.no_grad()
	def test_net(self, net, name=''):
		self.reset(isTrain=False, train_mode=False)
		batch_idx = 0
		test_length = self.cfg.data.test_size
		test_loader = iter(self.test_loader)
		while batch_idx < test_length:
			t1 = get_timepc()
			batch_idx += 1
			test_data = next(test_loader)
			self.set_input(test_data)
			self.forward(net)
			self.post_update()
			t2 = get_timepc()
			update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
			# ---------- log ----------
			if self.master:
				if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
					msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test ({name})'), self.master, None)
					log_msg(self.logger, msg)
		top1 = self.log_terms.get('top1_cnt').sum * 100. / min(self.cfg.data.test_length, self.log_terms.get('top_all').sum) if self.log_terms.get('top_all').sum > 0 else 0
		top5 = self.log_terms.get('top5_cnt').sum * 100. / min(self.cfg.data.test_length, self.log_terms.get('top_all').sum) if self.log_terms.get('top_all').sum > 0 else 0
		log_msg(self.logger, f'{name}: {top1:.3f} ({top5:.3f})')
		return top1, top5
	
	def test(self):
		tops = self.test_net(self.net, name='net')
		self.is_best = True if len(self.topk_recorder['net_top1']) == 0 or tops[0] > max(self.topk_recorder['net_top1']) else False
		self.topk_recorder['net_top1'].append(tops[0])
		self.topk_recorder['net_top5'].append(tops[1])
		max_top1 = max(self.topk_recorder['net_top1'])
		max_top1_idx = self.topk_recorder['net_top1'].index(max_top1) + 1
		msg = 'Max [top1: {:>3.3f} (epoch: {:d})]'.format(max_top1, max_top1_idx)
		if self.ema:
			tops = self.test_net(self.net_E, name='net_E')
			self.is_best_ema = True if len(self.topk_recorder['net_E_top1']) == 0 or tops[0] > max(self.topk_recorder['net_E_top1']) else False
			self.topk_recorder['net_E_top1'].append(tops[0])
			self.topk_recorder['net_E_top5'].append(tops[1])
			max_top1_ema = max(self.topk_recorder['net_E_top1'])
			max_top1_idx_ema = self.topk_recorder['net_E_top1'].index(max_top1_ema) + 1
			msg += ' [top1-ema: {:>3.3f} (epoch: {:d})]'.format(max_top1_ema, max_top1_idx_ema)
		log_msg(self.logger, msg)
	
	def test_ghost(self):
		for top_name in ['net_top1', 'net_top5', 'net_E_top1', 'net_E_top5']:
			self.topk_recorder[top_name].append(0)
			
	def save_checkpoint(self):
		if self.master and self.epoch % self.cfg.logging.train_save_model_epoch == 0:
			checkpoint_info = {'net': trans_state_dict(self.net.state_dict(), dist=False),
							   'net_E': trans_state_dict(self.net_E.state_dict(), dist=False) if self.net_E else None,
							   'optimizer': self.optim.state_dict(),
							   'scheduler': self.scheduler.state_dict(),
							   'scaler': self.loss_scaler.state_dict() if self.loss_scaler else None,
							   'iter': self.iter,
							   'epoch': self.epoch,
							   'topk_recorder': self.topk_recorder,
							   'total_time': self.cfg.total_time,
							   'nan_or_inf_cnt': self.nan_or_inf_cnt}
			save_checkpoint(checkpoint_info, is_best=self.is_best, is_best_ema=self.is_best_ema, log_dir=self.cfg.logdir, prefix='latest')
		self.is_best, self.is_best_ema = False, False
			
	def run(self):
		log_msg(self.logger, f'==> Starting {self.cfg.mode}ing with {self.cfg.nnodes} nodes x {self.cfg.ngpus_per_node} GPUs')
		self.train() if self.cfg.mode in ['train', 'ft'] else self.test_net(self.net, name='net')
