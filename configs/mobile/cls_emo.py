from argparse import Namespace as _Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN as _IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD as _IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as _F

# =========> shared <=================================
seed = 42
size = 224
type = 'CLS'
# =========> dataset <=================================
data = _Namespace()
data.name = 'DefaultCLS'
data.root = '/youtu_fuxi_team1_ceph/vtzhang/codes/data/imagenet'
data.loader_type = 'pil'
data.sampler = 'naive'
data.nb_classes = 1000

data.train_transforms = [
	dict(type='timm_create_transform', input_size=size, is_training=True, color_jitter=0.4,
		 auto_augment='rand-m9-mstd0.5-inc1', interpolation='random', mean=_IMAGENET_DEFAULT_MEAN, std=_IMAGENET_DEFAULT_STD,
		 re_prob=0.0, re_mode='pixel', re_count=1),
]
data.test_transforms = [
	dict(type='Resize', size=int(size / 0.875), interpolation=_F.InterpolationMode.BICUBIC),
	dict(type='CenterCrop', size=size),
	dict(type='ToTensor'),
	dict(type='Normalize', mean=_IMAGENET_DEFAULT_MEAN, std=_IMAGENET_DEFAULT_STD, inplace=True),
]

# =========> model <=================================
model = _Namespace()
model.name = 'EMO_1M'
model.model_kwargs = dict(pretrained=False, checkpoint_path='', ema=False, strict=True, num_classes=data.nb_classes)

# =========> optimizer <=================================
optim = _Namespace()
optim.lr = 1.5e-3
optim.optim_kwargs = dict(name='adamw', betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05, amsgrad=False)
# =========> trainer <=================================
trainer = _Namespace()
trainer.name = 'CLS'
trainer.checkpoint = 'runs_emo'
trainer.resume_dir = ''
trainer.cuda_deterministic = False
trainer.epoch_full = 300
trainer.scheduler_kwargs = dict(
	name='cosine', lr_noise=None, noise_pct=0.67, noise_std=1.0, noise_seed=42, lr_min=1e-6 / 4,
	warmup_lr=1e-6 / 4, warmup_iters=-1, cooldown_iters=0, warmup_epochs=20, cooldown_epochs=0, use_iters=True,
	patience_iters=0, patience_epochs=0, decay_iters=0, decay_epochs=0, cycle_decay=0.1,)

trainer.data = _Namespace()
trainer.data.batch_size = 2048
trainer.data.batch_size_per_gpu = None
trainer.data.batch_size_test = None
trainer.data.batch_size_per_gpu_test = 125
trainer.data.num_workers_per_gpu = 8
trainer.data.drop_last = True
trainer.data.pin_memory = True
trainer.data.persistent_workers = False

trainer.mixup_kwargs = dict(
	mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=0.0, switch_prob=0.5,
	mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=data.nb_classes)

trainer.scale_kwargs = dict(n_scale=0, base_h=size, base_w=size, min_h=160, max_h=320, min_w=160, max_w=320, check_scale_div_factor=32)

trainer.start_test_epoch = 200
trainer.every_test_epoch = 5
trainer.find_unused_parameters = False
trainer.sync_BN = 'none'  # [none, native, apex, timm]
trainer.dist_BN = '' # [ , reduce, broadcast], valid when sync_BN is 'none'
trainer.scaler = 'native'  # [none, native, apex]
trainer.ema = 0.9998  # [ None, 0.9998 ]

# =========> loss <=================================
tea_model = _Namespace()
tea_model.name = 'tresnet_l_v2'
tea_model.model_kwargs = dict(pretrained=False, checkpoint_path='pretrained/tresnet_l_v2_83_9.pth', ema=False, strict=True, num_classes=data.nb_classes)

loss = _Namespace()
loss.loss_terms = [
	dict(type='SoftTargetCE', name='CE', lam=1.0, fp32=True) if trainer.mixup_kwargs['prob'] > 0 else dict(type='LabelSmoothingCE', name='CE', lam=1.0, smoothing=trainer.mixup_kwargs['label_smoothing']),
]
loss.clip_grad = 5.0
loss.create_graph = False
loss.retain_graph = False

# =========> logging <=================================
logging = _Namespace()
logging.log_terms_train = [
	dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
	dict(name='data_t', fmt=':>5.3f'),
	dict(name='optim_t', fmt=':>5.3f'),
	dict(name='lr', fmt=':>7.6f'),
	dict(name='CE', fmt=':>5.3f', add_name='avg'),
	dict(name='KD', fmt=':>5.3f', add_name='avg'),
]
logging.log_terms_test = [
	dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
	dict(name='top1_cnt', fmt=':>6.0f', show_name='sum'),
	dict(name='top5_cnt', fmt=':>6.0f', show_name='sum'),
	dict(name='top_all', fmt=':>6.0f', show_name='sum'),
]
logging.train_reset_log_per = 50
logging.train_log_per = 50
logging.train_save_model_epoch = 1
logging.test_log_per = 50
