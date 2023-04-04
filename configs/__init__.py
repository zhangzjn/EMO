import importlib
from argparse import Namespace
from ast import literal_eval
from util.net import get_timepc


def get_cfg(opt_terminal):
	opt_terminal.cfg_path = opt_terminal.cfg_path.split('.')[0].replace('/', '.')
	dataset_lib = importlib.import_module(opt_terminal.cfg_path)
	cfg_terms = dataset_lib.__dict__
	ks = list(cfg_terms.keys())
	for k in ks:
		if k.startswith('_'):
			del cfg_terms[k]
	cfg = Namespace(**dataset_lib.__dict__)
	for key, val in opt_terminal.__dict__.items():
		cfg.__setattr__(key, val)
	
	cfg.command = f'python3 -m torch.distributed.launch --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$master_port --use_env run.py -c {cfg.cfg_path} -m {cfg.mode} --sleep {cfg.sleep} --memory {cfg.memory} --dist_url {cfg.dist_url} --logger_rank {cfg.logger_rank} {" ".join(cfg.opts)}'
	for opt in cfg.opts:
		cfg_ghost = cfg
		ks, v = opt.split('=')
		ks = ks.split('.')
		try:
			v = literal_eval(v)
		except:
			v = v
		for i, k in enumerate(ks):
			if i == len(ks) - 1:
				if isinstance(cfg_ghost, dict):
					cfg_ghost[k] = v
				else:
					cfg_ghost.__setattr__(k, v)
			else:
				if k not in cfg_ghost:
					cfg_ghost.__setattr__(k, Namespace())
				cfg_ghost = cfg_ghost.__dict__[k]
	cfg.task_start_time = get_timepc()
	return cfg
