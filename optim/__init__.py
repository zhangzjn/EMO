import torch
from torch import optim as optim
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.nadam import Nadam
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP
from timm.optim.lookahead import Lookahead


def check_keywords_in_name(name, keywords=()):
	isin = False
	for keyword in keywords:
		if keyword in name:
			isin = True
	return isin


def add_weight_decay(model, weight_decay=1e-5, skip_list=(), skip_keywords=()):
	decay = []
	no_decay = []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue  # frozen weights
		if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or check_keywords_in_name(name, skip_keywords):
			no_decay.append(param)
		else:
			decay.append(param)
	return [
		{'params': no_decay, 'weight_decay': 0.},
		{'params': decay, 'weight_decay': weight_decay}]


def get_optim(cfg, net, lr, betas=None, filter_bias_and_bn=True):
	optim_kwargs = {k: v for k, v in cfg.optim.optim_kwargs.items()}
	optim_split = optim_kwargs.pop('name').lower().split('_')
	optim_name = optim_split[-1]
	optim_lookahead = optim_split[0] if len(optim_split) == 2 else None
	
	if optim_kwargs.get('weight_decay', None) and filter_bias_and_bn:
		skip = {}
		skip_keywords = {}
		if hasattr(net, 'no_weight_decay'):
			skip = net.no_weight_decay()
		if hasattr(net, 'no_weight_decay_keywords'):
			skip_keywords = net.no_weight_decay_keywords()
		params = add_weight_decay(net, optim_kwargs['weight_decay'], skip, skip_keywords)
		optim_kwargs['weight_decay'] = 0.
	else:
		params = net.parameters()
		
	if optim_kwargs.get('betas', None) and betas:
		optim_kwargs['betas'] = betas
	
	optim_terms = {
		'sgd': optim.SGD,
		'adam': optim.Adam,
		'adamw': optim.AdamW,
		'adadelta': optim.Adadelta,
		'rmsprop': optim.RMSprop,
		'nadam': Nadam,
		'radam': RAdam,
		'adamp': AdamP,
		'sgdp': SGDP,
		'adafactor': Adafactor,
		'adahessian': Adahessian,
		'rmsproptf': RMSpropTF,
	}
	optimizer = optim_terms[optim_name](params, lr=lr, **optim_kwargs)
	if optim_lookahead:
		optimizer = Lookahead(optimizer)
	return optimizer
