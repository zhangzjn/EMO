from torchvision import transforms
import numpy as np
from . import TRANSFORMS


def get_transforms(cfg, train, cfg_transforms):
	transform_list = []
	for t in cfg_transforms:
		t = {k: v for k, v in t.items()}
		t_type = t.pop('type')
		t_tran = TRANSFORMS.get_module(t_type)(**t)
		transform_list.extend(t_tran) if isinstance(t_tran, list) else transform_list.append(t_tran)
	transform_out = TRANSFORMS.get_module('Compose')(transform_list)
	
	if train:
		if cfg.size <= 32 and cfg.type == 'CLS':
			transform_out[0] = transforms.RandomCrop(cfg.size, padding=4)
	return transform_out


def make_divisible(v, divisor=8, min_value=None):
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v


def get_scales(n_scale, base_h, base_w, min_h, max_h, min_w, max_w, check_scale_div_factor=32):
	hs = list(np.linspace(min_h, max_h, n_scale))
	if base_h not in hs:
		hs.append(base_h)
	ws = list(np.linspace(min_w, max_w, n_scale))
	if base_w not in ws:
		ws.append(base_w)
	scales = set()
	for h, w in zip(hs, ws):
		h = make_divisible(h, check_scale_div_factor)
		w = make_divisible(w, check_scale_div_factor)
		scales.add((h, w))
	scales = list(scales)
	return scales
