import glob
import importlib

import torch
import torch.nn as nn
from timm.models.registry import is_model_in_modules
from timm.models.helpers import load_checkpoint
from timm.models.layers import set_layer_config
from timm.models.hub import load_model_config_from_hf
from timm.models.factory import parse_model_name
from util.registry import Registry
MODEL = Registry('Model')

files = glob.glob('model/[!_]*.py')
for file in files:
	model_lib = importlib.import_module(file.split('.')[0].replace('/', '.'))


def get_model(cfg_model):
	model_name = cfg_model.name
	model_kwargs = {k: v for k, v in cfg_model.model_kwargs.items()}
	model_fn = MODEL.get_module(model_name)
	checkpoint_path = model_kwargs.pop('checkpoint_path')
	ema = model_kwargs.pop('ema')
	strict = model_kwargs.pop('strict')
	pretrained = model_kwargs.pop('pretrained')
	
	if model_name.startswith('timm_'):
		model_source, model_name = parse_model_name(model_name)
		pretrained_cfg = None
		if model_source == 'hf-hub':
			pretrained_cfg, model_name = load_model_config_from_hf(model_name)
		with set_layer_config(scriptable=None, exportable=None, no_jit=None):
			model = model_fn(pretrained=pretrained, pretrained_cfg=pretrained_cfg, **model_kwargs)
		if not pretrained and checkpoint_path:
			load_checkpoint(model, checkpoint_path)
	else:
		model = model_fn(**model_kwargs)
		if checkpoint_path:
			ckpt = torch.load(checkpoint_path, map_location='cpu')
			if 'net' in ckpt.keys() or 'net_E' in ckpt.keys():
				state_dict = ckpt['net_E' if ema else 'net']
			else:
				state_dict = ckpt
			if not strict:
				no_ft_keywords = model.no_ft_keywords()
				for no_ft_keyword in no_ft_keywords:
					del state_dict[no_ft_keyword]
				ft_head_keywords, num_classes = model.ft_head_keywords()
				for ft_head_keyword in ft_head_keywords:
					if state_dict[ft_head_keyword].shape[0] < num_classes:
						del state_dict[ft_head_keyword]
					elif state_dict[ft_head_keyword].shape[0] == num_classes:
						continue
					else:
						state_dict[ft_head_keyword] = state_dict[ft_head_keyword][:num_classes, :] if 'weight' in ft_head_keyword else state_dict[ft_head_keyword][:num_classes]
			if isinstance(model, nn.Module):
				model.load_state_dict(state_dict, strict=strict)
			else:
				for sub_model_name, sub_state_dict in state_dict.items():
					sub_model = getattr(model, sub_model_name, None)
					sub_model.load_state_dict(sub_state_dict, strict=strict) if sub_model else None
	return model
