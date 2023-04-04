from timm.models.registry import _model_entrypoints
from . import MODEL


for timm_name, timm_fn in _model_entrypoints.items():
	MODEL.register_module(timm_fn, f'timm_{timm_name}')


if __name__ == '__main__':
	print()
	