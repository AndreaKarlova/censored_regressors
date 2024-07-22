import torch

__all__ = ['print_module', 'apply_module']


def rgetattr(o, k_list):
    for k in k_list:
        o = getattr(o, k)
    return o


def apply_module(func, module):
    with torch.no_grad():
        for name, _ in module.named_parameters():
            name = name.replace('raw_', '')
            param = rgetattr(module, name.split('.'))
            func(name, param)


def print_module(module):
    apply_module(lambda name, param: print(f'{name:35} {tuple(param.shape)}\n{param.numpy().round(10)}'), module)
