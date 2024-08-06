import inspect
import re
import sys
import types
import typing
from dataclasses import dataclass
from pathlib import Path

import tomli

__all__ = ['Optional', 'make_schema_from_method', 'typecheck_schema', 'load_configs']

int_regex = re.compile(r'[+-]?\d+')
float_regex = re.compile(r'[+-]?\d+\.\d*')
help_flags = {'--help', '-h', '/h', '/?', 'help'}


@dataclass
class Optional:
    type: type
    default: object


class Invalid:
    def __str__(self):
        return '<MISSING>'


def make_schema_from_method(method):
    signature = inspect.getfullargspec(method)
    hyperparameters = signature.kwonlyargs[1:]
    schema = {'model_init': Optional(signature.annotations['model_init'], signature.kwonlydefaults['model_init'])}
    if len(hyperparameters) > 0:
        schema['hypers'] = {}
        for name in hyperparameters:
            arg_type = signature.annotations[name]
            if name not in signature.kwonlydefaults:
                schema['hypers'][name] = arg_type
            else:
                schema['hypers'][name] = Optional(arg_type, signature.kwonlydefaults[name])
    return schema


def typecheck_value(v: object, t):
    if isinstance(t, type):
        if not isinstance(v, t):
            return f'{type(v)} instead of {t}'
        else:
            return True
    elif isinstance(v, Invalid):
        if isinstance(t, Optional):
            return True
        else:
            return f'is missing.'
    elif isinstance(t, Optional):
        return typecheck_value(v, t.type)
    elif typing.get_origin(t) is typing.Literal:
        if v not in typing.get_args(t):
            return f'{v!r} instead of any of {list(typing.get_args(t))}'
        else:
            return True
    elif typing.get_origin(t) is types.UnionType or typing.get_origin(t) is typing.Union:
        sub_types = typing.get_args(t)
        for sub_t in sub_types:
            if typecheck_value(v, sub_t) is True:
                return True
        return f'{type(v)!r} instead of any of {list(sub_types)}'
    else:
        raise NotImplementedError(f'Unknown type: {t!r}')


def typecheck_schema(data_dict, schema, root='', ignore_extra=False):
    extra_keys = set(data_dict.keys()).difference(schema.keys())
    if not ignore_extra and len(extra_keys) > 0:
        raise IndexError(f'Unknown keys: {root}.[{", ".join(extra_keys)}]')
    for k, t in schema.items():
        v = data_dict.get(k, Invalid())
        if isinstance(t, dict):
            if not isinstance(v, Invalid):
                data_dict[k] = typecheck_schema(v, t, root=f'{root}.{k}', ignore_extra=ignore_extra)
            else:
                raise TypeError(f'{root}.{k} is missing.')
        elif isinstance(v, Invalid) and isinstance(t, Optional):
            data_dict[k] = t.default
        elif typecheck_value(v, t) is not True:
            raise TypeError(f'{root}.{k} is {typecheck_value(v, t)}')
    return data_dict


def parse_command_line(arg):
    args = arg.split('=', maxsplit=1)
    if len(args) < 2:
        raise TypeError(f'Invalid argument {arg!r}')

    value = args[1]
    if value == 'True' or value == 'true':
        value = True
    elif value == 'False' or value == 'false':
        value = False
    elif int_regex.fullmatch(value):
        value = int(value)
    elif float_regex.fullmatch(value):
        value = float(value)

    config = {}
    keys = args[0].split('.')
    c = config
    for k in keys[:-1]:
        c[k] = {}
        c = c[k]
    c[keys[-1]] = value
    return config


def deep_update(destination: dict, source: dict):
    for k, v_source in source.items():
        v_destination = destination.get(k)
        if not isinstance(v_destination, dict) or not isinstance(v_source, dict):
            destination[k] = v_source
        else:
            deep_update(v_destination, v_source)


def parse_toml(path):
    config = tomli.loads(path.read_text())
    if '@overlay' in config:
        assert isinstance(config['@overlay'], str)
        overlay = parse_toml((path.parent / config['@overlay']).resolve())
        deep_update(config, overlay)
        del config['@overlay']
    return config


def load_configs(args):
    if any(arg in help_flags for arg in args):
        print('TODO')
        sys.exit(1)
    config = {}
    configs = [
        parse_toml(Path(arg).resolve()) if arg[0] != '@' else parse_command_line(arg[1:])
        for arg in args
    ]
    for new_config in configs:
        deep_update(config, new_config)
    return config
