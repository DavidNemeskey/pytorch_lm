#!/usr/bin/env python3
"""Configuration-related functionality."""

from functools import partial
import importlib
import json
import os
from pkg_resources import resource_exists, resource_filename

from pytorch_lm.lr_schedule import ConstantLR


def get_config_file(config_file):
    """
    Returns the path to the configuration file specified. If there is a file at
    the path specified, it is returned as is; if not, the conf/ directory of the
    installed package is checked. If that fails as well, ValueError is raised.
    """
    if os.path.isfile(config_file):
        return config_file
    elif resource_exists('pytorch_lm.conf', config_file):
        return resource_filename('pytorch_lm.conf', config_file)
    else:
        raise ValueError('Could not find configuration file {}'.format(config_file))


def create_object(config, base_module=None, args=None, kwargs=None):
    """
    Creates an object from the specified configuration dictionary.
    Its format is:

        class: The fully qualified path name (but see below).
        args: A list of positional arguments (optional).
        kwargs: A dictionary of keyword arguments (optional).

    If base_module is specified, and class above does not contain any periods,
    then base_module.class will be loaded.

    The optional args and kwargs arguments are prepended to, and merged with,
    respectively, the ones from the configuration dictionary. These arguments
    can be used to supply arguments that would be difficult to serialize to
    JSON.
    """
    try:
        cls, args, kwargs = __clsfn_args_kwargs(config, 'class', base_module,
                                                args, kwargs)
        return cls(*args, **kwargs)
    except Exception as e:
        raise Exception(
            'Could not create object\n{}'.format(json.dumps(config, indent=4)),
            e
        )


def create_function(config, base_module=None, args=None, kwargs=None):
    """
    Creates a zero-parameter function from the specified configuration dictionary.
    Its format is:

        function: The fully qualified path name (but see below).
        args: A list of positional arguments (optional).
        kwargs: A dictionary of keyword arguments (optional).

    If base_module is specified, and function above does not contain any periods,
    then base_module.function will be loaded.

    The optional args and kwargs arguments are prepended to, and merged with,
    respectively, the ones from the configuration dictionary. These arguments
    can be used to supply arguments that would be difficult to serialize to
    JSON.
    """
    try:
        fun, args, kwargs = __clsfn_args_kwargs(config, 'function', base_module,
                                                args, kwargs)
        return partial(fun, *args, **kwargs)
    except Exception as e:
        raise Exception(
            'Could not create function\n{}'.format(json.dumps(config, indent=4)),
            e
        )


def __clsfn_args_kwargs(config, key, base_module=None, args=None, kwargs=None):
    """
    Utility function called by both create_object and create_function. It
    implements the code that is common to both.
    """
    args = args or []
    kwargs = kwargs or {}
    module_name, _, object_name = config[key].rpartition('.')
    if base_module and not module_name:
        module = importlib.import_module(base_module)
    else:
        module = importlib.import_module(module_name)
    obj = getattr(module, object_name)
    args += config.get('args', [])
    kwargs.update(**config.get('kwargs', {}))
    return obj, args, kwargs


def read_config(config_file, vocab_size):
    """
    Reads the configuration file, and creates the model, optimizer and
    learning rate schedule objects used by the training process.
    """
    with open(get_config_file(config_file)) as inf:
        config = json.load(inf)
    train = config.pop('train', {})
    valid = config.pop('valid', {})
    test = config.pop('test', {})

    # Copy all keys from the main dictionary to the sub-dictionaries
    for k, v in config.items():
        for d in [train, valid, test]:
            d[k] = v

    # Now for the model & stuff (train only)
    try:
        train['model'] = create_object(train['model'],
                                       base_module='pytorch_lm.model',
                                       args=[vocab_size])
    except KeyError:
        raise KeyError('Missing configuration key: "model".')
    try:
        train['optimizer'] = create_object(train['optimizer'],
                                           base_module='torch.optim',
                                           args=[train['model'].parameters()])
    except KeyError:
        raise KeyError('Missing configuration key: "optimizer".')
    try:
        train['initializer'] = create_function(train['initializer'],
                                               base_module='torch.nn.init')
    except KeyError:
        raise KeyError('Missing configuration key: "initializer".')
    if 'lr_scheduler' in config:
        train['lr_scheduler'] = create_object(config['lr_scheduler'],
                                              base_module='pytorch_lm.lr_schedule',
                                              args=[train['optimizer']])
    else:
        train['lr_scheduler'] = ConstantLR(train['optimizer'])
    return ({'train': train, 'valid': valid, 'test': test})
