#!/usr/bin/env python3

"""Auxiliary functions for standard classes."""


def getall(d, keys, default=None):
    """
    Calls get() for all of the listed keys and returns an iterator to the
    values extracted from dict.
    """
    return [d.get(key, default) for key in keys]


def public_dict(obj):
    """Same as obj.__dict__, but without private fields."""
    return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
