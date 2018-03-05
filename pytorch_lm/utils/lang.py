#!/usr/bin/env python3

"""Auxiliary functions for standard classes."""


def getall(d, keys, default=None):
    """
    Calls get() for all of the listed keys and returns an iterator to the
    values extracted from dict.
    """
    return [d.get(key, default) for key in keys]
