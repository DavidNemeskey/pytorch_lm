#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Various dropout variants. These are used only for the H-to-H connections in the
custom RNN classes; between layers the regular torch.nn dropout modules are
used.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lm.locked_dropout import LockedDropout


class Dropout(nn.Module, ABC):
    """Base class for the two dropout variants."""
    def __init__(self, p=0):
        """
        Constructor.

        Args:
            p: the dropout probability. Must be between 0 and 1.
            per_sequence: if `False` (the default), each call to
            :func:`forward` generates a new noise mask. If `False`,
            :func:`make_noise` (or, preferably, :func:`new_sequence`)
            must be called by hand. This is useful for per-sequence masks.
        """
        super(Dropout, self).__init__()
        if p < 0 or 1 < p:
            raise ValueError('Dropout: p must be between 0 and 1')
        self.p = p

    def make_noise(self, x):
        """Generates the noise Variable."""
        noise = x.new().resize_as_(x)
        if self.p == 1:
            noise.fill_(0)
        else:
            noise.bernoulli_(1 - self.p).div_(1 - self.p)
        # print('noise', noise.size(), noise, flush=True)
        return noise

    def reset_noise(self):
        """
        Resets the noise mask, at e.g. the beginning of a sequence. The default
        implementation does nothing.
        """
        pass

    @abstractmethod
    def forward(self, x):
        """To be implemented by subclasses."""

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.p)


class StatelessDropout(Dropout):
    """The regular stateless dropout."""
    def forward(self, x):
        if self.training and self.p:
            output = torch.mul(x, self.make_noise(x))  # expand_as?
            return output
        else:
            return x


class FunctionalDropout(Dropout):
    """The regular stateless using the stock dropout function."""
    def forward(self, x):
        if self.training and self.p:
            output = F.dropout(x, self.p, training=True)
            # print('output', output.size(), output, flush=True)
            return output
        else:
            return x


class StatefulDropout(Dropout):
    """
    A new noise mask is not generated for every input; rather, the mask is
    generated whenever it is not available. The usual use-case is per-sequence
    dropout for RNNs, where :func:`reset_noise` is called before processing the
    sequence.

    Because this dropout class is stateful, it should not be used concurrently.
    """
    def reset_noise(self):
        """Resets the noise mask, at e.g. the beginning of a sequence."""
        self.noise = None

    def forward(self, x):
        if self.training and self.p:
            if self.noise is None:
                self.noise = self.make_noise(x)
            return torch.mul(x, self.noise)  # expand_as?
        else:
            return x


class NoDropout(Dropout):
    """Does nothing; i.e. no dropout."""
    def forward(self, x):
        return x


def split_dropout(do_value):
    """
    Handles both string and number dropout values, with or without the "s"
    suffix (see create_dropout).
    """
    do_str = str(do_value)
    if do_str.endswith('s'):
        s = True
        do_str = do_str[:-1]
    else:
        s = False
    return float(do_str), s


def create_hidden_dropout(do_value, default_none=False):
    """
    Creates a dropout object from a DO string (or any object whose __str__
    method returns a string of the right format). The format is "<p>(s)", where
    <p> is the drop (not keep!) probability, and the s suffix is optional and
    marks per-sequence (stateful) dropout.

    This function returns instances of the :class:`Dropout` subclasses defined
    in this module; use :func:`create_dropout` to create
    :class:`torch.nn.Dropout` or :class:`LockedDropout` objects.

    If do_value evaluates to False, the return value depends on the default_none
    argument. If it is False (the default), a :class:`NoDropout` object is
    returned; otherwise, None.
    """
    if do_value:
        p, s = split_dropout(do_value)
        cls = StatefulDropout if s else StatelessDropout
        if p > 0:
            return cls(p)
    return NoDropout() if not default_none else None


def create_dropout(do_value, default_none=False):
    """
    Creates a dropout object from a DO string (or any object whose __str__
    method returns a string of the right format). The format is "<p>(s)", where
    <p> is the drop (not keep!) probability, and the s suffix is optional and
    marks per-sequence (stateful) dropout.

    This function returns instances of :class:`torch.nn.Dropout` or
    :class:`LockedDropout` objects; to create subclasses of the :class:`Dropout`
    class defined in this module; use :func:`create_hidden_dropout`.

    If do_value evaluates to False, the return value depends on the default_none
    argument. If it is False (the default), a regular :class:`nn.Dropout`
    object is returned; otherwise, None.
    """
    print('DO VALUE', do_value)
    if do_value:
        p, s = split_dropout(do_value)
        return (LockedDropout if s else nn.Dropout)(p)
    else:
        return nn.Dropout(0) if not default_none else None
