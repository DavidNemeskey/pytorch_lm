#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Various dropout variants."""

import torch
import torch.nn as nn


class Dropout(nn.Module):
    """The basic dropout version."""
    def __init__(self, p=0, per_sequence=False):
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
        self.per_sequence = per_sequence
        self.noise = None

    def make_noise(self, x):
        """Generates the noise Variable."""
        noise = x.new().resize_as_(x)
        if self.p == 1:
            noise.fill_(0)
        else:
            noise.bernoulli_(1 - self.p).div_(1 - self.p)
        self.noise = noise

    def new_sequence(self):
        """To be called at the start of each new sequence."""
        self.noise = None

    def forward(self, x):
        if self.training:
            if self.noise is None or not self.per_sequence:
                self.make_noise(x)
            return torch.mul(x, self.noise)  # expand_as?
        else:
            return x
