#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Various dropout variants."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dropout(nn.Module):
    """The basic dropout version."""
    def __init__(self, p=0, noise_per_call=True):
        """
        Constructor.

        Args:
            p: the dropout probability. Must be between 0 and 1.
            noise_per_call: if `True` (the default), each call to
                :func:`forward` generates a new noise mask. If `False`,
                :func:`make_noise` must be called by hand. This is useful
                for per-sequence masks.
        """
        super(Dropout, self).__init__()
        if p < 0 or 1 < p:
            raise ValueError('Dropout: p must be between 0 and 1')
        self.p = p
        self.noise_per_call = noise_per_call

    def make_noise(self, x):
        """Generates the noise Variable."""
        noise = x.new().resize_as_(x)
        if self.p == 1:
            noise.fill_(0)
        else:
            noise.bernoulli_(1 - self.p).div_(1 - self.p)
        self.noise = noise

    def forward(self, x):
        if self.training:
            if self.noise_per_call:
                self.make_noise(x)
            return torch.mul(x, self.noise)  # expand_as?
        else:
            return x
