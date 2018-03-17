#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Various dropout variants."""

import torch
import torch.nn as nn


class Dropout(nn.Module):
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
        return noise

    def reset_noise(self):
        """Resets the noise mask, at e.g. the beginning of a sequence."""
        raise NotImplementedError('reset_noise not implemented in {}'.format(
            self.__class__.__name__))

    def forward(self, x):
        raise NotImplementedError('forward not implemented in {}'.format(
            self.__class__.__name__))


class StatelessDropout(Dropout):
    """The regular stateless dropout."""
    def reset_noise(self):
        """A no-op."""
        pass

    def forward(self, x):
        if self.training and self.p:
            return torch.mul(x, self.make_noise(x))  # expand_as?
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
                self.make_noise(x)
            return torch.mul(x, self.noise)  # expand_as?
        else:
            return x
