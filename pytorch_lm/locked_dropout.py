#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
"Locked" (variational) dropout object copied from the
`awd-lstm-lm<https://github.com/salesforce/awd-lstm-lm/>`_.
"""

import torch.nn as nn
from torch.autograd import Variable

class LockedDropout(nn.Module):
    """Variational dropout object."""
    def __init__(self, dropout=0):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        """
        :param x: a 3D tensor, where the time dimension is the second
        """
        if not self.training or not self.dropout:
            return x
        m = x.data.new_empty((x.size(0), 1, x.size(2))).bernoulli_(1 - self.dropout)
        mask = Variable(m, requires_grad=False) / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x
