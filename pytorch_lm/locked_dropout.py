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
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        """
        :param x: a 3D tensor, where the first is the time dimension
        """
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x
