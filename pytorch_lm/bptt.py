#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""BPTT (length) related classes."""

import random

from pytorch_lm.utils.config import create_object


class NumSteps(object):
    """The default behavior: just an int wrapper."""
    def __init__(self, num_steps):
        self.len = num_steps

    def num_steps(self):
        return self.len


class RandomNumSteps(NumSteps):
    """Random BPTT sequence lengths, a la Merity et al. (2018)."""
    def __init__(self, num_steps, p, s):
        super(RandomNumSteps, self).__init__(num_steps)
        self.p = p
        self.s = s

    def num_steps(self):
        base_len = self.len if random.random() <= self.p else self.len // 2
        full_len = random.gauss(base_len, self.s)
        # Safeguard for the sequence being too short or long
        full_len = min(max(5, full_len), self.len + 10)


def create_num_steps(num_steps):
    """Creates a NumSteps object from a number or dict."""
    if isinstance(num_steps, dict):
        return create_object(num_steps, base_module='pytorch_lm.bptt')
    else:
        return NumSteps(int(num_steps))
