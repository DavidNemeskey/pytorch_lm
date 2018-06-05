#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""BPTT (length) related classes."""

from abc import ABC, abstractmethod
import random

from pytorch_lm.utils.config import create_object


class NumSteps(ABC):
    """
    Object that can be queried to get the number of BPTT steps in an iteration.
    """
    def __init__(self, num_steps):
        self.len = num_steps

    @abstractmethod
    def num_steps(self):
        """
        Returns a tuple: the number of steps, and the learning rate multiplier
        (only relevant for RandomNumSteps).
        """


class FixNumSteps(NumSteps):
    """The default behavior: just an int wrapper."""
    def num_steps(self):
        return self.len, 1


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
        full_len = round(min(max(5, full_len), self.len + 10))
        return full_len, full_len / self.len


def create_num_steps(num_steps):
    """Creates a NumSteps object from a number or dict."""
    if isinstance(num_steps, dict):
        return create_object(num_steps, base_module='pytorch_lm.bptt')
    else:
        return FixNumSteps(int(num_steps))
