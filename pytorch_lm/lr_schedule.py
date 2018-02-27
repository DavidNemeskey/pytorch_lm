#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""LR schedules."""


class LRSchedule:
    """
    The default LR schedule. Keeps the learning rate constant.

    Learning rate schedulers should implement the iterator protocol, and do a
    scheduling "step" (e.g. LR decay) on each call of next().
    """
    def __init__(self, lr):
        self.lr = lr
        self.orig_lr = lr

    def reset(self):
        self.lr = self.orig_lr

    def __iter__(self):
        return self

    def __next__(self):
        return self.lr

class ManualLRSchedule(LRSchedule):
    """
    Manual LR decay with an optional grace period. Used in e.g. Zaremba (2014).
    """
    def __init__(self, lr, lr_decay=0.9, decay_delay=0, max_steps=None):
        super(ManualLRSchedule, self).__init__(lr)
        self.lr_decay = lr_decay
        self.decay_delay = decay_delay
        self.max_steps = max_steps
        self.steps = 0

    def __next__(self):
        self.steps += 1
        if self.steps == self.max_steps:
            raise StopIteration()
        lr_decay = self.lr_decay ** max(self.steps - self.decay_delay, 0.0)
        self.lr = self.orig_lr * lr_decay
        return self.lr
