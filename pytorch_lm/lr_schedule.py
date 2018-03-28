#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""LR schedules."""

from bisect import bisect_right

from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR, ReduceLROnPlateau


class ConstantLR(_LRScheduler):
    """Keeps the learning rate constant."""
    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


class MultiScheduleLR(_LRScheduler):
    """
    Sets a different learning rate scheduler once the number of epochs reaches
    one of the milestones. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (dict): A dictionary of {milestone: scheduler}. All scheduler
                           objects must be associated with the same optimizer.
                           The first milestone should be 0.
        last_epoch (int): The index of the last (previous) epoch. Default: -1.
    """
    def __init__(self, optimizer, milestones, last_epoch=-1):
        if not (isinstance(milestones, dict) and dict):
            raise ValueError('Milestones should be non-empty dict of '
                             '{milestone: scheduler}. Got {}', milestones)
        for scheduler in milestones.values():
            if not isinstance(scheduler, _LRScheduler):
                raise TypeError('{} is not an _LRScheduler'.format(
                    type(scheduler).__name__))
            if scheduler.optimizer != optimizer:
                raise ValueError('All schedulers must be associated with the '
                                 'same Optimizer.')
        self.milestones, self.schedulers = zip(*sorted(milestones.items()))
        if self.milestones[0] > last_epoch + 1:
            raise ValueError('The first milestone must be less or equal '
                             'than last_epoch.')
        self.scheduler = None
        super(MultiScheduleLR, self).__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        super(MultiScheduleLR, self).step(epoch)
        scheduler = self.schedulers[
            bisect_right(self.milestones, self.last_epoch) - 1
        ]
        if scheduler != self.scheduler:
            self.scheduler = scheduler
        self.scheduler.step(epoch)

    def get_lr(self):
        milestone = bisect_right(self.milestones, self.last_epoch) - 1
        scheduler = self.schedulers[milestone]
        return scheduler.get_lr()


class ZarembaScheduleLR(MultiScheduleLR):
    """The Zaremba schedule."""
    def __init__(self, optimizer, lr_decay, decay_delay, last_epoch=-1):
        super(ZarembaScheduleLR, self).__init__(
            optimizer,
            {0: ConstantLR(optimizer),
             decay_delay: ExponentialLR(optimizer, lr_decay, last_epoch=0)}
        )


def lr_step_at_epoch_start(lr_scheduler):
    """
    Tells if the step() method should be invoked at the beginning of the
    epoch. True for all LR schedulers, with the only exception being
    ReduceLROnPlateau.
    """
    if isinstance(lr_scheduler, ReduceLROnPlateau):
        return False
    else:
        return True
