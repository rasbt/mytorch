# Sebastian Raschka 2018
# mytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class CyclicalLearningRate(_LRScheduler):

    """Learning rate scheduler based on the cyclical
    learning rate concept introduced in
        Leslie N. Smith
        "Cyclical learning rates for training neural networks."
        Applications of Computer Vision (WACV),
        2017 IEEE Winter Conference on. IEEE, 2017.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        max_lr (float): Maximum learning rate.
        mode (str): `'triangular'`, `'triangular2'`, or
            `'exp_range'` mode. Default: `'triangular'`
        gamma (float): Multiplicative factor of learning rate decay
            if mode=`exp_range`. Default: 0.999995.
        batch_count (int): The index of the most recent batch.
            Default: -1.

    Example:
        >>> num_epochs = 50
        >>> train_size = 50000
        >>> batch_size = 100
        >>> iterations_per_epoch = train_size // batch_size
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        >>> scheduler = CyclicalLearningRate(
        ...     optimizer,
        ...     step_size=step_size,
        ...     max_lr=0.06)
        >>>
        >>> for epoch in range(num_epochs):
        >>>     for batch in range(iterations_per_epoch):
        >>>         # train(...)
        >>>         # validate(...)
        >>>         # note that the scheduler should be called
        >>>         # after each batch (not only after each epoch)
        >>>         scheduler.step()
    """

    def __init__(self,
                 optimizer,
                 step_size,
                 max_lr,
                 mode='triangular',
                 gamma=0.999995,
                 batch_count=-1):

        self.step_size = step_size
        self.max_lr = max_lr
        self.mode = mode
        self.gamma = gamma
        self.batch_count = batch_count

        if self.batch_count == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when "
                                   "resuming an optimizer".format(i))

        self.base_lrs = list(map(lambda group: group['initial_lr'],
                             optimizer.param_groups))
        super(CyclicalLearningRate, self).__init__(optimizer)

    def _compute_lr(self, base_lr):
        cycle = np.floor(1 + self.batch_count / (2. * self.step_size))
        x = np.abs(self.batch_count / float(self.step_size) - 2 * cycle + 1)

        lr_delta = (self.max_lr - base_lr) * np.maximum(0, (1 - x))

        if self.mode == 'triangular':
            pass
        elif self.mode == 'triangular2':
            lr_delta = lr_delta * 1 / (2. ** (cycle - 1))
        elif self.mode == 'exp_range':
            lr_delta = lr_delta * (self.gamma**(self.batch_count))
        else:
            raise ValueError('mode must be "triangular", '
                             '"triangular2", or "exp_range"')

        return base_lr + lr_delta

    def step(self, epoch=None):
        self.batch_count += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        return [self._compute_lr(lr) for lr in self.base_lrs]
