# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch.optim.lr_scheduler import _LRScheduler
from loguru import logger


class PolynomialDecayLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_updates,
        tot_updates,
        lr,
        end_lr,
        power,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_updates = int(warmup_updates)
        self.tot_updates = int(tot_updates)
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def step(self):
        self.optimizer.step()
        super().step()
        # logger.debug(f"lr: {self.get_lr()}")

    def get_lr(self):
        # warmup_updates 보다 작으면 시작 lr 고정
        # tot_updates 보다 크면 end_lr 고정
        # 그 중간지점은 범위 안에서 점차적으로 decay
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False
