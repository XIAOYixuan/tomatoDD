# encoding: utf-8
# Author: Yixuan
# 
# Adapted from fairseq

import argparse
import math

from .base_scheduler import BaseLRScheduler

class TriStageLrScheduler(BaseLRScheduler):

    def __init__(self, args: argparse.Namespace, optimizer):
        """
        - warmup stage: init_lr -> peak_lr
        - hold stage: peak_lr
        - decay stage: peak_lr -> final_lr

        lr calculation:
        - warmup stage:
            lrs = torch.linespace(init_lr, peak_lr, warmup_steps)
            lr = lrs[update_num]
        - hold stage:
            peak_lr
        - decay stage:
            decay_factor = - math.log(final_lr_scale) / decay_steps
            lr = peak_lr * exp(- (update_num - warmup_steps - decay_steps) * decay_factor)
        """
        super().__init__(args, optimizer)
        init_lr_scale = args.init_lr_scale
        final_lr_scale = args.final_lr_scale

        self.peak_lr = args.lr
        self.init_lr = init_lr_scale * args.lr
        self.final_lr = final_lr_scale * args.lr

        # set three steps
        self.warmup_steps = args.warmup_steps
        self.hold_steps = args.hold_steps
        self.decay_steps = args.decay_steps

        self.warmup_rate = (
            (self.peak_lr - self.init_lr) / self.warmup_steps
            if self.warmup_steps != 0
            else 0
        )
        self.decay_factor = -math.log(args.final_lr_scale) / args.decay_steps

        # initial learning rate
        self.lr = self.init_lr
        self._set_optim_lr(self.lr)

    
    def _decide_stage(self, update_step):
        """
        return stage, and the corresponding steps within the current stage
        """
        if update_step < self.warmup_steps:
            # warmup state
            return 0, update_step

        offset = self.warmup_steps

        if update_step < offset + self.hold_steps:
            # hold stage
            return 1, update_step - offset

        offset += self.hold_steps

        if update_step <= offset + self.decay_steps:
            # decay stage
            return 2, update_step - offset

        offset += self.decay_steps

        # still here ? constant lr stage
        return 3, update_step - offset
    
    def _set_optim_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
    
    def step(self, num_updates):
        """Update the learning rate after each update."""
        stage, steps_in_stage = self._decide_stage(num_updates)
        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self._set_optim_lr(self.lr)

        return self.lr   
