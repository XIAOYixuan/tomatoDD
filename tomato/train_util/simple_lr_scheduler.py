# encoding: utf-8
# Author: Yixuan
#
#
import torch
import numpy as np
from .base_scheduler import BaseLRScheduler
class StepLR(BaseLRScheduler):

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        self.gamma = args.lr_gamma
        self.step_size = args.lr_step_size
        self.lr = args.lr

    def get_last_lr(self):
        return self.lr

    def step(self, epoch_num):
        if epoch_num % self.step_size == 0:
            self.lr = self.lr * (self.gamma ** (epoch_num// self.step_size))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr


class CosineAnnealingLR(BaseLRScheduler):

    def __init__(self, args, total_steps, optimizer):
        super().__init__(args, optimizer)
        self.lr_min = args.lr_min 
        self.base_lr = args.lr 
        self.cur_epoch = 0

        def cosine_annealing(step, total_steps, lr_max, lr_min):
            return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  # since lr_lambda computes multiplicative factor
                self.lr_min / self.base_lr))

    def step(self, epoch_num):
        for i in range(self.cur_epoch, epoch_num):
            self.scheduler.step()
        self.cur_epoch = epoch_num


    def get_last_lr(self):
        #TODO: will need the list sometimes
        return self.scheduler.get_last_lr()[-1]