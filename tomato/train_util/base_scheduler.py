# encoding: utf-8
# Author: Yixuan
#
#

import argparse

class BaseLRScheduler(object):

    def __init__(self, args: argparse.Namespace, optimizer):
        self.args = args
        self.optimizer = optimizer

    def step(self, step_num):
        raise NotImplementedError