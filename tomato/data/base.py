# encoding: utf-8
# Author: Yixuan
# 
#

import os
import argparse

import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np

from tomato.utils import logger

class AudioDataset(Dataset):

    def __init__(self, 
                 args: argparse.Namespace,
                 split: str,
                 train_mode: bool = True):
        self.args = args
        self.split = split
        self.train_mode = train_mode
        # to locate to a specific batch when we want to 
        # recover from the last checkpoint
        self.fast_forward_mode = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
    
    