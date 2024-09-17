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

from pathlib import Path
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

        # process data_path, if it's an absolute path, we use it
        # if it's a relative path, use the library path
        data_path = Path(args.data_path)
        if not data_path.is_absolute():
            data_path = Path(__file__).parent.parent.parent / data_path
            args.data_path = str(data_path)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError