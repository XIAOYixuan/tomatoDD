# encoding: utf-8
# Author: Yixuan
# 
#
"""Utility file for src toolkit."""
import os
import random

import numpy as np
import torch

#TODO: make it configurable
WHISPER_MODEL_WEIGHTS_PATH = "tomato/models/assets/tiny_enc.en.pt"

def set_seed(seed: int):
    """Fix PRNG seed for reproducable experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
