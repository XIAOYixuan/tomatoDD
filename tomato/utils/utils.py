# encoding: utf-8
# Author: Yixuan
# 
#

import argparse
import warnings
from typing import Callable

import torch
import torch.nn.functional as F
import yaml

from .log_config import logger

def check_key(args, key):
    """
    Used to check whether the key exists in the train_args
    """
    if not hasattr(args, key):
        raise ValueError(f"Key {key} does not exist in train_args")

def config2arg(config_file: str, entry: str):
    # TODO: mainly use argparse to parse config file?
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    if entry not in config:
        logger.warning(f"Entry {entry} not found in config file {config_file}")
        return None
    namespace = argparse.Namespace(**config[entry])
    return namespace

# taken from fairseq
def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """
    from tomato.models.operations import gelu, gelu_accurate
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        deprecation_warning(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate"
        )
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


# taken from fairseq
def deprecation_warning(message, stacklevel=3):
    # don't use DeprecationWarning, since it's ignored by default
    warnings.warn(message, stacklevel=stacklevel)


def post_process(sentence: str, symbol: str):
    # import ipdb; ipdb.set_trace()
    if symbol == 'bert_bpe_piece':
        # before_sentence = sentence
        sentence = (' ' + sentence).replace(' ##', '').lstrip()
        # print("In post_process:\nBefore sentence: {}\nAfter sentence: {}".format(before_sentence, sentence))
    elif symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == "wordpiece":
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == "letter":
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol == 'word':
        sentence = sentence.replace(' ##', '').strip()
    elif symbol is not None and symbol != "none":
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    return sentence


def item(tensor):
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        return tensor.cuda(device=device, non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)                    


def set_seed(seed: int):
    import os
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def log_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.log_softmax(x.float(), dim=dim)
    else:
        return F.log_softmax(x, dim=dim, dtype=torch.float32)