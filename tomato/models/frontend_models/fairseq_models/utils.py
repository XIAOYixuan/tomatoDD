# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Adapted by Yixuan
# fairseq.utils

import argparse
import collections
import contextlib
import copy
import importlib
import logging
import os
import sys
import warnings
from itertools import accumulate
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

def relu_squared(x: torch.Tensor):
    return F.relu(x).pow(2)


def deprecation_warning(message, stacklevel=3):
    # don't use DeprecationWarning, since it's ignored by default
    warnings.warn(message, stacklevel=stacklevel)


def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""
    from .modules import gelu, gelu_accurate

    if activation == "relu":
        return F.relu
    elif activation == "relu_squared":
        return relu_squared
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
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def buffered_arange(max, device="cpu"):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor().to(device)
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def is_xla_tensor(tensor):
    return torch.is_tensor(tensor) and tensor.device.type == "xla"


def index_put(tensor, indices, value):
    if is_xla_tensor(tensor):
        for _ in range(indices.dim(), tensor.dim()):
            indices = indices.unsqueeze(-1)
        if indices.size(-1) < tensor.size(-1):
            indices = indices.expand_as(tensor)
        tensor = torch.mul(tensor, ~indices) + torch.mul(value, indices)
    else:
        tensor[indices] = value
    return tensor


def eval_str_dict(x, type=dict):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    return x

def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)
