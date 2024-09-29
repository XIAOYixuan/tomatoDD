# encoding: utf-8
# Author: Yixuan
# 
#

import os
from datetime import datetime
from pathlib import Path
import argparse
import torch
import torch.nn as nn

from tomato.utils import utils, logger, main_loader

torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser(description="training script")
    # need config file
    # short cut "-c"
    parser.add_argument("-c", "--config", type=str, required=True, help="config file")
    parser.add_argument("-exp", "--exp", type=str, default=None, help="experiment name")
    parser.add_argument("-ckpt", "--ckpt", type=str, default=None, help="checkpoint path")
    parser.add_argument("-task", "--task", type=str, default="SVDDTask", help="task name")
    parser.add_argument("-debug", "--debug", action="store_true", help="debug mode")
    parser.add_argument("-cuda", "--cuda", type=int, default=0, help="cuda device")
    args = parser.parse_args()

    if args.exp is None:
        # use the parent directory name of the config file as exp
        config_path = Path(args.config)
        args.exp = config_path.parent.name
        logger.info(f"exp name is not specified, use {args.exp} as exp name")
    return args

def test_model_parameters(model):
    # check whether these parameters are trainable
    trainables = []
    not_trainables = []
    for name, param in model.named_parameters():
        # save the name 
        if param.requires_grad:
            trainables.append(name)
        else:
            not_trainables.append(name)

    logger.info("Trainable parameters: --------------------------------------------------") 
    for param in trainables:
        logger.info(param)
    logger.info("Not trainable parameters: --------------------------------------------------")
    for param in not_trainables:
        logger.info(param)
    logger.info("Architecture: --------------------------------------------------")
    logger.info(model)
    exit(0)
    
def main():
    args = parse_args()
    utils.set_seed(42)
    model = main_loader.load_model(args.config, args.cuda)
    logger.info("Architecture: --------------------------------------------------")
    logger.info(model)
    # Don't use decoder for detection task
    # decoder = main_loader.load_decoder(model, args.exp)
    # TODO: use config to set decoder
    # decoder.log_to_local = False
    loss_fn = main_loader.load_criterion(args.config, model)
    data = main_loader.load_data(args.config)
    import tomato.task
    task = tomato.task.get_task(args.task)
    task.setup(args.config, args.exp, data, model, loss_fn)
    if args.ckpt is not None:
        task.load_checkpoint("best", args.ckpt)
    task.train()

if __name__ == "__main__":
    main()