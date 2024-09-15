# encoding: utf-8
# Author: Yixuan
# 
#

import os
from datetime import datetime
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
    parser.add_argument("-exp", "--exp", type=str, default="infer", help="experiment name")
    parser.add_argument("-s", "--split", type=str, default="future", help="which split is used for infer")
    parser.add_argument("-tag", "--tag", type=str, required=True, help="output file name")
    parser.add_argument("-ckpt", "--ckpt_dir", type=str, default=None, help="checkpoint dir, the default is the exp dir")
    parser.add_argument("-ckpt_tag", "--ckpt_tag", type=str, default="best", help="load the best or the last model, default is best")
    parser.add_argument("-task", "--task", type=str, default="SVDDTask", help="task name")
    parser.add_argument("-o", "--output", type=str, default="./output/infer", help="output dir")
    parser.add_argument("-cuda", "--cuda", type=int, default=0, help="cuda device")
    parser.add_argument("-debug", "--debug", action="store_true", help="debug mode")

    return parser.parse_args()

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
    torch.manual_seed(335)
    
    data = main_loader.load_data(args.config, is_infer=True)
    model = main_loader.load_model(args.config, args.cuda)
    logger.info("Architecture: --------------------------------------------------")
    logger.info(model)
    # Don't use decoder for detection task
    # decoder = main_loader.load_decoder(model, args.exp)
    # TODO: use config to set decoder
    # decoder.log_to_local = False
    loss_fn = main_loader.load_criterion(args.config, model)

    import tomato.task
    task = tomato.task.get_task(args.task)
    task.setup(args.config, args.exp, data, model, loss_fn, is_infer=True)
    task.load_checkpoint(args.ckpt_tag, args.ckpt_dir)
    metrics = task.infer(args)
    for metric in metrics:
        print(f"{metric}: {metrics[metric]}")

if __name__ == "__main__":
    main()