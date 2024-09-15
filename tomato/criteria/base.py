# encoding: utf-8
# Author: Yixuan
# 
#
import argparse
import torch
import torch.nn as nn
import tomato 
import tomato.utils

class BaseCriterion(nn.Module):

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.criterion = None

    def forward(self, net_input, net_output, *kwargs):
        raise NotImplementedError
    
    def zero_logging_output(self):
        """
        Determine what needs to be logged in the tensorboard
        """
        raise NotImplementedError
    
    def load_checkpoint(self, ckpt_path):
        raise NotImplementedError("load_checkpoint() is not implemented")
    
    def change_weight(self, weight_pos, weight_neg):
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            weights = torch.tensor([weight_pos, weight_neg])
            weights = weights.to("cuda")
            self.criterion.weight = weights
        elif isinstance(self.criterion, nn.BCEWithLogitsLoss):
            pos_weight = torch.tensor([weight_pos])
            pos_weight = pos_weight.to("cuda")
            self.criterion.pos_weight = pos_weight 


class BuiltInCriterion(BaseCriterion):

    def __init__(self, args):
        super().__init__(args)
        #TODO: do we need a wrapper
        # to cuda
        if args.type == "BCEWithLogitsLoss":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            pos_weight, neg_weight = 0.5, 0.5
            if hasattr(args, "pos_weight") and hasattr(args, "neg_weight"):
                neg_weight, pos_weight = args.real_weight, args.fake_weight
            weight = torch.FloatTensor([neg_weight, pos_weight]).to("cuda:0")
            self.criterion = nn.CrossEntropyLoss(weight=weight)

    def set_criterion(self, criterion):
        self.criterion = criterion
    
    def forward(self, net_input, net_output, *kwargs):
        input = net_output["feats_out"] # logits
        target = net_input["labels"] # [bz]
        target = tomato.utils.move_to_cuda(target, input.device)
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            if input.ndim == 2:
                input = input.squeeze(1)
            return self.criterion(input, target.float())
        elif isinstance(self.criterion, nn.CrossEntropyLoss):
            return self.criterion(input, target)

    def zero_logging_output(self):
        return {
            "loss": 0.0
        }
    
    def train(self):
        self.criterion.train()

    def eval(self):
        self.criterion.eval()
