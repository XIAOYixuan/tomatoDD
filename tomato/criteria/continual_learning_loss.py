# encoding: utf-8
# Author: Yixuan
# 
# 
from typing import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseCriterion
from tomato.utils import logger

class OCSoftmax(BaseCriterion):
    """
    Trainable criterion for one-class classification
    """

    def __init__(self, args):
        super().__init__(args)
        # TODO: make these parameters configurable
        self.feat_dim = 256
        if hasattr(args, "feat_dim"):
            self.feat_dim = args.feat_dim
        self.m_real = args.m_real
        self.m_fake = args.m_fake
        self.alpha = 20.0
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()
        # TODO: change cuda:0 to the assigned cuda device
        device = torch.device("cuda:0")
        self.device = device
        self.to(device)


    def load_checkpoint(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.to(self.device)
        logger.info(f"Loaded checkpoint from {ckpt_path}")


    def forward(self, net_input: dict, net_output: dict, *kwargs):
        """ 
        """
        feats = net_output['feats'] # [batch_size, feat_dim]
        #print("feats shape:", feats.shape)
        labels = net_input['labels'] # [batch_size]
        #print("labels shape:", labels.shape)

        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(feats, p=2, dim=1)
        # get the max and min for each sample
        # neg_feats = x[labels == 0]
        #print("max of neg_feats:", neg_feats.max(dim=1))
        #print("min of neg_feats:", neg_feats.min(dim=1))
        #print("max of w:", w.max(dim=1))
        #print("min of w:", w.min(dim=1))

        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()

        #TODO: no need to compute loss if it's not for training 
        negative_scores = scores[labels == 0]
        positive_scores = scores[labels == 1] 
        #print("negative_scores:", negative_scores)
        #print("after times alpha", self.alpha * (self.m_real - negative_scores))
        #print("positive_scores shape:", positive_scores.shape)
        loss = None
        if negative_scores.shape[0] != 0:
            loss = self.softplus(self.alpha * (self.m_real - negative_scores)).mean()
        if positive_scores.shape[0] != 0:
            positive_loss = self.softplus(self.alpha * (positive_scores - self.m_fake)).mean()
            loss = loss + positive_loss if loss is not None else positive_loss

        assert loss is not None
        # print("loss.item:", loss.item())

        logging_output = {
            "loss": loss.item(),
        }
        
        return loss, output_scores.squeeze(1), logging_output 
    
    def zero_logging_output(self):
        return {
            "loss": 0.0,
        }

