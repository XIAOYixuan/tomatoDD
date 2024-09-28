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

class OCBCELoss(BaseCriterion):

    def __init__(self, args):
        super().__init__(args)
        self.m_real = getattr(args, "m_real", 0.9)
        self.m_fake = getattr(args, "m_fake", 0.2)
        self.alpha = getattr(args, "alpha", 20.0)
        self.softplus = nn.Softplus()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, net_input: dict, net_output: dict, *kwargs):
        scores = net_output["feats"]
        pred = net_output["feats_out"]
        targ = net_input["labels"]

        # bce loss
        if pred.ndim == 2:
            pred = pred.squeeze(1)
        loss = self.bce(pred, targ.float())

        # oc loss
        negative_scores = scores[targ == 0]
        positive_scores = scores[targ == 1]
        #logger.info(f"negative_scores: {negative_scores.shape}")
        #logger.info(f"positive_scores: {positive_scores.shape}")

        if negative_scores.shape[0] != 0:
            #soft_scores = self.softplus(self.alpha * (self.m_real - negative_scores))
            #mean_scores = soft_scores.mean()
            #logger.info(f"soft_scores: {soft_scores.shape}")
            #logger.info(f"mean_scores: {mean_scores.shape}")
            loss = loss + self.softplus(self.alpha * (self.m_real - negative_scores)).mean()
        if positive_scores.shape[0] != 0:
            loss = loss + self.softplus(self.alpha * (positive_scores - self.m_fake)).mean()

        return loss

    def zero_logging_output(self):
        return {
            "loss": 0.0
        }

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


if __name__ == "__main__":
    from argparse import Namespace
    args = Namespace()
    ocbceloss = OCBCELoss(args)

    scores = torch.randn(4, 30)
    pred = torch.randn(4, 1)
    targ = torch.randint(0, 2, (4,))

    net_output = {
        "feats": scores,
        "feats_out": pred,
    }
    net_input = {
        "labels": targ,
    }

    loss = ocbceloss(net_input, net_output)
    print(loss)