# encoding: utf-8
# Author: Yixuan
# 
# 
import torch
from .base import BaseCriterion

class HiFiDiscLoss(BaseCriterion):

    def __init__(self, args):
        super().__init__(args)
    
    def _disc_loss(self, d_out, label):
        loss = 0
        r_out = d_out[label == 0]
        g_out= d_out[label == 1]
        if r_out.shape[0] != 0:
            loss += torch.mean(r_out**2)
        if g_out.shape[0] != 0:
            loss += torch.mean((1-g_out)**2)
        return loss 

    def forward(self, net_input: dict, net_output: dict, *kwargs):
        mpd_out = net_output["mpd_out"]
        msd_out = net_output["msd_out"]
        labels = net_input["labels"]

        if labels.ndim == 2:
            labels = labels.squeeze(1)

        mpd_loss, msd_loss = 0, 0
        for d_out, label in zip(mpd_out, labels):
            mpd_loss += self._disc_loss(d_out, label) 

        for d_out, label in zip(msd_out, labels):
            msd_loss += self._disc_loss(d_out, label) 
        return mpd_loss + msd_loss 
    
    def zero_logging_output(self):
        return {
            "loss": 0.0
        }