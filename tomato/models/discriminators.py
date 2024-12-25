# encoding: utf-8
# Author: Yixuan
# 
#
from argparse import Namespace
import torch
import torch.nn as nn
from einops import rearrange

from tomato.utils import logger
from .predefined.hifigan import MultiPeriodDiscriminator, MultiScaleDiscriminator
from .base import ClassificationBase 

class HiFiDisc(ClassificationBase):

    def __init__(self, args: Namespace):
        super().__init__(args)
        if self.frontend is not None:
            raise ValueError("HiFiDisc does not support frontend model")
        logger.info("Doesn't support oc training for now")        
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

        self.to(self.device)

    def forward(self, source: dict, **kwargs) -> dict:
        super().forward_frontend(source)
        x = source['feats']
        mpd_out = self.mpd(x) # list of tensors, 5 different periods
        msd_out = self.msd(x) # list of tensors, 3 different scales

        # the feats_out is the mean of all the outputs
        feats_out = torch.mean(
            torch.stack(
                [torch.mean(out, dim=-1, keepdim=True) 
                    for out in mpd_out + msd_out], 
                dim=-1), dim=-1)


        return {
                "feats_out": feats_out, 
                "mpd_out": mpd_out, 
                "msd_out": msd_out
            }

if __name__ == '__main__':
    n, c, l = 4, 1, 64000
    x = torch.randn(n, c, l)
    args = Namespace(
        cuda=0,
        num_classes=1,
    )
    model = HiFiDisc(args)
    source = {"feats": x}
    out = model(source)

    for key in out.keys():
        for x in out[key]:
            print(key, x.shape)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")