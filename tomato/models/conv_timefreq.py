# encoding: utf-8
# Author: Yixuan
# 
#
from argparse import Namespace
import torch
import torch.nn as nn
from einops import rearrange

from tomato.utils import logger
from .base import ClassificationBase

class FreqBinConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_freq, patch_freq,
                 stride=1, padding=0, dilation=1, bias=True):
        super(FreqBinConv2d, self).__init__()
        if n_freq % patch_freq != 0:
            raise ValueError("n_freq must be divisible by patch_freq")
        self.num_patches = n_freq // patch_freq 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size 
        self.n_freq = n_freq
        self.patch_freq = patch_freq
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.conv = nn.Conv2d(
            in_channels * self.num_patches,
            out_channels * self.num_patches,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels * self.num_patches,
            bias=bias
        )

    def forward(self, x):
        n, c, f, t = x.size()
        # reshape x so that it can be processed by grouped conv2d
        x = rearrange(x, 'n c (n1 f1) t -> n (c n1) f1 t', n1=self.num_patches, f1=self.patch_freq) 
        #print("after rearrange", x.shape)
        out = self.conv(x)  # Shape: (batch_size, out_channels * num_patches, new_time_steps, patch_size)
        #print("output", out.shape)
        out = rearrange(out, 'n (c n1) f1 t -> n c (n1 f1) t', c=self.out_channels, n1=self.num_patches)
        #print("final output", out.shape)
        return out


class ConvFreqTime(ClassificationBase):

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.freq_bin_channels = getattr(args, 'freq_bin_channels', 4)
        padding = (0, 1)
        freq_scale = [2, 4, 8]
        self.conv1 = FreqBinConv2d(in_channels=1, 
                       out_channels=self.freq_bin_channels, 
                       kernel_size=(freq_scale[0], 3),
                       stride=(1, 1),
                       padding=padding,
                       n_freq=1024,
                       patch_freq=4,
                       bias=False)
        self.conv2 = FreqBinConv2d(in_channels=1, 
                       out_channels=self.freq_bin_channels, 
                       kernel_size=(freq_scale[1], 3),
                       stride=(1, 1),
                       padding=padding,
                       n_freq=1024,
                       patch_freq=8,
                       bias=False)
        self.conv3 = FreqBinConv2d(in_channels=1, 
                       out_channels=self.freq_bin_channels, 
                       kernel_size=(freq_scale[2], 3),
                       stride=(1, 1),
                       padding=padding,
                       n_freq=1024,
                       patch_freq=16,
                       bias=False)
        self.freq_bn = nn.BatchNorm2d(self.freq_bin_channels)

        kernal_size1, kernal_size2, kernal_size3 = (1, 4), (1, 8), (1, 16)
        padding1 = (1, kernal_size1[1] // 4)
        padding2 = (1, kernal_size2[1] // 4)
        padding3 = (1, kernal_size3[1] // 4)

        time_channels = self.freq_bin_channels
        time_group = True
        self.time1 = nn.Conv2d(in_channels=time_channels,
            out_channels=time_channels,
            kernel_size=kernal_size1,
            stride=(1, kernal_size1[1] // 2),
            padding=padding1,
            dilation=(1, 1),
            groups=time_channels if time_group else 1,
            bias=False)
        self.time2 = nn.Conv2d(in_channels=time_channels,
            out_channels=time_channels,
            kernel_size=kernal_size2,
            stride=(1, kernal_size2[1] // 2),
            padding=padding2,
            dilation=(1, 1),
            groups=time_channels if time_group else 1,
            bias=False)
        self.time3 = nn.Conv2d(in_channels=time_channels,
            out_channels=time_channels,
            kernel_size=kernal_size3,
            stride=(1, kernal_size3[1] // 2),
            padding=padding3,
            dilation=(1, 1),
            groups=time_channels if time_group else 1,
            bias=False)
        self.time_bn = nn.BatchNorm2d(time_channels)

        decision_channels = 8
        decision_group = False 

        self.dconv1 = nn.Sequential(
            nn.Conv2d(in_channels=time_channels,
                  out_channels=decision_channels,
                  kernel_size=(5, 5),
                  stride=2,
                  groups=decision_channels if decision_group else 1,
                  bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(decision_channels),
            nn.MaxPool2d(kernel_size=(4, 4))
        )
        
        self.dconv2 = nn.Sequential(
            nn.Conv2d(in_channels=decision_channels,
                  out_channels=decision_channels,
                  kernel_size=(5, 5),
                  groups=decision_channels if decision_group else 1,
                  bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(decision_channels),
            nn.MaxPool2d(kernel_size=(4, 4))
        )

        self.fc1 = nn.Linear(48, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(decision_channels, self.num_classes)

        self.to(self.device)

    def forward(self, source: dict, **kwargs) -> dict:
        super().forward(source, **kwargs)
        x = source['feats']
        # logger.info(f"Input features shape: {x.shape}")

        x1 = self.conv1(x)
        # logger.info(f"Shape after conv1: {x1.shape}")
        x2 = self.conv2(x)
        # logger.info(f"Shape after conv2: {x2.shape}")
        x3 = self.conv3(x)
        # logger.info(f"Shape after conv3: {x3.shape}")
        x = torch.cat([x1, x2, x3], dim=2)
        # logger.info(f"Shape after concatenating conv layers: {x.shape}")
        x = self.freq_bn(x)
        # logger.info(f"Shape after frequency batch normalization: {x.shape}")

        x1 = self.time1(x)
        # logger.info(f"Shape after time1: {x1.shape}")
        x2 = self.time2(x)
        # logger.info(f"Shape after time2: {x2.shape}")
        x3 = self.time3(x)
        # logger.info(f"Shape after time3: {x3.shape}")
        x = torch.cat([x1, x2, x3], dim=3)
        # logger.info(f"Shape after concatenating time layers: {x.shape}")
        x = self.time_bn(x)
        # logger.info(f"Shape after time batch normalization: {x.shape}")

        x = self.dconv1(x)
        # logger.info(f"Shape after dconv1: {x.shape}")
        x = self.dconv2(x)
        # logger.info(f"Shape after dconv2: {x.shape}")
        x = rearrange(x, 'n c f t -> n c (f t)')
        # logger.info(f"Shape after rearranging: {x.shape}")
        x = self.fc1(x)
        # logger.info(f"Shape after fc1: {x.shape}")
        feats = self.leaky_relu(x)
        # logger.info(f"Shape after leaky_relu: {feats.shape}")
        feats = feats.squeeze(-1)
        # logger.info(f"Shape after squeezing: {feats.shape}")
        feats_out = self.fc2(feats)
        # logger.info(f"Shape after fc2: {feats_out.shape}")

        return {
            'feats': feats,
            'feats_out': feats_out
        }
    
if __name__ == "__main__":
    #n, c, f, t = 4, 1, 1024, 200
    #x = torch.randn(n, c, f, t)
    n, c, l = 4, 1, 64320
    x = torch.randn(n, c, l)
    args = Namespace(
        cuda=0,
        num_classes=1,
        frontend='XLSR'
    )
    model = ConvFreqTime(args)
    source = {
        "feats": x
    }
    out = model(source)
    feat, feat_out = out["feats"], out["feats_out"]
    print(feat.shape)
    print(feat_out.shape)

    # print model num of parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params}")
