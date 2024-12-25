# encoding: utf-8
# Author: Yixuan
# 
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Conv1d
from torch.nn.utils import weight_norm, spectral_norm

# used to maintain the same lengths
def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class DiscriminatorP(torch.nn.Module):

    def __init__(self, period, kernel_size=5, stride=3):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))), 
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
    
    def forward(self, x):
        #print(f"Starting printing for {self.period}")
        b, c, t = x.shape
        if t % self.period != 0: # pad it to be divisible by period
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        #print(f"Input shape: {x.shape}")
        for idx, l in enumerate(self.convs):
            x = l(x)
            #print(f"Conv {idx} shape: {x.shape}")
            x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x) # （b, 1, T, period)
        #print(f"Conv post shape: {x.shape}")
        x = torch.flatten(x, 1, -1)
        #print(f"Flatten shape: {x.shape}")
        return x
                
class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y_hat):
        y_d_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_g = d(y_hat)
            y_d_gs.append(y_d_g)

        return y_d_gs

class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        for l in self.convs:
            x = l(x)
            #print(f"Conv shape: {x.shape}")
            x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        #print(f"Conv post shape: {x.shape}")
        x = torch.flatten(x, 1, -1)
        #print(f"Flatten shape: {x.shape}")
        return x

class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, y_hat):
        y_d_gs = []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                y_hat = self.meanpools[i-1](y_hat) # downsample
                #print(f"{i}th meanpool shape: {y_hat.shape}")
            y_d_g = d(y_hat)
            y_d_gs.append(y_d_g)
        return y_d_gs
    
class HiFiDiscriminator(torch.nn.Module):
    def __init__(self):
        super(HiFiDiscriminator, self).__init__()
        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator()
    
    def forward(self, y_hat):
        ms_out = self.msd(y_hat)
        mp_out = self.mpd(y_hat)
        return ms_out, mp_out
    