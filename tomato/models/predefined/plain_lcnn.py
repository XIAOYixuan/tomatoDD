# encoding: utf-8
# 
# Adpated by Yixuan
# Adapted from https://github.com/yongyizang/SingFake/tree/main/models/lfcc%2Bresnet
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
from torch import Tensor
import os
import numpy as np

class MaxFeatureMap2D(nn.Module):
    """ Max feature map (along 2D)

    MaxFeatureMap2D(max_dim=1)

    l_conv2d = MaxFeatureMap2D(1)
    data_in = torch.rand([1, 4, 5, 5])
    data_out = l_conv2d(data_in)

    Input:
    ------
    data_in: tensor of shape (batch, channel, ...)

    Output:
    -------
    data_out: tensor of shape (batch, channel//2, ...)

    Note
    ----
    By default, Max-feature-map is on channel dimension,
    and maxout is used on (channel ...)
    """

    def __init__(self, max_dim=1):
        super(MaxFeatureMap2D, self).__init__()
        self.max_dim = max_dim

    def forward(self, inputs):
        # suppose inputs (batchsize, channel, length, dim)

        shape = list(inputs.size())

        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)

        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, i = inputs.view(*shape).max(self.max_dim)
        return m

class PlainLCNN(nn.Module):
    def __init__(self, enc_dim=256, feat_len=401, nclasses=2, F_len=-1, T_len=-1):
        super(PlainLCNN, self).__init__()
        self.enc_dim = enc_dim 
        self.nclasses = nclasses
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, (5, 5), 1, padding=(2, 2)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 96, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)),
                                   nn.BatchNorm2d(48, affine=False))
        self.conv4 = nn.Sequential(nn.Conv2d(48, 96, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(48, affine=False))
        self.conv5 = nn.Sequential(nn.Conv2d(48, 128, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)))
        self.conv6 = nn.Sequential(nn.Conv2d(64, 128, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(64, affine=False))
        self.conv7 = nn.Sequential(nn.Conv2d(64, 64, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv8 = nn.Sequential(nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv9 = nn.Sequential(nn.Conv2d(32, 64, (3, 3), 1, padding=[1, 1]),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 2), (2, 2)))
        self.out = nn.Sequential(nn.Dropout(0.7),
                                 nn.Linear((F_len // 16) * (T_len // 16) * 32, 160),
                                 MaxFeatureMap2D(),
                                 nn.Linear(80, self.enc_dim))
        self.fc_mu = nn.Linear(self.enc_dim, nclasses) if nclasses >= 2 else nn.Linear(self.enc_dim, 1)

    def forward(self, x):
        # NCFT
        #print("input", x.shape)
        x = self.conv1(x)
        #print("conv1", x.shape)
        x = self.conv2(x)
        #print("conv2", x.shape)
        x = self.conv3(x)
        #print("conv3", x.shape)
        x = self.conv4(x)
        #print("conv4", x.shape)
        x = self.conv5(x)
        #print("conv5", x.shape)
        x = self.conv6(x)
        #print("conv6", x.shape)
        x = self.conv7(x)
        #print("conv7", x.shape)
        x = self.conv8(x)
        #print("conv8", x.shape)
        x = self.conv9(x)
        feat = torch.flatten(x, 1)
        #print("flatten 9", feat.shape)
        #print("out w shape", self.out[1].weight.shape)
        feat = self.out(feat)
        out = self.fc_mu(feat)

        return feat, out
    

if __name__ == "__main__":
    import numpy as np
    # NCFT
    x = np.random.randn(2, 1, 401, 60).astype(np.float32)
    x = torch.from_numpy(x)
    model = PlainLCNN(enc_dim=256,
                      feat_len=401,
                      nclasses=2)
    feat, feat_out = model(x)
    print(feat.shape)
    print(feat_out.shape)