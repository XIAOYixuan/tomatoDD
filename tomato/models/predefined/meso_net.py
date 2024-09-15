"""
This code is modified version of MesoNet DeepFake detection solution
from FakeAVCeleb repository - https://github.com/DASH-Lab/FakeAVCeleb/blob/main/models/MesoNet.py.

Adapted by Yixuan
"""
import torch
import torch.nn as nn


class MesoInception4(nn.Module):
    """
    Pytorch Implemention of MesoInception4
    Author: Honggu Liu
    Date: July 7, 2019
    """
    def __init__(self, num_classes=1, **kwargs):
        super().__init__()
        self.use_old = False 

        self.fc1_dim = kwargs.get("fc1_dim", 1024)
        input_channels = kwargs.get("input_channels", 3)
        self.num_classes = kwargs.get("num_classes", 2) 
        self.return_feat_map = kwargs.get("return_feat_map", False)
        self.feat_maps = []
        #InceptionLayer1
        if self.use_old:
            self.Incption1_conv1 = nn.Conv2d(input_channels, 1, 1, padding=0, bias=False)
        self.Incption1_conv2_1 = nn.Conv2d(input_channels, 4, 1, padding=0, bias=False)
        self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption1_conv3_1 = nn.Conv2d(input_channels, 4, 1, padding=0, bias=False)
        self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption1_conv4_1 = nn.Conv2d(input_channels, 2, 1, padding=0, bias=False)
        self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption1_bn = nn.BatchNorm2d(11)


        #InceptionLayer2
        self.Incption2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption2_bn = nn.BatchNorm2d(12)

        #Normal Layer
        self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        if self.use_old:
            self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))
        else:
            # pool more on temporal dimension
            self.speech_timepool2 = nn.MaxPool2d(kernel_size=(2, 4))
            #self.lfcc_timepool1 = nn.MaxPool2d(kernel_size=(1, 2))
            #self.lfcc_timepool2 = nn.MaxPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(self.fc1_dim, 16)
        self.fc2 = nn.Linear(16, num_classes)


    #InceptionLayer
    def InceptionLayer1(self, input):
        if self.use_old:
            x1 = self.Incption1_conv1(input)
        else:
            x1 = input # the input channel of speech is always 1, no need to reduce dimension
        x2 = self.Incption1_conv2_1(input)
        if self.return_feat_map:
            self.feat_maps.append(["incption1_conv2_1", x2.cpu().detach().numpy()])
        x2 = self.Incption1_conv2_2(x2)
        if self.return_feat_map:
            self.feat_maps.append(["incption1_conv2_2", x2.cpu().detach().numpy()])
        x3 = self.Incption1_conv3_1(input)
        if self.return_feat_map:
            self.feat_maps.append(["incption1_conv3_1", x3.cpu().detach().numpy()])
        x3 = self.Incption1_conv3_2(x3)
        if self.return_feat_map:
            self.feat_maps.append(["incption1_conv3_2", x3.cpu().detach().numpy()])
        x4 = self.Incption1_conv4_1(input)
        if self.return_feat_map:
            self.feat_maps.append(["incption1_conv4_1", x4.cpu().detach().numpy()])
        x4 = self.Incption1_conv4_2(x4)
        if self.return_feat_map:
            self.feat_maps.append(["incption1_conv4_2", x4.cpu().detach().numpy()])
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption1_bn(y)
        y = self.maxpooling1(y)
        if self.return_feat_map:
            self.feat_maps.append(["incption1_max_pool", y.cpu().detach().numpy()])


        return y

    def InceptionLayer2(self, input):
        x1 = self.Incption2_conv1(input)
        if self.return_feat_map:
            self.feat_maps.append(["incption2_conv1", x1.cpu().detach().numpy()])
        x2 = self.Incption2_conv2_1(input)
        if self.return_feat_map:
            self.feat_maps.append(["incption2_conv2_1", x2.cpu().detach().numpy()])
        x2 = self.Incption2_conv2_2(x2)
        if self.return_feat_map:
            self.feat_maps.append(["incption2_conv2_2", x2.cpu().detach().numpy()])
        x3 = self.Incption2_conv3_1(input)
        if self.return_feat_map:
            self.feat_maps.append(["incption2_conv3_1", x3.cpu().detach().numpy()])
        x3 = self.Incption2_conv3_2(x3)
        if self.return_feat_map:
            self.feat_maps.append(["incption2_conv3_2", x3.cpu().detach().numpy()])
        x4 = self.Incption2_conv4_1(input)
        if self.return_feat_map:
            self.feat_maps.append(["incption2_conv4_1", x4.cpu().detach().numpy()])
        x4 = self.Incption2_conv4_2(x4)
        if self.return_feat_map:
            self.feat_maps.append(["incption2_conv4_2", x4.cpu().detach().numpy()])
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption2_bn(y)
        y = self.maxpooling1(y)
        if self.return_feat_map:
            self.feat_maps.append(["incption2_max_pool", y.cpu().detach().numpy()])

        return y

    def forward(self, input):
        x = self._compute_embedding(input)
        return x

    def _compute_embedding(self, input):
        x = self.InceptionLayer1(input) #(Batch, 11, 128, 128)
        x = self.InceptionLayer2(x) #(Batch, 12, 64, 64)

        x = self.conv1(x) #(Batch, 16, 64 ,64)
        if self.return_feat_map:
            self.feat_maps.append(["conv1", x.cpu().detach().numpy()])
        x = self.relu(x)
        if self.return_feat_map:
            self.feat_maps.append(["relu1", x.cpu().detach().numpy()])
        x = self.bn1(x)
        x = self.maxpooling1(x) #(Batch, 16, 32, 32)
        if self.return_feat_map:
            self.feat_maps.append(["maxpool1", x.cpu().detach().numpy()])
        #print(f"before lfcc_timepool: {x.shape}")
        #x = self.lfcc_timepool1(x)

        x = self.conv2(x) #(Batch, 16, 32, 32)
        if self.return_feat_map:
            self.feat_maps.append(["conv2", x.cpu().detach().numpy()])
        x = self.relu(x)
        if self.return_feat_map:
            self.feat_maps.append(["relu2", x.cpu().detach().numpy()])
        x = self.bn1(x)
        if self.use_old:
            x = self.maxpooling2(x)
        else:
            x = self.speech_timepool2(x) #(Batch, 16, 8, 8)
        if self.return_feat_map:
            self.feat_maps.append(["speech_timepool2", x.cpu().detach().numpy()])
        #x = self.mfcc_timepool2(x)
        #x = self.lfcc_timepool2(x)
        x = x.view(x.size(0), -1) #(Batch, 16*8*8)
        x = self.dropout(x)
        feat_before_out = x

        x = nn.AdaptiveAvgPool1d(self.fc1_dim)(x)
        x = self.fc1(x) #(Batch, 16)  ### <-- o tu
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return feat_before_out, x


class W2V2MesoInception(nn.Module):
    # NCTF

    def __init__(self, num_classes=1, **kwargs):
        super().__init__()
        self.fc1_dim = kwargs.get("fc1_dim", 1024)
        input_channels = kwargs.get("input_channels", 3)
        self.num_classes = kwargs.get("num_classes", 2) 
        self.return_feat_map = kwargs.get("return_feat_map", False)
        self.feat_maps = []

        #InceptionLayer1
        #self.Incption1_conv1 = nn.Conv2d(input_channels, 1, 1, padding=0, bias=False)
        self.Incption1_conv2_1 = nn.Conv2d(input_channels, 4, 1, padding=0, bias=False)
        self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption1_conv3_1 = nn.Conv2d(input_channels, 4, 1, padding=0, bias=False)
        self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption1_conv4_1 = nn.Conv2d(input_channels, 2, 1, padding=0, bias=False)
        self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption1_bn = nn.BatchNorm2d(11)


        #InceptionLayer2
        self.Incption2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption2_bn = nn.BatchNorm2d(12)

        #Normal Layer
        self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.speech_timepool1 = nn.MaxPool2d(kernel_size=(2, 4))

        self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))
        self.speech_timepool2 = nn.MaxPool2d(kernel_size=(2, 4))
        #self.lfcc_timepool1 = nn.MaxPool2d(kernel_size=(1, 2))
        #self.lfcc_timepool2 = nn.MaxPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(self.fc1_dim, 16)
        self.fc2 = nn.Linear(16, num_classes)

        #InceptionLayer
    def InceptionLayer1(self, input):
        #x1 = self.Incption1_conv1(input)
        x1 = input # the input channel of speech is always 1, no need to reduce dimension
        if self.return_feat_map:
            self.feat_maps.append(["meso_input", x1.cpu().detach().numpy()])
        x2 = self.Incption1_conv2_1(input)
        if self.return_feat_map:
            self.feat_maps.append(["incption1_conv2_1", x2.cpu().detach().numpy()])
        x2 = self.Incption1_conv2_2(x2)
        if self.return_feat_map:
            self.feat_maps.append(["incption1_conv2_2", x2.cpu().detach().numpy()])
        x3 = self.Incption1_conv3_1(input)
        if self.return_feat_map:
            self.feat_maps.append(["incption1_conv3_1", x3.cpu().detach().numpy()])
        x3 = self.Incption1_conv3_2(x3)
        if self.return_feat_map:
            self.feat_maps.append(["incption1_conv3_2", x3.cpu().detach().numpy()])
        x4 = self.Incption1_conv4_1(input)
        if self.return_feat_map:
            self.feat_maps.append(["incption1_conv4_1", x4.cpu().detach().numpy()])
        x4 = self.Incption1_conv4_2(x4)
        if self.return_feat_map:
            self.feat_maps.append(["incption1_conv4_2", x4.cpu().detach().numpy()])
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption1_bn(y)
        y = self.maxpooling1(y)
        if self.return_feat_map:
            self.feat_maps.append(["incption1_max_pool", y.cpu().detach().numpy()])

        return y

    def InceptionLayer2(self, input):
        x1 = self.Incption2_conv1(input)
        if self.return_feat_map:
            self.feat_maps.append(["incption2_conv1", x1.cpu().detach().numpy()])
        x2 = self.Incption2_conv2_1(input)
        if self.return_feat_map:
            self.feat_maps.append(["incption2_conv2_1", x2.cpu().detach().numpy()])
        x2 = self.Incption2_conv2_2(x2)
        if self.return_feat_map:
            self.feat_maps.append(["incption2_conv2_2", x2.cpu().detach().numpy()])
        x3 = self.Incption2_conv3_1(input)
        if self.return_feat_map:
            self.feat_maps.append(["incption2_conv3_1", x3.cpu().detach().numpy()])
        x3 = self.Incption2_conv3_2(x3)
        if self.return_feat_map:
            self.feat_maps.append(["incption2_conv3_2", x3.cpu().detach().numpy()])
        x4 = self.Incption2_conv4_1(input)
        if self.return_feat_map:
            self.feat_maps.append(["incption2_conv4_1", x4.cpu().detach().numpy()])
        x4 = self.Incption2_conv4_2(x4)
        if self.return_feat_map:
            self.feat_maps.append(["incption2_conv4_2", x4.cpu().detach().numpy()])
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption2_bn(y)
        y = self.maxpooling1(y)
        if self.return_feat_map:
            self.feat_maps.append(["incption2_max_pool", y.cpu().detach().numpy()])

        return y

    def forward(self, input):
        x = self.InceptionLayer1(input) #(Batch, 11, 128, 128)
        x = self.InceptionLayer2(x) #(Batch, 12, 64, 64)

        x = self.conv1(x) #(Batch, 16, 64 ,64)
        if self.return_feat_map:
            self.feat_maps.append(["conv1", x.cpu().detach().numpy()])
        x = self.relu(x)
        x = self.bn1(x)
        #print(f"shape before maxpooling1: {x.shape}")
        x = self.speech_timepool1(x) #(Batch, 16, 32, 32)
        if self.return_feat_map:
            self.feat_maps.append(["speech_timepool1", x.cpu().detach().numpy()])
        #print(f"shape after maxpooling1: {x.shape}")
        #x = self.lfcc_timepool1(x)

        x = self.conv2(x) #(Batch, 16, 32, 32)
        if self.return_feat_map:
            self.feat_maps.append(["conv2", x.cpu().detach().numpy()])
        x = self.relu(x)
        x = self.bn1(x)
        #print(f"shape before speech_timepool2: {x.shape}")
        x = self.speech_timepool2(x) #(Batch, 16, 12, 16)
        if self.return_feat_map:
            self.feat_maps.append(["speech_timepool2", x.cpu().detach().numpy()])
        #print(f"shape after speech_timepool2: {x.shape}")

        x = x.view(x.size(0), -1) #(Batch, 16*8*8)
        x = self.dropout(x)
        feat_before_out = x

        x = nn.AdaptiveAvgPool1d(self.fc1_dim)(x)
        x = self.fc1(x) #(Batch, 16)  ### <-- o tu
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return feat_before_out, x


if __name__ == "__main__":
    import numpy as np
    # NCFT
    x = np.random.randn(3, 1, 199, 1024).astype(np.float32)
    x = torch.from_numpy(x)
    model = MesoInception4(
        input_channels=1,
        fc1_dim=1024,
        num_classes=1
    )
    feats, feats_out = model(x)
    print(f"feats: {feats.shape}")
    print(f"feats_out: {feats_out.shape}")