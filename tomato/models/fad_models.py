# encoding: utf-8
# Author: Yixuan
#
#

from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
from torch import Tensor
import os
import numpy as np

from .base import ClassificationBase
from tomato.utils import utils, logger
import tomato.models.predefined as predefined

class ResNet(ClassificationBase):

    def __init__(self, args):
        super().__init__(args)
        # TODO:compute the num nodes, the H_out of layer 4
        # num_node: H_out layer 4
        num_nodes = getattr(args, 'num_nodes', 3)
        enc_dim = getattr(args, 'enc_dim', 256)
        resnet_type = getattr(args, 'resnet_type', '18')
        self.model = predefined.ResNet(num_nodes=num_nodes,
                            enc_dim=enc_dim, 
                            resnet_type=resnet_type,
                            nclasses=self.num_classes)
        self.model.to(self.device)

    def forward(self, source: dict, **kwargs) -> dict:
        source = utils.move_to_cuda(source, self.device)
        # we need: [N, C, F, T], channel = 1
        feats = source["feats"]
        feats, feats_out = self.model(feats)
        return {
            "feats": feats,
            "feats_out": feats_out 
            }


class WhisperMeso(ClassificationBase):
    # to be detangled
    def __init__(self, args: Namespace):
        super().__init__(args)
        input_channels = getattr(args, 'input_channels', 1)
        freeze_encoder = getattr(args, 'freeze_encoder', True)
        fc1_dim = getattr(args, 'fc1_dim', 1024)
        self.model = predefined.WhisperMesoNet(
            input_channels=input_channels,
            freeze_encoder=freeze_encoder,
            fc1_dim=fc1_dim,
            device=self.device,
            num_classes=self.num_classes
        )
        self.model.to(self.device)

    def forward(self, source: dict, **kwargs) -> dict:
        source = utils.move_to_cuda(source, self.device)
        feats = source["feats"]
        if feats.shape[1] != 1:
            raise ValueError("Input shape must be [N, 1, T], single channel")
        feats = feats.squeeze(1)
        feats, feats_out = self.model(feats)
        feats_out = feats_out.squeeze(1)
        return {
            "feats": feats,
            "feats_out": feats_out
        }


class Wav2Vec2AASIST(ClassificationBase):
    # to be decoupled
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.model = predefined.Wav2Vec2Model(
            device=self.device,
            num_classes=self.num_classes
        )
        self.model.to(self.device)

        freeze_w2v2 = getattr(args, 'freeze_w2v2', True)
        if freeze_w2v2:
            for param in self.model.ssl_model.parameters():
                param.requires_grad = False

    def forward(self, source: dict, **kwargs) -> dict:
        source = utils.move_to_cuda(source, self.device)
        feats = source["feats"]
        if feats.shape[1] != 1:
            raise ValueError("Input shape must be [N, 1, T], single channel")
        feats = feats.squeeze(1)
        feats, feats_out = self.model(feats)
        return {
            "feats": feats, 
            "feats_out": feats_out
        }


class PlainLCNN(ClassificationBase):
    """ No blstm 
    """
    def __init__(self, args: Namespace):
        super().__init__(args)
        enc_dim =  getattr(args, 'enc_dim', 256)
        feat_len = getattr(args, 'feat_len', 401)
        F_len = getattr(args, 'F_len', None)
        T_len = getattr(args, 'T_len', None)
        if F_len is None or T_len is None:
            raise ValueError("F_len and T_len must be specified. Input format: NCFT")
        self.model = predefined.PlainLCNN(
            enc_dim=enc_dim,
            feat_len=feat_len,
            nclasses=self.num_classes,
            F_len = F_len,
            T_len = T_len
        )
        self.model.to(self.device)

    def forward(self, source: dict, **kwargs) -> dict:
        source = utils.move_to_cuda(source, self.device)
        # source: batch, 
        feats = source["feats"]
        #feats = feats.transpose(2, 3)
        feats,feats_out = self.model(feats)
        return {
            "feats": feats,
            "feats_out": feats_out
        }

class MesoNet(ClassificationBase):

    def __init__(self, args: Namespace):
        super().__init__(args)
        from .predefined.meso_net import MesoInception4, W2V2MesoInception
        input_channels = getattr(args, 'input_channels', 1)
        fc1_dim = getattr(args, 'fc1_dim', 1024)

        if self.frontend is None:
            self.model = MesoInception4(
                input_channels=input_channels,
                fc1_dim=fc1_dim,
                num_classes=self.num_classes,
                frontend=self.frontend
            )
        elif self.frontend == "XLSR":
            self.model = W2V2MesoInception(
                input_channels=input_channels,
                fc1_dim=fc1_dim,
                num_classes=self.num_classes,
                frontend=self.frontend
            )

        self.model.to(self.device)

    def forward(self, source: dict, **kwargs) -> dict:
        super().forward(source)
        feats = source["feats"]
        # print device of feats

        feats, feats_out = self.model(feats)
        # print output shape
        #logger.info(f"feats shape: {feats.shape}")
        #logger.info(f"feats_out shape: {feats_out.shape}")
        return {
            "feats": feats,
            "feats_out": feats_out
        }
    
    #def load_checkpoint(self, model_path):
    #    self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
    #    self.to(self.device)
    #    logger.info(f"Loaded checkpoint from {model_path}")


if __name__ == "__main__":
    import numpy as np
    # NCFT
    x = np.random.randn(2, 401, 384).astype(np.float32)
    x = torch.from_numpy(x)
    # namespace, cuda:0
    args = Namespace(cuda=0)
    model = MesoNet(args)
    source = {
        "feats": x
    }
    out = model(source)
    feat, feat_out = out["feats"], out["feats_out"] 
    print(feat.shape)
    print(feat_out.shape)

