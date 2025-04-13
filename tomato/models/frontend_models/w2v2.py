# encoding: utf-8
# Author: Yixuan
# 
#
import os

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from tomato.utils import logger
from .base import BaseFrontEnd

class FairseqFrontend(BaseFrontEnd):

    def __init__(self, device, args=None):
        import fairseq
        super(FairseqFrontend, self).__init__(device)
        # TODO: set the path in the config
        frontend_path = getattr(args, 'frontend_path', None)
        if frontend_path is None:
            cp_path = os.environ.get(f"{self.model_tag}_CP_PATH") 
        else:
            cp_path = frontend_path
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device

    def extract_feat(self, input_data, return_layer_outs=False, layer_id=None):
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)

        # input_data: NCT
        if input_data.shape[1] != 1:
            raise ValueError(f"{self.model_tag} frontend only support single channel input, got {input_data.shape[1]} channels")
        input_data = input_data.squeeze(1)
        output = self.model(input_data, mask=False, features_only=True)
        if return_layer_outs:
            post_feats = output['post_features'] # [n, t, d]
            if layer_id is None:
                hiddens = output['layer_results']
            else:
                hiddens = []
                for i in range(len(output['layer_results'])):
                    if i + 1 > layer_id:
                        break
                    cur_layer = output['layer_results'][i][0] # [t, n, d]
                    cur_layer = rearrange(cur_layer, 't n d -> n t d')
                    hiddens.append(cur_layer)
            layer_outs = [post_feats] + hiddens
            return layer_outs
        else:
            emb = output['x']
            emb = rearrange(emb, 'n t f -> n 1 f t') 
            return emb
    
class XLSR(FairseqFrontend):
    def __init__(self, device, args=None):
        self.model_tag = "XLSR"
        super(XLSR, self).__init__(device, args)
        self.out_dim = 1024
        return
    
class HuBERT(FairseqFrontend):

    def __init__(self, device, args=None):
        self.model_tag = "HUBERT"
        super(HuBERT, self).__init__(device, args)
        self.out_dim = 1024

if __name__ == "__main__":
    device = "cpu"
    bs = 3
    c = 1
    wav_length = 64_000
    x = torch.rand(bs, c, wav_length)
    
    def load_fairseq_model(Model):
        model = Model("cpu")
        print(f"x shape {x.shape}")
        feat = model.extract_feat(x)
        print(f"feat shape {feat.shape}")
    
    load_fairseq_model(HuBERT)
