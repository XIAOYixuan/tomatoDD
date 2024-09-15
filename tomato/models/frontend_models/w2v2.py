# encoding: utf-8
# Author: Yixuan
# 
#
import os

import torch
import torch.nn as nn
import numpy as np
#import fairseq

from tomato.utils import logger

class BaseFrontEnd(nn.Module):

    def __init__(self, device):
        super(BaseFrontEnd, self).__init__()
        pass

    def extract_feat(self, input_data):
        raise NotImplementedError


class XLSR(BaseFrontEnd):

    def __init__(self,device):
        super(XLSR, self).__init__(device)

        # TODO: set the path in the config
        cp_path = os.environ.get("XLSR_CP_PATH") 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        # if is not 2
        if input_data.ndim != 2:
            raise ValueError(f"input_data should be 2D, but got {input_data.ndim}D")
        # [batch, length, dim]
        emb = self.model(input_data, mask=False, features_only=True)['x']
        # we need NCFT
        #emb = emb.permute(0, 2, 1)
        return emb



if __name__ == "__main__":
    device = "cpu"
    xlsr = XLSR("cpu")
    
    bs = 3
    wav_length = 64_000
    x = np.random.rand(bs, wav_length).astype(np.float32)
    x = torch.from_numpy(x)
    feat = xlsr.extract_feat(x)
    print(f"feat shape {feat.shape}")
