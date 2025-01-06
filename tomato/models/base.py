# encoding: utf-8
# Author: Yixuan
#
#

import argparse

import torch
import torch.nn as nn

from tomato.utils import utils, logger
import tomato.models.frontend_models as frontend_models


class BaseModel(nn.Module):

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.device = torch.device(f"cuda:{args.cuda}")
        self.args = args


    def load_checkpoint(self, model_path):
        self.load_state_dict(torch.load(model_path, map_location=self.device), strict=True)
        self.to(self.device)
        logger.info(f"Loaded checkpoint from {model_path}")


class ClassificationBase(BaseModel):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.num_classes = getattr(args, 'num_classes', 2)
        self.frontend = getattr(args, 'frontend', None)
        self.freeze_frontend = getattr(args, 'freeze_frontend', True)
        self.have_padding_mask = getattr(args, "have_padding_mask", False)
        self.frontend_model = None

        if self.frontend is None:
            logger.info("No frontend model is specified")
        elif self.frontend == "XLSR":
            self.frontend_model = frontend_models.XLSR(self.device, args)
        elif self.frontend == "HuBERT":
            self.frontend_model = frontend_models.HuBERT(self.device, args)
        elif self.frontend == "facodec":
            self.frontend_model = frontend_models.FACodec(self.device, args)
        elif self.frontend == "whisper":
            self.frontend_model = frontend_models.Whisper(self.device, args)
        elif self.frontend == "tf_w2v2":
            self.frontend_model = frontend_models.TFW2V2(self.device, args)
        else:
            raise ValueError(f"Frontend {self.frontend} is not supported")

        if self.frontend_model is not None and self.freeze_frontend:
            logger.info("Freezing frontend model")
            for param in self.frontend_model.parameters():
                param.requires_grad = False
            self.frontend_model.eval()

    def forward_frontend(self, source: dict, **kwargs) -> dict:
        for key in source:
            source[key] = utils.move_to_cuda(source[key], self.device)
        feats = source["feats"] # NCT
        if self.frontend_model is not None:
            # NCT
            feats = self.frontend_model.extract_feat(feats)
            #logger.info(f"feats shape: {feats.shape}")
            # NCFT
            source["feats"] = feats
    
    def forward(self, source: dict, **kwargs) -> dict:
        return self.forward_frontend(source, **kwargs)

    def state_dict(self):
        state_dict = super().state_dict()
        if self.frontend_model is not None and self.freeze_frontend:
            # remove frontend model parameters
            frontend_state_dict = self.frontend_model.state_dict()
            for key in frontend_state_dict:
                #logger.info(key)
                real_key = f"frontend_model.{key}"
                del state_dict[real_key]
        return state_dict
    
    def load_checkpoint(self, model_path):
        other_keys = self.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        self.to(self.device)
        logger.info(f"Loaded checkpoint from {model_path}")

        missing_keys = set(other_keys.missing_keys)
        if self.frontend_model is not None and self.freeze_frontend:
            frontend_state_dict = self.frontend_model.state_dict()
            for key in frontend_state_dict:
                real_key = f"frontend_model.{key}"
                if real_key in missing_keys:
                    missing_keys.remove(real_key)
        if len(missing_keys) > 0:
            raise ValueError(f"Missing keys after loading checkpoint: {missing_keys}")

        if len(other_keys.unexpected_keys) > 0:
            raise ValueError(f"Unexpected keys after loading checkpoint: {other_keys.unexpected_keys}")

        #logger.info(other_keys.missing_keys)
        #logger.info(other_keys.unexpected_keys)
