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
        self.frontend_model = None

        if self.frontend is None:
            logger.info("No frontend model is specified")
        elif self.frontend == "XLSR":
            self.frontend_model = frontend_models.XLSR(self.device)
        else:
            raise ValueError(f"Frontend {self.frontend} is not supported")

        if self.frontend_model is not None and self.freeze_frontend:
            logger.info("Freezing frontend model")
            for param in self.frontend_model.parameters():
                param.requires_grad = False

    def forward(self, source: dict, **kwargs) -> dict:
        for key in source:
            source[key] = utils.move_to_cuda(source[key], self.device)
        feats = source["feats"] # NCT
        if self.frontend_model is not None:
            # NCT
            feats = self.frontend_model.extract_feat(feats.squeeze(1))
            source["feats"] = feats.unsqueeze(1)