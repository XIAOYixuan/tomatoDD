# encoding: utf-8
# Author: Yixuan
#
#

import torch
import argparse
from tomato.utils import utils
from .base import ClassificationBase

class SLMAdapterBase(ClassificationBase):
    """
    Speech Language Model Adapter Base Class
    Library: Huggingface Transformers
    """

    def __init__(self, args):
        self.frontend = "tf_w2v2"
        super().__init__(args)
        self.frontend_dim = 1024
        self.low_dim = 256
        self.attn_num_heads = 4
        self.attn_hiddim = 256
        self.layer_id = getattr(args, "layer_id", None)
        if self.layer_id is None:
            self.num_layers = getattr(args, "num_layers", 25)
        else:
            self.num_layers = self.layer_id + 1

    def extract_features(self, source: dict):
        source = utils.move_to_cuda(source, self.device)
        feats = source["feats"]
        padding_masks = None if "padding_mask" not in source else source["padding_mask"]

        with torch.no_grad():
            out = self.frontend_model(input_values=feats,
                                      attention_mask=padding_masks,
                                      output_hidden_states=True)
        return out
    

if __name__ == "__main__":
    import random
    args = argparse.Namespace(
        cuda=0,
        num_classes=1,
        frontend='tf_w2v2',
        tf_mdl = "facebook/wav2vec2-xls-r-300m",
    )
    model = SLMAdapterBase(args)

    bz, T = 3, 52332
    feats = torch.randn(bz, T)
    padding_mask = torch.zeros(bz, T)
    for i in range(bz):
        mask_len = random.randint(10000, 20000)
        padding_mask[i, :mask_len] = 1
    source = {"feats": feats, "padding_mask": padding_mask}
    out = model.extract_features(source)
    for item in out:
        if isinstance(item, tuple):
            for subitem in item:
                print(subitem.shape)
        elif isinstance(item, torch.Tensor):
            print(item.shape) # shape N, T, D
            print(item)
        else:
            print(type(item))
