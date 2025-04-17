# encoding: utf-8
# Author: Yixuan
# 
#

import sys
import argparse
from enum import Enum
from pathlib import Path

from omegaconf import OmegaConf
import torch
import fairseq

def convert_fairseq_model(mdl_path: str, out_path: Path):
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = out_path / "model.pt"
    if out_path.exists():
        res = input("File already exists, overwrite? (y/n)")
        if res == "n":
            print("Stopping conversion.")
            return
        elif res == "y":
            print("Overwriting file.")
        else:
            print("Invalid input, stopping conversion.")
            return

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([mdl_path])
    model = model[0]
    mdl_cfg = model.cfg
    cfg_dict = OmegaConf.to_container(mdl_cfg, resolve=True)
    args = argparse.Namespace(**cfg_dict)
    for k, v in args.__dict__.items():
        if isinstance(v, Enum):
            args.__dict__[k] = str(v) 

    print(f"Saving file to {out_path}.")
    torch.save({"cfg": args, "state_dict": model.state_dict()}, out_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tools/fairseq_model_converter.py [path to the pretrained model] [output directory]")
        sys.exit(1)
    mdl_path = sys.argv[1]
    out_path = Path(sys.argv[2])
    convert_fairseq_model(mdl_path, out_path)