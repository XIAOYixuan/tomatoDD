# encoding: utf-8
# Author: Yixuan
# 
#
from .fad_trim import SVDD, MLAADMono, ASVspoof19, GPTSoVITS, GeneralFAD, FastGeneralFAD
from .mutli_datasets import MultiDS
def get_dataset_class(class_name):
    if class_name == "SVDD":
        return SVDD
    elif class_name == "MLAADMono":
        return MLAADMono 
    elif class_name == "ASVspoof19":
        return ASVspoof19
    elif class_name == "GPTSoVITS":
        return GPTSoVITS 
    elif class_name == "MultiDS":
        return MultiDS
    elif class_name == "GeneralFAD":
        return GeneralFAD
    elif class_name == "FastGeneralFAD":
        return FastGeneralFAD 
    else:
        raise ValueError(f"Dataset class {class_name} not found")