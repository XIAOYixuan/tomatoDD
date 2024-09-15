# encoding: utf-8
# Author: Yixuan
# 
# 
from .fad_task import FADBaseTask, FADCLBaseTask, FADMemoryReplayBase 
from .fad_task import XentTraining, OCSoftmaxTraining
from .fad_cl_task import FADLwFTask, LwFTraining
def get_task(name):
    if name == "cl-xent":
        return FADCLBaseTask(XentTraining())
    elif name == "cl-oc":
        return FADCLBaseTask(OCSoftmaxTraining()) 
    elif name == "xent":
        return FADBaseTask(XentTraining())
    elif name == "oc":
        return FADBaseTask(OCSoftmaxTraining()) 
    elif name == "replay-xent":
        return FADMemoryReplayBase(XentTraining()) 
    elif name == "lwf":
        return FADLwFTask(LwFTraining())
    else:
        raise ValueError("Unknown task class: {}".format(name))