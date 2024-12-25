# encoding: utf-8
# Author: Yixuan
# 
#
from .base import BuiltInCriterion 
from .continual_learning_loss import OCSoftmax, OCBCELoss, OCSoftmaxK, TCSoftmax, OCSoftmaxRecon
from .disc_loss import HiFiDiscLoss
def get_criterion_class(class_name):
    if class_name == "OCSoftmax":
        return OCSoftmax
    elif class_name == "OCBCELoss":
        return OCBCELoss
    elif class_name == "OCSoftmaxK":
        return OCSoftmaxK
    elif class_name == "TCSoftmax":
        return TCSoftmax
    elif class_name == "OCSoftmaxRecon":
        return OCSoftmaxRecon
    elif class_name == "HiFi":
        return HiFiDiscLoss
    elif class_name == "built-in":
        return BuiltInCriterion
    else:
        raise ValueError("Unknown criterion class: {}".format(class_name))