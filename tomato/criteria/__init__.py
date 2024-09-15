# encoding: utf-8
# Author: Yixuan
# 
#
from .base import BuiltInCriterion 
from .continual_learning_loss import OCSoftmax

def get_criterion_class(class_name):
    if class_name == "OCSoftmax":
        return OCSoftmax
    elif class_name == "built-in":
        return BuiltInCriterion
    else:
        raise ValueError("Unknown criterion class: {}".format(class_name))