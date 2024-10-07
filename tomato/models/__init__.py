from .fad_models import ResNet, WhisperMeso, Wav2Vec2AASIST, PlainLCNN, MesoNet, AASIST
from .megabyte import MegaByteFAD, LGMegaByte, OCMegaByte, TimeFreqMega, TimeFreqConvHead
from .conv_timefreq import ConvFreqTime
def get_model_class(class_name):
    if class_name == "ResNet":
        return ResNet 
    elif class_name == "PlainLCNN":
        return PlainLCNN
    elif class_name == "WhisperMeso":
        return WhisperMeso
    elif class_name == "wav2vecAASIST":
        return Wav2Vec2AASIST 
    elif class_name == "MesoNet":
        return MesoNet
    elif class_name == "AASIST":
        return AASIST
    elif class_name == "MegaByteFAD":
        return MegaByteFAD
    elif class_name == "LGMegaByte":
        return LGMegaByte
    elif class_name == "OCMegaByte":
        return OCMegaByte
    elif class_name == "TimeFreqMega":
        return TimeFreqMega
    elif class_name == "TimeFreqConvHead":
        return TimeFreqConvHead
    elif class_name == "ConvFreqTime":
        return ConvFreqTime
    else:
        raise ValueError("Unknown model class: {}".format(class_name))