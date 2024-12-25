from .fad_models import ResNet, WhisperMeso, Wav2Vec2AASIST, PlainLCNN, MesoNet, AASIST
from .megabyte import MegaByteFAD, LGMegaByte, OCMegaByte, TimeFreqMega, TimeFreqConvHead
from .conv_timefreq import ConvFreqTime
from .discriminators import HiFiDisc
from .frontend_only import XLSRAdapter, XLSRTimeFirst, XLSRAllAttn, XLSRTimeAttnOnly, XLSRTimeAttnWithBottleneck
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
    elif class_name == "HiFiDisc":
        return HiFiDisc
    elif class_name == "XLSRAdapter":
        return XLSRAdapter
    elif class_name == "XLSRTimeFirst":
        return XLSRTimeFirst
    elif class_name == "XLSRAllAttn":
        return XLSRAllAttn
    elif class_name == "XLSRTimeAttnOnly":
        return XLSRTimeAttnOnly
    elif class_name == "XLSRTimeAttnWithBottleneck":
        return XLSRTimeAttnWithBottleneck
    else:
        raise ValueError("Unknown model class: {}".format(class_name))