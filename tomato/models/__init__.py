from .fad_models import ResNet, WhisperMeso, Wav2Vec2AASIST, PlainLCNN, MesoNet

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
    else:
        raise ValueError("Unknown model class: {}".format(class_name))