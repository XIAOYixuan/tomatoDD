import torch
import torch.nn as nn

class BaseFrontEnd(nn.Module):

    def __init__(self, device, args=None):
        super(BaseFrontEnd, self).__init__()
        pass

    def extract_feat(self, input_data):
        raise NotImplementedError

