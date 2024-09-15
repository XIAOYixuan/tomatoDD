# encoding: utf-8
# Author: Yixuan
# 
# Adapted from https://github.com/piotrkawa/deepfake-whisper-features

import torch
#from src import frontends

from .whisper_main import ModelDimensions, Whisper, log_mel_spectrogram
from .meso_net import MesoInception4
from .commons import WHISPER_MODEL_WEIGHTS_PATH


class WhisperMesoNet(MesoInception4):
    def __init__(self, freeze_encoder, **kwargs):
        super().__init__(**kwargs)

        self.device = kwargs['device']
        checkpoint = torch.load(WHISPER_MODEL_WEIGHTS_PATH)
        dims = ModelDimensions(**checkpoint["dims"].__dict__)
        model = Whisper(dims)
        model = model.to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.whisper_model = model
        if freeze_encoder:
            for param in self.whisper_model.parameters():
                param.requires_grad = False

    def compute_whisper_features(self, x):
        specs = []
        for sample in x:
            specs.append(log_mel_spectrogram(sample))
        x = torch.stack(specs)
        x = self.whisper_model(x)

        x = x.permute(0, 2, 1)  # (bs, frames, 3 x n_lfcc)
        x = x.unsqueeze(1)  # (bs, 1, frames, 3 x n_lfcc)
        x = x.repeat(
            (1, 1, 1, 2)
        )  # (bs, 1, frames, 3 x n_lfcc) -> (bs, 1, frames, 3000)
        return x

    def forward(self, x):
        # we assume that the data is correct (i.e. 30s)
        x = self.compute_whisper_features(x)
        feat, feat_out = self._compute_embedding(x)
        return feat, feat_out 


if __name__ == "__main__":
    import numpy as np

    input_channels = 1
    device = "cpu"
    classifier = WhisperMesoNet(
        input_channels=input_channels,
        freeze_encoder=True,
        fc1_dim=1024,
        device=device,
    )

    #input_channels = 2
    #classifier_2 = WhisperMultiFrontMesoNet(
    #    input_channels=input_channels,
    #    freeze_encoder=True,
    #    fc1_dim=1024,
    #    device=device,
    #    frontend_algorithm="lfcc"
    #)
    x = np.random.rand(2, 30 * 16_000).astype(np.float32)
    x = torch.from_numpy(x)

    feat, feat_out = classifier(x)
    print(feat.shape)
    print(feat_out.shape)

    #out = classifier_2(x)
    #print(out.shape)