# encoding: utf-8
# Author: Yixuan
# 
#

# implement the transformation functions like lfcc, mfcc, etc
import argparse
import torch
import torchaudio
from tomato.utils import logger, utils
from abc import ABC, abstractmethod

class BaseTransform(ABC):
    
    def __init__(self, args: argparse.Namespace):
        pass

    @abstractmethod
    def __call__(self, x):
        pass


class Spectrogram(BaseTransform):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        n_fft = getattr(args, "n_fft", 512)
        win_length = getattr(args, "win_length", 400)
        hop_length = getattr(args, "hop_length", 160) 
        self.func = torchaudio.transforms.Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
            )
        

class LFCC(BaseTransform):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        sample_rate = getattr(args, "sample_rate", 16000)
        n_filter = getattr(args, "n_filter", 128)
        n_lfcc = getattr(args, "n_lfcc", 128)
        n_fft = getattr(args, "n_fft", 512)
        win_length = getattr(args, "win_length", 400)
        hop_length = getattr(args, "hop_length", 160) 
        self.func = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_filter=n_filter,
            n_lfcc=n_lfcc,
            speckwargs={
                "n_fft": n_fft, 
                "win_length": win_length, 
                "hop_length": hop_length, 
            },
        )#.to("cuda")

        
        self.with_delta = getattr(args, "with_delta", False)
        if self.with_delta:
            self.delta_fn = torchaudio.transforms.ComputeDeltas(
                win_length=win_length,
                mode="replicate",
            )

    
    # TODO: make it run on GPU? at the model side?
    def __call__(self, x):
        # x.shape = [C, F]
        if x.shape[0] != 1:
            raise ValueError("LFCC only support single channel input")
        #x = utils.move_to_cuda(x, "cuda")
        feats = self.func(x)
        if self.with_delta:
            delta = self.delta_fn(feats)
            double_delta = self.delta_fn(delta)
            feats = torch.cat((feats, delta, double_delta), 1)
        return feats

class MFCC(BaseTransform):

    def __init__(self, args) -> None:
        super().__init__(args)
        sample_rate = getattr(args, "sample_rate", 16000)
        n_mfcc = getattr(args, "n_mfcc", 128)
        n_fft = getattr(args, "n_fft", 512)
        win_length = getattr(args, "win_length", 400)
        hop_length = getattr(args, "hop_length", 160) 
        self.func = torchaudio.transforms.MFCC(
        sample_rate= sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft, 
            "win_length": win_length, 
            "hop_length": hop_length, 
        },
        )#.to("cuda")

        
        self.with_delta = getattr(args, "with_delta", False)
        if self.with_delta:
            self.delta_fn = torchaudio.transforms.ComputeDeltas(
                win_length=win_length,
                mode="replicate",
            )


    def __call__(self, x):
        # x shape: [channel, T]
        # mfcc output: [channel, F, T]
        mfcc = self.func(x) 
        if self.with_delta:
            delta = self.delta_fn(mfcc) 
            double_delta = self.delta_fn(delta) 
            mfcc = torch.cat((mfcc, delta, double_delta), 1)
        return mfcc


if __name__ == "__main__":
    import numpy as np
    x = np.random.randn(1, 160000).astype(np.float32)
    lfcc = LFCC(
        argparse.Namespace(
            n_filter=20,
            n_lfcc=20,
            win_length=320,
            hop_length=160,
            with_delta=True
        )
    )
    out = lfcc(torch.from_numpy(x))
    print(out.shape)

    mfcc = MFCC(
        argparse.Namespace(
            n_mfcc=128,
            n_fft=512,
            win_length=400,
            hop_length=160
        )
    )
    out = mfcc(torch.from_numpy(x))
    print(out.shape)