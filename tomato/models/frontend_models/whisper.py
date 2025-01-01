# encoding: utf-8
# Author: Yixuan
# 
#
import os
from dataclasses import dataclass
from argparse import Namespace 
from typing import Iterable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import numpy as np
from functools import lru_cache
import librosa

from tomato.utils import logger
from .base import BaseFrontEnd

class WhisperLogMel(nn.Module):

    def __init__(self, n_mels, device='cpu', n_audio_ctx=200) -> None:
        super().__init__()
        self.sample_rate = 16_000
        self.n_fft = 400
        self.n_mels = n_mels
        self.n_audio_ctx = n_audio_ctx
        self.mel_fitlers = self._load_mel_filters().to(device)
        # log_mel parameters
        self.window = torch.hann_window(self.n_fft).to(device)
        self.to(device)

    def _load_mel_filters(self):
        mel_fil = librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels)
        mel_fil = torch.from_numpy(mel_fil)
        return mel_fil.unsqueeze(0)
    
    def forward(self, audio: torch.Tensor):
        stft = torch.stft(audio, self.n_fft, hop_length=160, window=self.window, return_complex=True)
        magnitudes = stft[:, :, :-1].abs() ** 2
        mel_spec = torch.matmul(self.mel_fitlers, magnitudes) # [n, n_mel, t]
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        # shape: n, n_mel, t
        # pad it to 3000 frames, the pad vec is the last frame
        if log_spec.shape[-1] < self.n_audio_ctx:
            log_spec = F.pad(log_spec, (0, self.n_audio_ctx - log_spec.shape[-1]))
        return log_spec 
    

class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )
    

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

def sinusoids(length, channels, max_timescale=10_000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv)

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
    

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        if self.cross_attn and self.cross_attn_ln:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor, layer_id=None):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x)) # N, D, T
        x = F.gelu(self.conv2(x)) # N, D, T
        x = x.permute(0, 2, 1) # N, T, D
        N, T, D = x.shape
        sub_pos = self.positional_embedding[:T, :]
        assert x.shape[1:] == sub_pos.shape, f"incorrect audio shape {x.shape[1:]} != {sub_pos.shape}"
        x = (x + sub_pos).to(x.dtype)
        #layer_outputs = [x.detach().clone()]
        layer_outputs = [x]

        if layer_id is not None and layer_id == 0:
            return x, layer_outputs

        for idx, block in enumerate(self.blocks):
            x = block(x)
            #layer_outputs.append(x.detach().clone())
            layer_outputs.append(x)
            #print(f"forward {idx+1}, {layer_id}, len {len(layer_outputs)}")
            if idx + 1 == layer_id:
                #print(f"returning, the length of layer_outputs is {len(layer_outputs)}") 
                return x, layer_outputs

        x = self.ln_post(x)
        return x, layer_outputs

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int

class Whisper(BaseFrontEnd):
    def __init__(self, device, args=None):
        super().__init__(device)
        self.model_tag = "Whisper"
        frontend_path = getattr(args, 'frontend_path', None)
        ckpt = torch.load(frontend_path, map_location=device)
        logger.info(f"Loaded checkpoint from {frontend_path}")
        dims = ModelDimensions(**ckpt['dims'].__dict__)
        self.n_mels = dims.n_mels
        self.n_audio_ctx = getattr(args, 'n_audio_ctx', 200)
        #logger.info(f"------- n_mels: {self.n_mels}, n_audio_ctx: {self.n_audio_ctx}") 
        self.n_audio_state = dims.n_audio_state
        self.n_audio_head = dims.n_audio_head 
        self.n_audio_layer = dims.n_audio_layer
        
        self.log_mel = WhisperLogMel(self.n_mels, device, self.n_audio_ctx)
        self.encoder = AudioEncoder(
            self.n_mels,
            self.n_audio_ctx,
            self.n_audio_state,
            self.n_audio_head,
            self.n_audio_layer,
        )
        
        missing, unexpected = self.load_state_dict(ckpt['model_state_dict'], strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        self.to(device)
        self.eval()

    def extract_feat(self, input_data: torch.Tensor, return_layer_outs: bool = False, layer_id: int = None):
        # input is mel-spec
        log_mel = self.log_mel(input_data)
        final_out, layer_outs = self.encoder(log_mel, layer_id)
        for i in range(len(layer_outs)):
            layer_outs[i] = rearrange(layer_outs[i], 'n t d -> t n d')
        
        if return_layer_outs:
            #print(f"final return len {len(layer_outs)}")
            return layer_outs
        else:
            #print(f"final_out {type(final_out)}") 
            return final_out


if __name__ == "__main__":
    frontend_path = os.environ.get('FRONTEND_PATH')
    if frontend_path is None:
        raise ValueError("FRONTEND_PATH environment variable must be set")
    args = Namespace(
        frontend_path=frontend_path,
        device='cpu',
    )
    model = Whisper(args.device, args)
    n, t = 4, 64000
    x = torch.randn(n, t)
    layer_out = model.extract_feat(x, return_layer_outs=True, layer_id=2)
    for i, layer in enumerate(layer_out):
        print(f"Layer {i}: {layer.shape}") # 200, 4, 1024