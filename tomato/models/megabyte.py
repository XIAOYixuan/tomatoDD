# encoding: utf-8
#
# Based on UniAudio implementation: https://github.com/yangdongchao/UniAudio/blob/main/UniAudio/model.py
# Adapted by: Yixuan Xiao
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from tomato.utils import logger
from .base import ClassificationBase

class MultiHeadAttention(nn.Module):

    def __init__(
            self, 
            dim_hidden: int, 
            n_head: int,
            ) -> None:
        super().__init__()
        if dim_hidden % n_head != 0:
            raise ValueError("dim_hidden must be divisible by n_head")
        self.n_head = n_head
        self.query = nn.Linear(dim_hidden, dim_hidden)
        self.key = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.value = nn.Linear(dim_hidden, dim_hidden)
        self.out = nn.Linear(dim_hidden, dim_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, L, D = x.shape # N could be B (global) or BT(local)
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scale = (D // self.n_head) ** -0.25
        q = rearrange(q, 'n l (h d) -> n h l d', h=self.n_head) * scale
        k = rearrange(k, 'n l (h d) -> n h d l', h=self.n_head) * scale
        v = rearrange(v, 'n l (h d) -> n h l d', h=self.n_head) * scale

        qk = q @ k
        w = F.softmax(qk, dim=-1)
        wv = w @ v
        wv = rearrange(wv, 'n h l d -> n l (h d)')
        return self.out(wv)


class ResidualAttentionBlock(nn.Module):

    def __init__(
            self, 
            dim_hidden: int, 
            n_head: int,
            ) -> None:
        super().__init__()

        self.attn = MultiHeadAttention(dim_hidden, n_head)
        self.attn_ln = nn.LayerNorm(dim_hidden)

        dim_mlp = dim_hidden * 4
        self.mlp = nn.Sequential(
            nn.Linear(dim_hidden, dim_mlp),
            nn.GELU(),
            nn.Linear(dim_mlp, dim_hidden)
        )
        self.mlp_ln = nn.LayerNorm(dim_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_ln(x))
        x = x + self.mlp(self.mlp_ln(x))
        return x

class MegaByteFAD(ClassificationBase):
    # TODO: multi-task, predict the discrete id?
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.patch_size = getattr(args, 'patch_size', 4)
        self.n_head = getattr(args, 'n_head', 8) # number of attention heads
        self.dim_l_attn = getattr(args, 'dim_attn', 64)
        self.dim_g_attn = self.dim_l_attn * self.patch_size
        self.dim_emb = getattr(args, 'dim_embed', 256) # embedding dim, the dim of frontend out or acoustic feat
        self.global_depth = getattr(args, 'global_depth', 8)
        self.local_depth = getattr(args, 'local_depth', 4)
        self.max_len = getattr(args, 'max_len', 800)
        self.frame_level_fad = getattr(args, 'frame_level_fad', False)
        

        if self.max_len % self.patch_size != 0:
            raise ValueError("max_g_len must be divisible by patch size")

        # global transformer
        self.g_sos = nn.Parameter(torch.randn(1, self.patch_size * self.dim_emb))
        self.g_pos = nn.Parameter(torch.randn(self.max_len//self.patch_size, self.dim_emb*self.patch_size))
        self.g_layers = nn.ModuleList([
            ResidualAttentionBlock(
                self.dim_g_attn,
                self.n_head
            ) for _ in range(self.global_depth)
        ])
        self.g_ln = nn.LayerNorm(self.dim_g_attn)

        # local transformer
        self.l_sos = nn.Parameter(torch.randn(1, 1, self.dim_emb))
        self.l_pos = nn.Parameter(torch.randn(self.patch_size, self.dim_emb))
        self.l_layers = nn.ModuleList([
            ResidualAttentionBlock(
                self.dim_l_attn,
                self.n_head
            ) for _ in range(self.local_depth)
        ])
        self.l_ln = nn.LayerNorm(self.dim_l_attn)
        
        if self.patch_size*self.dim_emb != self.dim_g_attn:
            self.g_proj = nn.Linear(self.dim_emb*self.patch_size, self.dim_g_attn)
        else:
            self.g_proj = None
        if self.dim_emb != self.dim_l_attn:
            self.l_proj = nn.Linear(self.dim_emb, self.dim_l_attn)
        else:
            self.l_proj = None

        if self.frame_level_fad: 
            self.predict_head = nn.Linear(self.dim_l_attn, self.num_classes)
        else:
            self.flatten_head = nn.Linear(self.dim_l_attn, 1)
            self.frame_lvl_predict = nn.Tanh() 
            self.predict_head = nn.Linear(self.max_len, self.num_classes)

        self.apply(self._init_weights)
        self.to(self.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def global_forward(self, x):
        B, T, D = x.shape
        x = x + self.g_pos[:T, :].unsqueeze(0)
        if self.g_proj:
            #logger.info(f"x.shape: {x.shape}")
            #logger.info(f"g_proj: {self.g_proj}")
            x = self.g_proj(x)
        for layer in self.g_layers:
            x = layer(x)
        x = self.g_ln(x)
        return x
    
    def local_forward(self, x, out_global):
        x = x + self.l_pos
        if self.l_proj:
            x = self.l_proj(x)

        nt, p, d = x.shape # nt, p, dim_attn
        n, t, pd = out_global.shape
        out_global = rearrange(out_global, 'n t (p d) -> (n t) p d', p=self.patch_size)

        x = out_global
        for layer in self.l_layers:
            x = layer(x)
        x = self.l_ln(x)
        x = rearrange(x, '(n t) p d -> n (t p) d', t=t)
        return x

    def predict_forward(self, feats):
        # shape: n, t, d
        if self.frame_level_fad:
            feats_out = self.predict_head(feats)
            return feats, feats_out
        else:
            feats = self.flatten_head(feats)
            feats = feats.squeeze(-1)
            feats_pred = self.frame_lvl_predict(feats)
            feats_out = self.predict_head(feats_pred)
            return feats, feats_out

    def forward(self, input_dict):
        super().forward(input_dict)
        x = input_dict["feats"]
        # N, C, F, T
        if x.shape[1] != 1:
            raise ValueError("Only single channel audio is supported")
        x = rearrange(x, 'n 1 f t -> n f t')
        if x.shape[2] % self.patch_size != 0:
            raise ValueError(f"Input length {x.shape[2]} ({x.shape}) must be divisible by patch size")
        
        x = rearrange(x, 'n f (t p) -> n t (p f)', p=self.patch_size)
        # g_sos: [g_dim]
        x_global = torch.cat((self.g_sos.expand(x.shape[0], 1, x.shape[-1]), x[:, :-1, :]), dim=1)
        out_global = self.global_forward(x_global)
        # out_global shape = n, t, (p f)

        x = rearrange(x, 'n t (p f) -> (n t) p f', p=self.patch_size)
        x = self.local_forward(x, out_global)
        feats, feats_out = self.predict_forward(x)
        return {
            "feats": feats,
            "feats_out": feats_out
        }
    
if __name__ == "__main__":

    def use_frontend(frontend: str, dim_frontend: int, max_len: int = 200):
        n, c, t = 4, 1, 64320
        x = torch.rand(n, c, t)
        args = Namespace(
            cuda=0,
            frontend=frontend,
            dim_embed=dim_frontend,
            max_len=max_len
        )
        model = MegaByteFAD(args)
        source = {
            "feats": x
        }
        out = model(source)
        feat, feat_out = out["feats"], out["feats_out"]
        print(feat.shape)
        print(feat_out.shape)

    #use_frontend("facodec", 256)
    use_frontend("XLSR", 1024)