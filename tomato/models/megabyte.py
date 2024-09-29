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


class MegaByteFeat(nn.Module):

    def __init__(self, args: Namespace):
        super().__init__()
        self.patch_size = getattr(args, 'patch_size', 4)
        self.n_head = getattr(args, 'n_head', 8)
        self.patch_size = getattr(args, 'patch_size', 4)
        self.n_head = getattr(args, 'n_head', 8) # number of attention heads
        self.dim_l_attn = getattr(args, 'dim_attn', 64)
        self.dim_g_attn = self.dim_l_attn * self.patch_size
        self.dim_emb = getattr(args, 'dim_embed', 256) # embedding dim, the dim of frontend out or acoustic feat
        self.global_depth = getattr(args, 'global_depth', 4)
        self.local_depth = getattr(args, 'local_depth', 2)
        self.max_len = getattr(args, 'max_len', 800)
        self.focus_domain = getattr(args, 'focus_domain', 'time') # could be freq, time
        if self.focus_domain == 'freq':
            # transpose the input from ncft to nctf
            # swap dim_emb and max_len
            self.dim_emb, self.max_len = self.max_len, self.dim_emb
        if self.max_len % self.patch_size != 0:
            raise ValueError(f"max_len {self.max_len} must be divisible by patch size")
        
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
        self._add_attn_input_project()
    
    def _add_attn_input_project(self):
        if self.patch_size*self.dim_emb != self.dim_g_attn:
            self.g_proj = nn.Sequential(
                nn.Linear(self.dim_emb*self.patch_size, self.dim_g_attn),
                nn.GELU()
            )
        else:
            self.g_proj = None
        if self.dim_emb != self.dim_l_attn:
            self.l_proj = nn.Sequential(
                nn.Linear(self.dim_emb, self.dim_l_attn),
                nn.GELU()
            )
        else:
            self.l_proj = None
        
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
    
    def forward(self, x):
        # feat shape n 1 f t
        if self.focus_domain == 'freq':
            x = rearrange(x, 'n 1 f t -> n 1 t f')

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
        # x shape: n, t, d (dim_l_attn)
        return x
    

class MegaByteFAD(ClassificationBase):
    # TODO: multi-task, predict the discrete id?
    def __init__(self, args: Namespace):
        super().__init__(args)

        self.feat_bn = nn.BatchNorm2d(num_features=1)
        self.feat_selu = nn.SELU(inplace=True)
        global_depth = getattr(args, 'global_depth', 8)
        local_depth = getattr(args, 'local_depth', 4)
        # update global_depth and local_depth
        args.global_depth = global_depth
        args.local_depth = local_depth
        self.mega_feat = MegaByteFeat(args)
        self.frame_level_fad = getattr(args, 'frame_level_fad', False)
        if self.frame_level_fad:
            raise NotImplementedError("Need to adjust the code for frame-level cls, some updates such as focus_domain have conflicts")

        self._prepare_predict_head()
        self._extra_module_init()

        self.apply(self._init_weights)
        self.to(self.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _extra_module_init(self):
        pass

    def _prepare_predict_head(self):
        if self.frame_level_fad: 
            self.predict_head = nn.Linear(self.mega_feat.dim_l_attn, self.num_classes)
        else:
            self.flatten_head = nn.Linear(self.mega_feat.dim_l_attn, 1)
            self.pred_tanh = nn.Tanh() 
            self.predict_head = nn.Linear(self.mega_feat.max_len, self.num_classes)

    def predict_forward(self, feats):
        # shape: n, t, d
        if self.frame_level_fad:
            feats_out = self.predict_head(feats)
            return feats, feats_out
        else:
            feats = self.flatten_head(feats)
            feats = feats.squeeze(-1)
            feats_pred = self.pred_tanh(feats)
            feats_out = self.predict_head(feats_pred)
            return feats, feats_out
        
    def forward(self, input_dict):
        super().forward(input_dict)
        x = input_dict["feats"]

        # post processing the front-end features
        # reference: wav2vecAASIST model
        # n c f t
        x = F.max_pool2d(x, (3, 3))
        x = self.feat_bn(x)
        x = self.feat_selu(x)

        x = self.mega_feat(x)
        feats, feats_out = self.predict_forward(x)
        return {
            "feats": feats,
            "feats_out": feats_out
        }
    

class OCMegaByte(MegaByteFAD):

    def __init__(self, args: Namespace):
        super().__init__(args)
        if self.focus_domain != 'time':
            raise ValueError("Only time focus domain is supported in OC model")

    def _extra_module_init(self):
        self.m_real = getattr(self.args, 'm_real', 0.5)
        self.m_fake = getattr(self.args, 'm_fake', 0.5)
        self.alpha = getattr(self.args, 'alpha', 20.0)
        self.center = nn.Parameter(torch.randn(1, self.dim_l_attn))
        self.softplus = nn.Softplus()

    def _init_weights(self, module):
        super()._init_weights(module)
        nn.init.kaiming_uniform_(self.center, 0.25)

    def _prepare_predict_head(self):
        if self.frame_level_fad:
            self.predict_head = None
        else:
            self.predict_head = nn.Linear(self.max_len, self.num_classes)

    def _oc_forward(self, x):
        # x: B, T, D (dim_l_attn)
        b, t, d = x.shape
        x = rearrange(x, 'b t d -> (b t) d')
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0, 1)
        scores = rearrange(scores, '(b t) 1 -> b t', b=b)
        return scores
    
    def predict_forward(self, x):
        scores = self._oc_forward(x)
        if self.frame_level_fad:
            raise ValueError("Frame level FAD is not supported in OC model")
        else:
            feats_out = self.predict_head(scores)
            return scores, feats_out


class LGMegaByte(MegaByteFAD):

    def __init__(self, args: Namespace):
        super().__init__(args)
        if self.focus_domain != 'time':
            raise ValueError("Only time focus domain is supported in LGMegaByte")

    def _extra_module_init(self):
        self.g_pos = nn.Parameter(torch.randn(self.max_len//self.patch_size, self.dim_l_attn*self.patch_size))
        self.l2g_gelu = nn.GELU()

    def _prepare_predict_head(self):
        if self.frame_level_fad:
            self.predict_head = nn.Linear(self.dim_g_attn, self.num_classes)
        else:
            self.flatten_head = nn.Linear(self.dim_g_attn, 1)
            self.frame_lvl_predict = nn.Tanh() 
            self.predict_head = nn.Linear(self.max_len//self.patch_size, self.num_classes)

    def _add_attn_input_project(self):
        if self.patch_size*self.dim_l_attn != self.dim_g_attn:
            self.g_proj = nn.Sequential(
                nn.Linear(self.dim_l_attn*self.patch_size, self.dim_g_attn),
                nn.GELU()
            )
        else:
            self.g_proj = None
        if self.dim_emb != self.dim_l_attn:
            self.l_proj = nn.Sequential(
                nn.Linear(self.dim_emb, self.dim_l_attn),
                nn.GELU()
            )
        else:
            self.l_proj = None

    def local_forward(self, x):
        x = x + self.l_pos
        if self.l_proj:
            x = self.l_proj(x)
        for layer in self.l_layers:
            x = layer(x)
        x = self.l2g_gelu(x)
        x = self.l_ln(x)
        return x
    
    def global_forward(self, x):
        # x : n, t, D
        x = x + self.g_pos[:x.shape[1], :].unsqueeze(0)
        if self.g_proj:
            x = self.g_proj(x)
        for layer in self.g_layers:
            x = layer(x)
        x = self.g_ln(x)
        return x

    def forward(self, input_dict):
        self.forward_frontend(input_dict)
        x = input_dict["feats"]
        # N, C, F, T
        if x.shape[1] != 1:
            raise ValueError("Only single channel audio is supported")
        x = rearrange(x, 'n 1 f t -> n f t')
        if x.shape[2] % self.patch_size != 0:
            raise ValueError(f"Input length {x.shape[2]} ({x.shape}) must be divisible by patch size")
        N, F, T = x.shape
        #logger.info(f"x.shape: {x.shape}") 
        x = rearrange(x, 'n f (t p) -> (n t) p f', p = self.patch_size)
        #logger.info(f"(n t) p f: x.shape: {x.shape}")
        x = self.local_forward(x)
        #logger.info(f"local_forward: x.shape: {x.shape}")

        num_patches = T // self.patch_size
        x = rearrange(x, '(n t) p f -> n t (p f)', t = num_patches, p = self.patch_size)
        x = self.global_forward(x)
        feats, feats_out = self.predict_forward(x)
        return {
            "feats": feats,
            "feats_out": feats_out
        }
    
if __name__ == "__main__":

    def use_acoustic(dim_embed: 128, max_len: int = 200):
        n, c, f, t = 2, 1, 128, max_len
        x = torch.rand(n, c, f, t)
        args = Namespace(
            cuda=0,
            dim_embed=dim_embed,
            max_len=max_len,
            focus_domain="freq",
        )
        model = MegaByteFAD(args)
        print(model)
        source = {
            "feats": x
        }
        out = model(source)
        feat, feat_out = out["feats"], out["feats_out"]
        print(feat.shape)
        print(feat_out.shape)

    def use_frontend(frontend: str, dim_frontend: int, max_len: int = 200):
        n, c, t = 4, 1, 64320
        x = torch.rand(n, c, t)
        args = Namespace(
            cuda=0,
            frontend=frontend,
            dim_embed=dim_frontend,
            max_len=max_len,
            patch_size=3
            #focus_domain="freq",
        )
        model = MegaByteFAD(args)
        print(model)
        #model = LGMegaByte(args)
        #model = OCMegaByte(args)
        source = {
            "feats": x
        }
        out = model(source)
        feat, feat_out = out["feats"], out["feats_out"]
        print(feat.shape)
        print(feat_out.shape)

    #use_frontend("facodec", 256)
    #use_frontend("XLSR", 1024)
    use_frontend("XLSR", 341, 66)
    #use_acoustic(128, 200)