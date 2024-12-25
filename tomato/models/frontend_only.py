# encoding: utf-8
# Author: Yixuan
#
#
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq
from einops import rearrange

from tomato.utils import utils, logger

from .base import ClassificationBase
import uuid

class XLSRAdapterBase(ClassificationBase):

    def __init__(self, args):
        self.frontend = getattr(args, "frontend", "XLSR")
        super().__init__(args)
        self.frontend_dim = getattr(args, "frontend_dim", 1024)
        self.low_dim = 256
        self.attn_num_heads = 4
        self.attn_hiddim = 256
        # forward till the specified layer (included) 
        # e.g., if the layer_id is 2, then layer 0, 1, 2 will be included
        self.layer_id = getattr(args, "layer_id", None) 
        if self.layer_id is None:
            self.num_layers = getattr(args, "num_layers", 25)
        else:
            self.num_layers = self.layer_id + 1


    def extract_features(self, source: dict):
        if self.have_padding_mask and "padding_mask" not in source:
            raise ValueError("Need padding mask")
        source = utils.move_to_cuda(source, self.device)
        feats = source["feats"]
        # N, C, T
        feats = feats.squeeze(1) # N, T
        padding_mask = None
        with torch.no_grad():
            if self.frontend == "XLSR":
                if self.have_padding_mask:
                    # debug
                    # logger.info("Using padding mask")
                    out_dict = self.frontend_model.model(feats, padding_mask=source['padding_mask'], mask=False, features_only=True)
                else:
                    out_dict = self.frontend_model.model(feats, mask=False, features_only=True)

                post_feats = out_dict['post_features']
                N, T, D = post_feats.shape
                post_feats = rearrange(post_feats, 'n t d -> t n d')

                hiddens = [layer_out[0] for layer_out in out_dict['layer_results']]
                hiddens = [post_feats] + hiddens

                if self.have_padding_mask:
                    padding_mask = out_dict['padding_mask']
            elif self.frontend == "whisper" or self.frontend == "tfwhisper":
                hiddens = self.frontend_model.extract_feat(feats, return_layer_outs=True, layer_id=self.layer_id)
                if self.have_padding_mask:
                    raise ValueError("Padding mask is not supported for whisper frontend")

        if self.have_padding_mask:
            # padding mask shape: [B, T]
            # hiddens[0] shape: [T, B, 1024]
            if padding_mask is None:
                raise ValueError("Padding mask is None")
            if padding_mask.shape[:2] != hiddens[0].shape[:2][::-1]:
                raise ValueError(f"Padding mask shape {padding_mask.shape[:2]} does not match hiddens shape {hiddens[0].shape[:2][::-1]}")
            return hiddens, padding_mask
        else:
            return hiddens


class XLSRAdapter(XLSRAdapterBase):

    def __init__(self, args):
        super().__init__(args) 
        self.gamma = nn.Parameter(torch.randn(self.num_layers))
        self.proj = nn.Linear(self.frontend_dim, self.low_dim)
        self.proj_relu = nn.ReLU()
        self.attn_ln1 = nn.Linear(self.low_dim, self.attn_hiddim)
        self.attn_relu = nn.ReLU()
        self.attn_ln2 = nn.Linear(self.attn_hiddim, self.attn_num_heads)
        self.to(self.device)

    def forward(self, source: dict, **kwards) -> dict:
        hiddens = self.extract_features(source)
        T, N, D = hiddens[0].shape

        gamma_prob = F.softmax(self.gamma, dim=0)

        e = torch.einsum('c, ctnd -> tnd', gamma_prob, torch.stack(hiddens, dim=0))

        e = e.view(-1, D)
        e_proj = self.proj(e) # shape [T*N, 256]
        e_proj = self.proj_relu(e_proj)

        # do attention
        attn = self.attn_ln1(e_proj) # T*N, attn_hiddim
        attn = self.attn_relu(attn)
        attn = self.attn_ln2(attn) # T*N, attn_num_head
        attn = attn.view(T, N, -1) # T, N, attn_num_head
        score = torch.logsumexp(attn, dim=2) # T, N
        score = rearrange(score, 't n -> n t')
        score = F.softmax(score, dim=1) # N, T
        score = rearrange(score, 'n t -> t n 1') # T, N, 1
        e_proj = e_proj.view(T, N, -1) # T, N, 256
        weighted_e = e_proj * score # T, N, 256
        feats = torch.sum(weighted_e, dim=0) # N, 256

        return {
            "feats": feats,
        }
    
class XLSRTimeFirst(XLSRAdapterBase):

    def __init__(self, args):
        super().__init__(args)
        self.attn_head = 4
        self.attn_lns = nn.Parameter(torch.Tensor(self.num_layers, self.frontend_dim, self.attn_head))
        nn.init.xavier_uniform_(self.attn_lns)
        self.gamma = nn.Parameter(torch.randn(self.num_layers))
        self.post_proj = nn.Linear(self.frontend_dim, self.low_dim)
        self.to(self.device)

    def forward(self, source: dict, **kwargs):
        hiddens = self.extract_features(source)
        T, N, D = hiddens[0].shape
        L = self.num_layers
        stacked_h = torch.stack(hiddens, dim=0) # L, T, N, D
        stacked_h = rearrange(stacked_h, 'l t n d -> l (n t) d')

        attn = torch.bmm(stacked_h, self.attn_lns) # L, N*T, A
        score = torch.logsumexp(attn, dim=-1) # L, N*T
        score = rearrange(score, 'l (n t) -> l n t', n=N, t=T)
        score = F.softmax(score, dim=-1) # L, N, T
        
        stacked_h = stacked_h.view(L, N, T, D)
        weighted_h = stacked_h * score.unsqueeze(-1) #L, N, T, D 
        utt_h = torch.sum(weighted_h, dim=2) # L, N, D

        gamma_prob = F.softmax(self.gamma, dim=0) # L
        e = torch.einsum('l, lnd -> nd', gamma_prob, utt_h)
        feats = self.post_proj(e)

        return {
            "feats": feats,
        }


# try relu first
# then snake later

class XLSRAllAttn(XLSRAdapterBase):

    def __init__(self, args):
        super().__init__(args)
        self.layer_attn_head = 4
        self.time_attn_head = 4

        # time attn for each layer, num param: L * 1024 * A
        self.time_attn_ln1 = nn.Parameter(torch.Tensor(self.num_layers, self.frontend_dim, self.layer_attn_head))
        nn.init.xavier_uniform_(self.time_attn_ln1)

        # layer attn input, N, L, D, num parame: 1024*attn_hidden, attn_hidden*A
        self.layer_attn_ln1 = nn.Linear(self.frontend_dim, self.attn_hiddim) 
        self.layer_attn_relu1 = nn.ReLU()
        self.layer_attn_ln2 = nn.Linear(self.attn_hiddim, self.time_attn_head)
        
        self.post_proj = nn.Linear(self.frontend_dim, self.low_dim)
        self.to(self.device)

        # for debug
        save_gamma_path = getattr(args, "save_gamma_path", None)
        if save_gamma_path is not None:
            self.save_gamma_path = save_gamma_path
            self.save_layer_score = True
        else:
            self.save_layer_score = False

    def forward(self, source: dict, **kwargs):
        hiddens = self.extract_features(source)
        T, N, D = hiddens[0].shape
        L = self.num_layers
        stacked_h = torch.stack(hiddens, dim=0) # L, T, N, D
        # print(f"stacked_h shape: {stacked_h.shape}")
        stacked_h = rearrange(stacked_h, 'l t n d -> l (n t) d')
        # print(f"stacked_h rearranged shape: {stacked_h.shape}")

        # time attn
        time_attn = torch.bmm(stacked_h, self.time_attn_ln1) # L, N*T, A
        # print(f"time_attn shape: {time_attn.shape}")
        time_score = torch.logsumexp(time_attn, dim=-1) # L, N*T
        # print(f"time_score shape: {time_score.shape}")
        time_score = rearrange(time_score, 'l (n t) -> l n t', n=N, t=T)
        # print(f"time_score rearranged shape: {time_score.shape}")
        time_score = F.softmax(time_score, dim=-1) # L, N, T
        # print(f"time_score softmax shape: {time_score.shape}")
        stacked_h = stacked_h.view(L, N, T, D)
        # print(f"stacked_h reshaped shape: {stacked_h.shape}")
        weighted_h = stacked_h * time_score.unsqueeze(-1) # L, N, T, D
        # print(f"weighted_h shape: {weighted_h.shape}")
        utt_h = torch.sum(weighted_h, dim=2) # L, N, D
        # print(f"utt_h shape: {utt_h.shape}")
        
        # layer attn
        utt_h = rearrange(utt_h, 'l n d -> (l n) d') # L*N, D
        # print(f"utt_h rearranged shape: {utt_h.shape}")
        layer_hidden = self.layer_attn_ln1(utt_h) # L*N, attn_hiddim
        # print(f"layer_hidden shape: {layer_hidden.shape}")
        layer_hidden = self.layer_attn_relu1(layer_hidden)
        # print(f"layer_hidden relu shape: {layer_hidden.shape}")
        layer_attn = self.layer_attn_ln2(layer_hidden) # L*N, A
        # print(f"layer_attn shape: {layer_attn.shape}")
        layer_score = torch.logsumexp(layer_attn, dim=-1) # L*N
        # print(f"layer_score shape: {layer_score.shape}")
        layer_score = rearrange(layer_score, '(l n) -> n l', l=L, n=N)
        # print(f"layer_score rearranged shape: {layer_score.shape}")
        layer_score = F.softmax(layer_score, dim=-1) # N, L
        # print(f"layer_score softmax shape: {layer_score.shape}")
        layer_score = rearrange(layer_score, 'n l -> l n') # L, N
        
        # for debug
        if self.save_layer_score:
            unique_id = uuid.uuid4().hex
            save_path = f"{self.save_gamma_path}/{unique_id}.pt"
            torch.save(layer_score, save_path)
            
        # print(f"layer_score final rearranged shape: {layer_score.shape}")
        utt_h = rearrange(utt_h, '(l n) d -> l n d', l=L, n=N) # L, N, D
        # print(f"utt_h final rearranged shape: {utt_h.shape}")
        weighted_utt = utt_h * layer_score.unsqueeze(-1)
        # print(f"weighted_utt shape: {weighted_utt.shape}")
        e = torch.sum(weighted_utt, dim=0) # N, D
        # print(f"e shape: {e.shape}")

        feats = self.post_proj(e)
        # print(f"feats shape: {feats.shape}")

        return {
            "feats": feats,
        }

class XLSRTimeAttnOnly(XLSRAdapterBase):
    
    def __init__(self, args):
        super().__init__(args)
        self.layer_attn_head = getattr(args, "layer_attn_head", 8)
        # when to project time
        self.reduce_dim = getattr(args, "reduce_dim", False)
        attn_input_dim = self.frontend_dim
        self.have_padding_mask = getattr(args, "have_padding_mask", False)
        if self.reduce_dim: 
            logger.info("Using bottleneck to reduce time dimension")
            self.bottleneck_dim = args.bottleneck_dim
            self.time_proj = nn.Parameter(torch.Tensor(self.num_layers, self.frontend_dim, self.bottleneck_dim))
            nn.init.xavier_uniform_(self.time_proj)
            self.time_ln = nn.LayerNorm(self.bottleneck_dim, elementwise_affine=False)
            attn_input_dim = self.bottleneck_dim

        # how to calculate attention weights
        attn_cfg = getattr(args, "attn_cfg", "default")
        if attn_cfg == "default":
            # time attn for each layer, num param: L * D * A
            self.time_attn_mlp1 = nn.Parameter(torch.Tensor(self.num_layers, attn_input_dim, self.layer_attn_head))
            nn.init.xavier_uniform_(self.time_attn_mlp1)
        elif attn_cfg == "LN-ASP": # layer norm, ln, relu, ln
            logger.info(f"Using LN-ASP for time attn with num heads: {self.layer_attn_head}")
            self.time_attn_ln = nn.LayerNorm(self.frontend_dim, elementwise_affine=False)
            self.time_attn_mlp1 = nn.Parameter(torch.Tensor(self.num_layers, attn_input_dim, 256))
            self.time_attn_relu = nn.ReLU()
            self.time_attn_mlp2 = nn.Parameter(torch.Tensor(self.num_layers, 256, self.layer_attn_head))
            nn.init.xavier_uniform_(self.time_attn_mlp1)
            nn.init.xavier_uniform_(self.time_attn_mlp2)
        self.attn_cfg = attn_cfg
        self.to(self.device)

    def forward(self, source, **kwargs):
        #print(f'feats shape: {source["feats"].shape}')
        if self.have_padding_mask:
            hiddens, padding_mask = self.extract_features(source)
            # after extract_feat func, padding_mask True is the voiced frames
            #valid_steps = (~padding_mask).sum(dim=-1)
            #if (valid_steps == 0).any():
            #    raise ValueError("All steps are masked")
        else:
            hiddens = self.extract_features(source)

        T, N, D = hiddens[0].shape
        L = self.num_layers
        stacked_h = torch.stack(hiddens, dim=0) # L, T, N, D
        stacked_h = rearrange(stacked_h, 'l t n d -> l (n t) d')
        if self.reduce_dim:
            stacked_h = torch.bmm(stacked_h, self.time_proj) # L, N*T, D
            stacked_h = self.time_ln(stacked_h)
            D = self.bottleneck_dim

        # time attn
        if self.attn_cfg == "default":
            time_attn = torch.bmm(stacked_h, self.time_attn_mlp1) # L, N*T, A
        elif self.attn_cfg == "LN-ASP":
            raise ValueError("This setting is bad, should be banned")
            stacked_h = self.time_attn_ln(stacked_h)
            stacked_h = rearrange(stacked_h, 'l (t n) d -> l (n t) d')
            #print(f"after layer norm: {stacked_h.shape}")
            time_attn = torch.bmm(stacked_h, self.time_attn_mlp1) # L, N*T, 256
            #print(f"stacked_h shape after mlp1: {time_attn.shape}")
            time_attn = self.time_attn_relu(time_attn)  
            time_attn = torch.bmm(time_attn, self.time_attn_mlp2)  # L, N*T, A
            #print(f"time_attn shape after mlp2: {time_attn.shape}")
        
        time_score = torch.logsumexp(time_attn, dim=-1) # L, N*T
        time_score = rearrange(time_score, 'l (n t) -> l n t', n=N, t=T)
        if self.have_padding_mask:
            time_score = time_score.masked_fill(padding_mask.unsqueeze(0), float('-inf'))
        time_score = F.softmax(time_score, dim=-1) # L, N, T
        stacked_h = stacked_h.view(L, N, T, D)
        weighted_h = stacked_h * time_score.unsqueeze(-1) # L, N, T, D
        utt_h = torch.sum(weighted_h, dim=2) # L, N, D
        utt_h = rearrange(utt_h, 'l n d -> n l d') # N, L, D

        return {
            "feats": utt_h,
        }

class XLSRTimeAttnWithBottleneck(XLSRTimeAttnOnly):
    def __init__(self, args):
        super().__init__(args)
        proj_cfg = getattr(args, "proj_cfg", "default")   
        self.bottleneck_dim = args.bottleneck_dim
        if proj_cfg == "default":
            self.bottleneck_proj = nn.Linear(self.frontend_dim, self.bottleneck_dim)
        elif proj_cfg == "NN": # relu, dropout, proj
            logger.info("Using NN for bottleneck projection")
            self.bottleneck_proj = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.frontend_dim, self.bottleneck_dim),
            ) 
        self.bottleneck_recon = nn.Linear(self.bottleneck_dim, self.frontend_dim)
        self.to(self.device)

    def forward(self, source, **kwargs):
        # Get original utt_h from parent class
        out = super().forward(source, **kwargs)
        utt_h = out["feats"]  # N, L, D

        # Project to bottleneck
        bottleneck_feat = self.bottleneck_proj(utt_h)          # N, L, B
        utt_h_reconstructed = self.bottleneck_recon(bottleneck_feat)  # N, L, D

        return {
            "feats": bottleneck_feat,
            "orign_feats": utt_h,
            "recon_feats": utt_h_reconstructed,
        }


def reconstruct_loss(bottleneck_feat, original_feat):
    # Assume original_feat and bottleneck_feat have the same shape
    return F.mse_loss(bottleneck_feat, original_feat)


if __name__ == "__main__":
    #n, c, l = 3, 1, 64000
    #x = torch.randn(n, c, l)

    x_lengths = [1240, 533, 128000]
    max_l = max(x_lengths)
    x_features = []
    for l in x_lengths:
        x = torch.randn(l)
        x_features.append(x)
    
    x_batch = torch.zeros(len(x_features), max_l)
    for i, x in enumerate(x_features):
        x_batch[i, :len(x)] = x
    
    x_mask = torch.zeros_like(x_batch)
    #print(x_batch)
    for i, l in enumerate(x_lengths):
        x_mask[i, l:] = 1
    
    frontend_path = os.environ.get('FRONTEND_PATH')
    if frontend_path is None:
        raise ValueError("FRONTEND_PATH environment variable must be set")
    
    args = argparse.Namespace(
        cuda=0,
        num_classes=1,
        frontend='XLSR',
        #frontend='whisper',
        frontend_path=frontend_path,
        reduce_dim=True,
        bottleneck_dim=256,
        have_padding_mask=True,
    )
    #model = XLSRTimeFirst(args)
    #model = XLSRAllAttn(args)
    model = XLSRTimeAttnOnly(args)
    #out = model({"feats": x})
    print(x_batch.unsqueeze(1).shape)
    out = model(
        {
            "feats": x_batch.unsqueeze(1),
            "padding_mask": x_mask,
         })
    
    for key in out:
        print(key, out[key].shape)
