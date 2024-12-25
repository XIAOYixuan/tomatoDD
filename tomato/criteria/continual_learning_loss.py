# encoding: utf-8
# Author: Yixuan
# 
# 
from typing import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseCriterion
from tomato.utils import logger

class OCBCELoss(BaseCriterion):

    def __init__(self, args):
        super().__init__(args)
        self.m_real = getattr(args, "m_real", 0.9)
        self.m_fake = getattr(args, "m_fake", 0.2)
        self.alpha = getattr(args, "alpha", 20.0)
        self.softplus = nn.Softplus()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, net_input: dict, net_output: dict, *kwargs):
        scores = net_output["feats"]
        pred = net_output["feats_out"]
        targ = net_input["labels"]

        # bce loss
        if pred.ndim == 2:
            pred = pred.squeeze(1)
        loss = self.bce(pred, targ.float())

        # oc loss
        negative_scores = scores[targ == 0]
        positive_scores = scores[targ == 1]
        #logger.info(f"negative_scores: {negative_scores.shape}")
        #logger.info(f"positive_scores: {positive_scores.shape}")

        if negative_scores.shape[0] != 0:
            #soft_scores = self.softplus(self.alpha * (self.m_real - negative_scores))
            #mean_scores = soft_scores.mean()
            #logger.info(f"soft_scores: {soft_scores.shape}")
            #logger.info(f"mean_scores: {mean_scores.shape}")
            loss = loss + self.softplus(self.alpha * (self.m_real - negative_scores)).mean()
        if positive_scores.shape[0] != 0:
            loss = loss + self.softplus(self.alpha * (positive_scores - self.m_fake)).mean()

        return loss

    def zero_logging_output(self):
        return {
            "loss": 0.0
        }

class OCSoftmax(BaseCriterion):
    """
    Trainable criterion for one-class classification
    """

    def __init__(self, args):
        super().__init__(args)
        # TODO: make these parameters configurable
        self.feat_dim = getattr(args, "feat_dim", 256)
        self.m_real = getattr(args, "m_real", 0.9)
        self.m_fake = getattr(args, "m_fake", 0.2)
        self.alpha = getattr(args, "alpha", 20.0)
        self.real_weight = getattr(args, "real_weight", 1.0)
        self.fake_weight = getattr(args, "fake_weight", 1.0)
        logger.info(f"weight for real and fake: {self.real_weight}, {self.fake_weight}")
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()
        device = getattr(args, "device", torch.device("cuda:0"))
        self.device = device
        self.to(device)

    def load_checkpoint(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.to(self.device)
        logger.info(f"Loaded checkpoint from {ckpt_path}")

    def forward(self, net_input: dict, net_output: dict, *kwargs):
        """ 
        """
        feats = net_output['feats'] # [batch_size, feat_dim]
        #print("feats shape:", feats.shape)
        labels = net_input['labels'] # [batch_size]
        #print("labels shape:", labels.shape)

        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(feats, p=2, dim=1)
        # get the max and min for each sample
        # neg_feats = x[labels == 0]
        #print("max of neg_feats:", neg_feats.max(dim=1))
        #print("min of neg_feats:", neg_feats.min(dim=1))
        #print("max of w:", w.max(dim=1))
        #print("min of w:", w.min(dim=1))

        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()

        #TODO: no need to compute loss if it's not for training 
        negative_scores = scores[labels == 0]
        positive_scores = scores[labels == 1] 
        #print("negative_scores:", negative_scores)
        #print("after times alpha", self.alpha * (self.m_real - negative_scores))
        #print("positive_scores shape:", positive_scores.shape)
        loss = None
        if negative_scores.shape[0] != 0:
            # in ocs, real is positive, fake is negative
            loss = self.real_weight * self.softplus(self.alpha * (self.m_real - negative_scores)).mean()
        if positive_scores.shape[0] != 0:
            positive_loss = self.fake_weight * self.softplus(self.alpha * (positive_scores - self.m_fake)).mean()
            loss = loss + positive_loss if loss is not None else positive_loss

        assert loss is not None
        # print("loss.item:", loss.item())

        logging_output = {
            "loss": loss.item(),
        }
        
        return loss, output_scores.squeeze(1), logging_output 
    
    def zero_logging_output(self):
        return {
            "loss": 0.0,
        }

class OCSoftmaxK(BaseCriterion):

    def __init__(self, args):
        super().__init__(args)
        self.feat_dim = getattr(args, "feat_dim", 1024)
        self.m_real = getattr(args, "m_real", 0.9)
        self.m_fake = getattr(args, "m_fake", 0.2)
        self.alpha = getattr(args, "alpha", 20.0)
        self.real_weight = getattr(args, "real_weight", 1.0)
        self.fake_weight = getattr(args, "fake_weight", 1.0)
        self.n_center = getattr(args, "n_center", 25)
        logger.info(f"weight for real and fake: {self.real_weight}, {self.fake_weight}")
        self.center = nn.Parameter(torch.randn(self.n_center, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()
        # TODO: change cuda:0 to the assigned cuda device
        #device = torch.device("cuda:0")
        device = getattr(args, "device", torch.device("cuda:0"))
        self.device = device
        self.to(device)

    def forward(self, net_input: dict, net_output: dict, *kwargs):
        """
        feats: [batch_size, k, feat_dim]
        labels: [batch_size]
        self.center: [k, feat_dim], one center per sub-module
        """
        feats = net_output['feats']  # [batch_size, k, feat_dim]
        labels = net_input['labels'] # [batch_size]

        w = F.normalize(self.center, p=2, dim=1)     # [k, feat_dim]
        x = F.normalize(feats, p=2, dim=2)           # [batch_size, k, feat_dim]

        # Compute scores: dot product per sub-module
        # scores: [batch_size, k]
        scores = (x * w.unsqueeze(0)).sum(dim=-1)
        #print('score', scores.shape)
        output_scores = scores.clone()
        #output_scores = output_scores.mean(dim=1)
        output_scores = output_scores.sum(dim=1)
        #print('output_score', output_scores.shape)

        total_loss = None
        # Compute loss for each sub-module and sum
        for i in range(scores.size(1)):
            negative_scores = scores[labels == 0, i]
            positive_scores = scores[labels == 1, i]

            sub_loss = None
            if negative_scores.numel() > 0:
                sub_loss = self.fake_weight * self.softplus(self.alpha * (self.m_real - negative_scores)).mean()
            if positive_scores.numel() > 0:
                p_loss = self.real_weight * self.softplus(self.alpha * (positive_scores - self.m_fake)).mean()
                sub_loss = p_loss if sub_loss is None else sub_loss + p_loss

            if sub_loss is not None:
                total_loss = sub_loss if total_loss is None else total_loss + sub_loss

        assert total_loss is not None
        # print("loss.item:", loss.item())

        logging_output = {
            "loss": total_loss.item(),
        }
        
        return total_loss, output_scores, logging_output 
    
    def zero_logging_output(self):
        return {
            "loss": 0.0,
        }

    def load_checkpoint(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.to(self.device)
        logger.info(f"Loaded checkpoint from {ckpt_path}")


# two class softmax
class TCSoftmax(OCSoftmax):

    def __init__(self, args):
        super().__init__(args)
        self.n_fake_center = getattr(args, "n_fake_center", 1)
        self.fake_center = nn.Parameter(torch.randn(self.n_fake_center, self.feat_dim))
        self.real_center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.fake_center, 0.25)
        self.to(self.device)

    def zero_logging_output(self):
        return super().zero_logging_output()

    def forward(self, net_input: dict, net_output: dict, *kwargs):
        """
        self.real_center, shape: [1, feat_dim]
        self.fake_center, shape: [1, feat_dim]
        """

        feats = net_output['feats'] # [batch_size, feat_dim]
        labels = net_input['labels'] # [batch_size]

        w_real = F.normalize(self.real_center, p=2, dim=1)  
        w_fake = F.normalize(self.fake_center, p=2, dim=1)  
        x = F.normalize(feats, p=2, dim=1)                  

        # real_center
        rc_scores = (x @ w_real.transpose(0, 1)).squeeze(1) # [batch_size]
        # fake center
        fc_scores = (x @ w_fake.transpose(0, 1)).squeeze(1) # [batch_size]

        # Separate real and fake samples
        real_mask = (labels == 0) # actually are real samples, but labeled as negative
        fake_mask = (labels == 1)  
        output_scores = rc_scores.clone() - fc_scores.clone()
        loss = None

        if real_mask.any():
            dist2rc = rc_scores[real_mask]  
            dist2fc = fc_scores[real_mask]  
            
            loss_rc = self.real_weight * self.softplus(self.alpha * (self.m_real - dist2rc)).mean()
            loss_fc = self.real_weight * self.softplus(self.alpha * (dist2fc - self.m_fake)).mean()
            loss = loss_rc + loss_fc

        if fake_mask.any():
            dist2rc = rc_scores[fake_mask]
            dist2fc = fc_scores[fake_mask]

            loss_rc = self.fake_weight * self.softplus(self.alpha * (dist2rc - self.m_fake)).mean()
            loss_fc = self.fake_weight * self.softplus(self.alpha * (self.m_real - dist2fc)).mean()
            
            fake_loss = loss_rc + loss_fc 
            loss = fake_loss if loss is None else (loss + fake_loss)

        return loss, output_scores, {"loss": loss.item()}
 
    
    def load_checkpoint(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.to(self.device)
        logger.info(f"Loaded checkpoint from {ckpt_path}")

class OCSoftmaxRecon(OCSoftmaxK):

    def __init__(self, args):
        super().__init__(args)
        self.recon_weight = getattr(args, "recon_weight", 0.5)
        self.recon_loss = nn.MSELoss()
        self.to(self.device)

    def load_checkpoint(self, ckpt_path):
        return super().load_checkpoint(ckpt_path)
    
    def zero_logging_output(self):
        return {
            "loss": 0.0,
            "recon_loss": 0.0,
            "oc_loss": 0.0,
        }
    
    def forward(self, net_input: dict, net_output: dict, *kwargs):
        """
        feats: [batch_size, feat_dim]
        labels: [batch_size]
        """
        loss, output_scores, logging_output = super().forward(net_input, net_output)
        logging_output['oc_loss'] = loss.item()
        orign_feats = net_output['orign_feats']
        recon_feats = net_output['recon_feats']
        recon_loss = self.recon_weight * self.recon_loss(orign_feats, recon_feats)
        loss = loss + recon_loss
        logging_output["loss"] = loss.item()
        logging_output["recon_loss"] = recon_loss.item()
        return loss, output_scores, logging_output


if __name__ == "__main__":
    from argparse import Namespace
    args = Namespace(device='cpu')
    """
    ocbceloss = OCBCELoss(args)

    scores = torch.randn(4, 30)
    pred = torch.randn(4, 1)
    targ = torch.randint(0, 2, (4,))

    net_output = {
        "feats": scores,
        "feats_out": pred,
    }
    net_input = {
        "labels": targ,
    }

    loss = ocbceloss(net_input, net_output)
    print(loss)
    # ---- 
    ocsk = OCSoftmaxK(args)
    feats = torch.randn(4, 24, 256)
    label = torch.randint(0, 2, (4,))
    net_output = {
        "feats": feats,
    }
    net_input = {
        "labels": label,
    }
    loss, score, log_out = ocsk(net_input, net_output)
    print(loss.shape)
    """
    tcs = TCSoftmax(args)
    feats = torch.randn(4, 256)
    label = torch.randint(0, 2, (4,))
    net_output = {
        "feats": feats,
    }
    net_input = {
        "labels": label,
    }
    loss, score, log_out = tcs(net_input, net_output)
    print(loss.shape)