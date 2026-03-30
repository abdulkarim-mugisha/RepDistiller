from __future__ import print_function

import torch
import torch.nn as nn

from crd.memory import ContrastMemory
from crd.criterion import Embed

eps = 1e-7


class WeightedContrastLoss(nn.Module):
    """Contrastive loss with optional per-sample weighting."""

    def __init__(self, n_data):
        super(WeightedContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x, weights=None):
        """
        Args:
            x: (B, K+1) exp(sim / T) scores where x[:, 0] is positive.
            weights: (B,) per-sample weights (optional).
        """
        bsz = x.shape[0]
        m = x.size(1) - 1
        pn = 1.0 / float(self.n_data)

        p_pos = x.select(1, 0)
        log_d1 = torch.div(p_pos, p_pos.add(m * pn + eps)).log()

        p_neg = x.narrow(1, 1, m)
        log_d0 = torch.div(p_neg.clone().fill_(m * pn), p_neg.add(m * pn + eps)).log()

        per_sample = -(log_d1 + log_d0.sum(dim=1))
        if weights is not None:
            w = weights / (weights.mean() + eps)
            per_sample = per_sample * w

        return per_sample.mean()


class RDXContrastLoss(nn.Module):
    """RDX-weighted multi-negative contrastive loss (teacher -> student)."""

    def __init__(self, opt):
        super(RDXContrastLoss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.contrast = ContrastMemory(
            opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m
        )
        self.criterion = WeightedContrastLoss(opt.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None, weights=None):
        """
        Args:
            f_s: student features [B, s_dim, ...]
            f_t: teacher features [B, t_dim, ...]
            idx: dataset indices for positives [B]
            contrast_idx: negative sample indices [B, nce_k] (optional)
            weights: per-sample RDX weights [B] (optional)
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, _ = self.contrast(f_s, f_t, idx, contrast_idx)
        return self.criterion(out_s, weights)
