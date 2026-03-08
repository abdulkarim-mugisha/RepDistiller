from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class RDXTripletLoss(nn.Module):
    """Affinity-weighted soft contrastive loss driven by RDX structure.

    For each anchor, a *positive* (teacher-unique neighbour) and a
    *negative* (student-unique neighbour) are looked up from a
    pre-computed table.  Each triplet is weighted by its RDX affinity
    magnitude — stronger teacher-student disagreement produces a
    stronger learning signal.

    Uses softplus instead of hard-margin ReLU so gradients never
    vanish, even when the constraint is already satisfied.
    """

    def __init__(self, **kwargs):
        super(RDXTripletLoss, self).__init__()

    def forward(self, f_anchor, f_pos, f_neg, weights):
        """
        Args:
            f_anchor: (B, d) student features for anchor images.
            f_pos:    (B, d) student features for positive images.
            f_neg:    (B, d) student features for negative images.
            weights:  (B,)   per-sample RDX affinity weights.

        Returns:
            Scalar weighted softplus triplet loss.
        """
        d_pos = F.pairwise_distance(f_anchor, f_pos, p=2)
        d_neg = F.pairwise_distance(f_anchor, f_neg, p=2)

        w = weights / (weights.mean() + 1e-8)

        loss = (w * F.softplus(d_pos - d_neg)).mean()
        return loss
