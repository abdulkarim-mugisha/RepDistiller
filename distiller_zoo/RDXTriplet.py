from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class RDXTripletLoss(nn.Module):
    """Triplet loss driven by RDX affinity structure.

    For each anchor in the batch, a *positive* (teacher-unique neighbour)
    and a *negative* (student-unique neighbour) are looked up from a
    pre-computed table and the standard triplet margin loss is applied
    on the student's feature space.

    The lookup table is refreshed periodically via :meth:`update_table`.
    Before the first refresh, forward returns zero loss.
    """

    def __init__(self, margin=1.0, feat_dim=None):
        super(RDXTripletLoss, self).__init__()
        self.margin = margin
        self.feat_dim = feat_dim

    def forward(self, f_anchor, f_pos, f_neg):
        """
        Args:
            f_anchor: (B, d) student features for anchor images.
            f_pos:    (B, d) student features for positive images.
            f_neg:    (B, d) student features for negative images.

        Returns:
            Scalar triplet loss averaged over the batch.
        """
        d_pos = F.pairwise_distance(f_anchor, f_pos, p=2)
        d_neg = F.pairwise_distance(f_anchor, f_neg, p=2)
        loss = F.relu(d_pos - d_neg + self.margin).mean()
        return loss
