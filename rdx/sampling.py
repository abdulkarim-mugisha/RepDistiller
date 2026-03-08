"""
RDX-based per-sample difficulty scoring for curriculum distillation.

Implements the core RDX math (rank-normalized distances, locally-biased
differences, affinities) in a memory-efficient row-at-a-time fashion.
No clustering step — each sample gets a scalar KNA score on the
teacher-unique affinity, measuring how much the teacher's representation
of that region differs from the student's.

The full N x N affinity matrix is never materialised.
"""

from __future__ import print_function

import numpy as np
import torch


def compute_rdx_scores(emb_s, emb_t, anchor_idx=None,
                        gamma=0.1, beta=5.0, kna_k=8, batch_size=512):
    """
    Score every sample by teacher-unique RDX affinity, row-at-a-time.

    For each sample *i* the function computes its affinity to a set of
    anchor points (default: all N samples), then aggregates via KNA
    (sum of the *kna_k* highest affinities).  Affinity captures
    **teacher-unique** structure — pairs close in the teacher's space
    but far in the student's.  A high score means the teacher has
    learned something about that region that the student has not,
    making it "hard" for distillation.

    Memory is O(batch_size × M) per iteration — the full N × M matrix
    is never stored.

    Args:
        emb_s: (N, d) student embeddings (numpy float32).
        emb_t: (N, d) teacher embeddings (numpy float32).
        anchor_idx: optional 1-D array/list of M anchor indices.
                    When *None* all N samples are used as anchors.
        gamma: RDX locally-biased difference saturation.
        beta:  RDX affinity sharpness.
        kna_k: number of top affinities to sum per sample.
        batch_size: samples processed per iteration.

    Returns:
        scores: (N,) float32 array — per-sample difficulty (higher = harder).
    """
    N = len(emb_s)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    emb_s_t = torch.as_tensor(emb_s, dtype=torch.float32)
    emb_t_t = torch.as_tensor(emb_t, dtype=torch.float32)

    if anchor_idx is not None:
        anchor_idx = np.asarray(anchor_idx)
        anc_s = emb_s_t[anchor_idx].to(device)
        anc_t = emb_t_t[anchor_idx].to(device)
        M = len(anchor_idx)
        anchor_set = set(anchor_idx.tolist())
        anchor_to_col = {int(g): c for c, g in enumerate(anchor_idx)}
    else:
        anc_s = emb_s_t.to(device)
        anc_t = emb_t_t.to(device)
        M = N
        anchor_set = None
        anchor_to_col = None

    scores = np.zeros(N, dtype=np.float32)
    k = min(kna_k, M - 1)
    if k == 0:
        return scores

    print(f"    scoring {N} samples against {M} anchors "
          f"(gamma={gamma}, beta={beta}, kna_k={k})")

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start

        batch_s = emb_s_t[start:end].to(device)
        batch_t = emb_t_t[start:end].to(device)

        # 1. Pairwise Euclidean distances to anchors
        d_s = torch.cdist(batch_s, anc_s)          # (B, M)
        d_t = torch.cdist(batch_t, anc_t)          # (B, M)

        # 2. Rank-normalize per row (double argsort)
        r_s = d_s.argsort(dim=1).argsort(dim=1).float()
        r_t = d_t.argsort(dim=1).argsort(dim=1).float()

        # 3. Locally-biased difference  (teacher-unique direction)
        #    High affinity when teacher considers pair close but student does not
        denom = torch.min(r_s, r_t) + 1
        diff = torch.tanh(gamma * (r_t - r_s) / denom)

        # 4. Affinity
        aff = torch.exp(-beta * diff)              # (B, M)

        # 5. Zero out self-affinity
        if anchor_to_col is not None:
            for local_i in range(B):
                g = start + local_i
                if g in anchor_set:
                    aff[local_i, anchor_to_col[g]] = 0.0
        else:
            idx = torch.arange(B, device=device)
            aff[idx, torch.arange(start, end, device=device)] = 0.0

        # 6. KNA: sum of top-k affinities
        topk_vals, _ = torch.topk(aff, k, dim=1)
        scores[start:end] = topk_vals.sum(dim=1).cpu().numpy()

    return scores
