"""
RDX-based per-sample difficulty scoring and triplet mining for distillation.

Implements the core RDX math (rank-normalized distances, locally-biased
differences, affinities) in a memory-efficient row-at-a-time fashion.
No clustering step — each sample gets a scalar KNA score on the
teacher-unique affinity, and optionally a (positive, negative) pair for
triplet-based representation alignment.

The full N x N affinity matrix is never materialised.
"""

from __future__ import print_function

import numpy as np
import torch


def _prepare_anchors(emb_s, emb_t, anchor_idx):
    """Resolve anchor tensors and build index-mapping helpers."""
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

    return emb_s_t, emb_t_t, anc_s, anc_t, M, anchor_set, anchor_to_col, device


def _compute_diff_batch(emb_s_t, emb_t_t, anc_s, anc_t, start, end,
                         gamma, device):
    """
    Compute the RDX locally-biased difference for one batch of rows.

    Returns *diff* (B, M) where ``diff > 0`` ⇒ student-unique and
    ``diff < 0`` ⇒ teacher-unique.
    """
    batch_s = emb_s_t[start:end].to(device)
    batch_t = emb_t_t[start:end].to(device)

    d_s = torch.cdist(batch_s, anc_s)
    d_t = torch.cdist(batch_t, anc_t)

    r_s = d_s.argsort(dim=1).argsort(dim=1).float()
    r_t = d_t.argsort(dim=1).argsort(dim=1).float()

    denom = torch.min(r_s, r_t) + 1
    return torch.tanh(gamma * (r_t - r_s) / denom)


def _mask_self(mat, start, M, anchor_set, anchor_to_col, device, fill):
    """Set self-entries to *fill* in-place."""
    B = mat.shape[0]
    if anchor_to_col is not None:
        for local_i in range(B):
            g = start + local_i
            if g in anchor_set:
                mat[local_i, anchor_to_col[g]] = fill
    else:
        idx = torch.arange(B, device=device)
        mat[idx, torch.arange(start, start + B, device=device)] = fill


# ── Public API ─────────────────────────────────────────────────────────


def compute_rdx_scores(emb_s, emb_t, anchor_idx=None,
                        gamma=0.1, beta=5.0, kna_k=8, batch_size=512):
    """
    Score every sample by teacher-unique RDX affinity, row-at-a-time.

    High score → teacher has learned structure here that the student
    has not → "hard" for distillation.

    Args:
        emb_s: (N, d) student embeddings (numpy float32).
        emb_t: (N, d) teacher embeddings (numpy float32).
        anchor_idx: optional 1-D array of M anchor indices (None = all N).
        gamma: RDX difference saturation.
        beta:  RDX affinity sharpness.
        kna_k: top-k affinities to sum per sample.
        batch_size: rows per iteration.

    Returns:
        scores: (N,) float32 array — per-sample difficulty.
    """
    N = len(emb_s)
    (emb_s_t, emb_t_t, anc_s, anc_t,
     M, anchor_set, anchor_to_col, device) = _prepare_anchors(
        emb_s, emb_t, anchor_idx)

    scores = np.zeros(N, dtype=np.float32)
    k = min(kna_k, M - 1)
    if k == 0:
        return scores

    print(f"    scoring {N} samples against {M} anchors "
          f"(gamma={gamma}, beta={beta}, kna_k={k})")

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        diff = _compute_diff_batch(emb_s_t, emb_t_t, anc_s, anc_t,
                                    start, end, gamma, device)
        aff = torch.exp(-beta * diff)
        _mask_self(aff, start, M, anchor_set, anchor_to_col, device, 0.0)

        topk_vals, _ = torch.topk(aff, k, dim=1)
        scores[start:end] = topk_vals.sum(dim=1).cpu().numpy()

    return scores


def compute_rdx_triplet_table(emb_s, emb_t, anchor_idx=None,
                               gamma=0.1, beta=5.0, batch_size=512):
    """
    For every sample, find its RDX positive and negative.

    * **positive** — argmax of teacher-unique affinity (am_10):
      the anchor point whose teacher groups it with the sample but
      the student does not.  The student should *pull* toward it.
    * **negative** — argmax of student-unique affinity (am_01):
      the anchor point whose student groups it with the sample but
      the teacher does not.  The student should *push* away from it.

    Both are derived from a single diff vector per row:
    ``am_10 = exp(-β·diff)`` and ``am_01 = exp(β·diff)`` so
    positive = argmin(diff), negative = argmax(diff).

    Args:
        emb_s, emb_t, anchor_idx, gamma, beta, batch_size:
            same as :func:`compute_rdx_scores`.

    Returns:
        pos_idx: (N,) int64 array — global dataset index of each positive.
        neg_idx: (N,) int64 array — global dataset index of each negative.
    """
    N = len(emb_s)
    (emb_s_t, emb_t_t, anc_s, anc_t,
     M, anchor_set, anchor_to_col, device) = _prepare_anchors(
        emb_s, emb_t, anchor_idx)

    pos_idx = np.zeros(N, dtype=np.int64)
    neg_idx = np.zeros(N, dtype=np.int64)

    use_anchor_map = anchor_idx is not None

    print(f"    mining triplets for {N} samples against {M} anchors "
          f"(gamma={gamma}, beta={beta})")

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        diff = _compute_diff_batch(emb_s_t, emb_t_t, anc_s, anc_t,
                                    start, end, gamma, device)

        # mask self so it is never selected as pos or neg
        _mask_self(diff, start, M, anchor_set, anchor_to_col, device, 0.0)

        # positive = most teacher-unique = min diff per row
        pos_cols = diff.argmin(dim=1).cpu().numpy()
        # negative = most student-unique = max diff per row
        neg_cols = diff.argmax(dim=1).cpu().numpy()

        if use_anchor_map:
            pos_idx[start:end] = anchor_idx[pos_cols]
            neg_idx[start:end] = anchor_idx[neg_cols]
        else:
            pos_idx[start:end] = pos_cols
            neg_idx[start:end] = neg_cols

    return pos_idx, neg_idx
