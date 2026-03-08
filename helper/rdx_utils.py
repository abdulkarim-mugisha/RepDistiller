"""
Utilities for RDX-based curriculum learning and triplet mining in RepDistiller.

Provides embedding extraction from RepDistiller models and orchestrates
RDX per-sample difficulty scoring and triplet table computation.
"""

from __future__ import print_function

import numpy as np
import torch

from rdx.sampling import compute_rdx_scores, compute_rdx_triplet_table


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """
    Extract penultimate-layer features for every sample in *dataloader*.

    Uses the RepDistiller ``model(x, is_feat=True)`` interface, which
    returns ``(feat_list, logits)``.  ``feat[-1]`` is the final feature
    map; if it is spatial (4-D) it is globally average-pooled to a vector.

    Returns:
        embeddings: (n, d) numpy float32 array.
    """
    model.eval()
    all_emb = []
    for data in dataloader:
        input_data = data[0].float().to(device)
        feat, _ = model(input_data, is_feat=True)
        emb = feat[-1]
        if emb.dim() > 2:
            emb = emb.mean(dim=[2, 3])
        all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0).numpy()


def compute_rdx_curriculum_order(model_s, model_t, embed_loader, device,
                                  anchor_n=0, gamma=0.1, beta=5.0,
                                  kna_k=8, random_state=42):
    """
    Score all training samples by RDX difficulty and return them
    ordered easy → hard.

    Steps:
        1. Extract student and teacher embeddings.
        2. Optionally select M anchor points (``anchor_n > 0``).
        3. Row-at-a-time RDX scoring (rank-norm → diff → affinity → KNA).
        4. Sort ascending (low score = easy, high score = hard).

    Args:
        model_s: student model (supports ``is_feat=True``).
        model_t: teacher model (supports ``is_feat=True``).
        embed_loader: deterministic DataLoader (test-time transforms,
                      ``shuffle=False``).
        device: torch device.
        anchor_n: number of anchor samples for scoring.
                  0 (default) = use all N samples as anchors.
        gamma: RDX difference saturation parameter.
        beta:  RDX affinity sharpness parameter.
        kna_k: KNA neighbourhood size.
        random_state: seed for anchor selection.

    Returns:
        sorted_indices: np.ndarray of dataset indices, easy → hard.
        scores: np.ndarray of per-sample difficulty scores.
    """
    print("==> Curriculum (RDX): extracting student embeddings...")
    emb_s = extract_embeddings(model_s, embed_loader, device)
    print(f"    shape = {emb_s.shape}")

    print("==> Curriculum (RDX): extracting teacher embeddings...")
    emb_t = extract_embeddings(model_t, embed_loader, device)
    print(f"    shape = {emb_t.shape}")

    N = len(emb_s)

    anchor_idx = None
    if 0 < anchor_n < N:
        np.random.seed(random_state)
        anchor_idx = np.sort(
            np.random.choice(N, size=anchor_n, replace=False))
        print(f"==> Curriculum (RDX): using {anchor_n} anchor points")
    else:
        print(f"==> Curriculum (RDX): using all {N} samples as anchors")

    print("==> Curriculum (RDX): computing RDX scores...")
    scores = compute_rdx_scores(
        emb_s, emb_t,
        anchor_idx=anchor_idx,
        gamma=gamma,
        beta=beta,
        kna_k=kna_k,
    )

    sorted_indices = np.argsort(scores)
    print(f"    difficulty range: [{scores.min():.4f}, {scores.max():.4f}]")

    return sorted_indices, scores


def compute_rdx_triplet_lookup(model_s, model_t, embed_loader, device,
                                anchor_n=0, gamma=0.1, beta=5.0,
                                random_state=42):
    """
    Compute per-sample positive and negative indices for RDX triplet loss.

    Steps:
        1. Extract student and teacher embeddings.
        2. Optionally select M anchor points.
        3. Row-at-a-time RDX diff computation → argmin (positive) and
           argmax (negative) per sample.

    Returns:
        pos_idx: (N,) int64 array — positive index for each sample.
        neg_idx: (N,) int64 array — negative index for each sample.
        weights: (N,) float32 array — per-sample RDX affinity weights.
    """
    print("==> RDX Triplet: extracting student embeddings...")
    emb_s = extract_embeddings(model_s, embed_loader, device)
    print(f"    shape = {emb_s.shape}")

    print("==> RDX Triplet: extracting teacher embeddings...")
    emb_t = extract_embeddings(model_t, embed_loader, device)
    print(f"    shape = {emb_t.shape}")

    N = len(emb_s)

    anchor_idx = None
    if 0 < anchor_n < N:
        np.random.seed(random_state)
        anchor_idx = np.sort(
            np.random.choice(N, size=anchor_n, replace=False))
        print(f"==> RDX Triplet: using {anchor_n} anchor points")
    else:
        print(f"==> RDX Triplet: using all {N} samples as anchors")

    print("==> RDX Triplet: mining positive/negative pairs...")
    pos_idx, neg_idx, weights = compute_rdx_triplet_table(
        emb_s, emb_t,
        anchor_idx=anchor_idx,
        gamma=gamma,
        beta=beta,
    )

    return pos_idx, neg_idx, weights
