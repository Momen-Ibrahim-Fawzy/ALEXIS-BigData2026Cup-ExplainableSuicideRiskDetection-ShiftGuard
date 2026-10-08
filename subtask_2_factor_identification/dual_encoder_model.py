"""
Label-semantic dual-encoder model for Sub2 (backlog item, inspired by
DEXML — Gupta et al., ICLR 2024, "Dual-Encoders for Extreme Multi-label
Classification"). Not applicable at DEXML's original scale (millions of
labels) — we only have 24 — but two ideas transfer directly:

1. Score factors via a shared post/factor text encoder instead of an
   opaque per-label weight vector. Rare factors (5-20 positive examples)
   get a semantic anchor from their own description, not just whatever a
   randomly-initialized weight learns from a handful of examples.

2. DecoupledSoftmax loss: a plain multi-label softmax lets a confidently-
   correct EASY co-occurring positive "steal" probability mass from a
   HARDER co-occurring positive in the same normalization, penalizing the
   easy one for being confident. DecoupledSoftmax excludes other true
   positives from a given positive's denominator (keeps negatives + the
   positive itself), so each positive is judged only against negatives.
   Real posts here average ~3.8 co-occurring factors of very different
   difficulty (e.g. common "hopelessness" alongside rare "sexual
   orientation" in the same post) — exactly the scenario this addresses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DualEncoderFactorModel(nn.Module):
    def __init__(self, model_name, embed_dim=256, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, embed_dim))
        # CLIP-style learnable temperature, clamped to avoid runaway scaling
        self.logit_scale = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(1 / 0.07)))))

    def encode(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        emb = self.proj(pooled)
        return F.normalize(emb, dim=-1)

    def forward(self, post_ids, post_mask, factor_ids, factor_mask):
        """post_ids/post_mask: (B, Lp). factor_ids/factor_mask: (F, Lf) — the F=24
        factor descriptions, re-encoded every call (cheap: F is small and texts
        are short) since the encoder is being fine-tuned."""
        post_emb = self.encode(post_ids, post_mask)      # (B, D)
        factor_emb = self.encode(factor_ids, factor_mask)  # (F, D)
        scale = self.logit_scale.exp().clamp(max=100.0)
        scores = scale * (post_emb @ factor_emb.T)  # (B, F)
        return scores


def decoupled_softmax_loss(scores, labels):
    """scores, labels: (B, L) float/binary. For each positive label j in a
    sample, computes -(s_j - logsumexp(negatives_of_that_sample + s_j)),
    i.e. excludes OTHER true positives from j's denominator so co-occurring
    positives of different difficulty don't compete against each other."""
    B, L = scores.shape
    device = scores.device
    labels_bool = labels.bool()

    total_loss = torch.zeros((), device=device)
    total_count = 0
    for j in range(L):
        is_pos_j = labels_bool[:, j]
        n_pos_j = int(is_pos_j.sum().item())
        if n_pos_j == 0:
            continue
        mask_out = labels_bool.clone()
        mask_out[:, j] = False  # keep column j itself, and all negatives; mask other positives
        masked_scores = scores.masked_fill(mask_out, float("-inf"))
        denom = torch.logsumexp(masked_scores, dim=1)  # (B,)
        loss_j = -(scores[:, j] - denom)  # (B,)
        total_loss = total_loss + loss_j[is_pos_j].sum()
        total_count += n_pos_j

    return total_loss / max(total_count, 1)
