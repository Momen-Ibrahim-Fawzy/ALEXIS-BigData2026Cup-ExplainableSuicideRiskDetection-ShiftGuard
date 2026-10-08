"""Distribution-Balanced Loss (Wu et al., ECCV 2020,
https://arxiv.org/abs/2007.09654) and a weighted multi-label supervised
contrastive loss, for the v2 Sub2 architecture.

DB Loss has three components, reimplemented here from the official
DistributionBalancedLoss repo (resample_loss.py):
  1. Rebalancing weight: inverse-class-frequency, normalized by each
     SAMPLE's "repeat rate" (how oversampled its specific combination of
     positive labels effectively is), sigmoid-mapped into a stable range.
     This is the one mechanism genuinely untried here -- it accounts for
     label CO-OCCURRENCE when rebalancing, unlike our existing pos_weight
     (which only looks at each factor in isolation).
  2. Negative-tolerant regularization: a per-class logit bias (calibrating
     the decision boundary toward each class's true prevalence) plus a
     down-scaling of negative-label logits, preventing the huge volume of
     negatives (24 factors, most posts have only 2-5 positive) from
     dominating the gradient and over-suppressing rare-factor recall.
  3. Optional focal-style modulation -- DISABLED by default here (see
     --db-focal in train_v2.py). This project already tested focal loss in
     isolation for Sub2 this session and it underperformed weighted-BCE
     (0.5619 vs 0.5806 tuned). Keeping it as an explicit opt-in flag rather
     than silently including it respects the user's request to implement
     DB Loss as designed, while not blindly ignoring our own negative
     evidence -- toggle --db-focal to test the "full" paper version.

The paper's default hyperparameters (map_alpha=10.0, map_beta=0.2,
map_gamma=0.1, neg_scale=5.0, init_bias_param=0.1) were tuned on VOC/COCO
(thousands-to-millions of examples per class) -- almost certainly need
retuning for our 1,635-post dataset. Exposed as constructor args, not
hardcoded, specifically so they can be swept if the defaults underperform.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionBalancedLoss(nn.Module):
    def __init__(self, class_freq, train_num, map_alpha=10.0, map_beta=0.2, map_gamma=0.1,
                 neg_scale=5.0, init_bias_param=0.1, use_focal=False, focal_gamma=2.0,
                 balance_param=2.0):
        """class_freq: (num_classes,) tensor, positive count per class in the
        training set (or training fold). train_num: total number of training
        examples used to compute class_freq (must match, for init_bias)."""
        super().__init__()
        self.map_alpha = map_alpha
        self.map_beta = map_beta
        self.map_gamma = map_gamma
        self.neg_scale = neg_scale
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.balance_param = balance_param

        freq_inv = 1.0 / class_freq.clamp(min=1.0)
        self.register_buffer("freq_inv", freq_inv)

        class_freq_c = class_freq.clamp(min=1.0)
        init_bias = -torch.log(train_num / class_freq_c - 1.0) * init_bias_param / neg_scale
        self.register_buffer("init_bias", init_bias)

    def forward(self, logits, labels):
        labels = labels.float()

        # 1. Rebalancing weight
        repeat_rate = torch.sum(labels * self.freq_inv.unsqueeze(0), dim=1, keepdim=True).clamp(min=1e-6)
        pos_weight = self.freq_inv.unsqueeze(0) / repeat_rate
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha

        # 2. Negative-tolerant regularization
        adj_logits = logits + self.init_bias.unsqueeze(0)
        adj_logits = adj_logits * (1 - labels) * self.neg_scale + adj_logits * labels
        weight = weight / self.neg_scale * (1 - labels) + weight * labels

        loss = F.binary_cross_entropy_with_logits(adj_logits, labels, reduction="none")

        # 3. Optional focal modulation (off by default -- see docstring)
        if self.use_focal:
            pt = torch.exp(-loss)
            loss = ((1 - pt) ** self.focal_gamma) * loss * self.balance_param

        return (loss * weight).mean()


class WeightedMultiLabelContrastiveLoss(nn.Module):
    """Soft-weighted supervised contrastive loss (Khosla et al. 2020 SupCon,
    generalized to multi-label via continuous Jaccard-similarity weighting
    instead of a binary same-class indicator -- with 24 factors and small
    batches, exact-match positives are rare, so a soft target lets every
    pair in the batch contribute proportionally to how much their label
    sets overlap, rather than requiring an exact match).

    Applied to the pooled representation as a small-weight AUXILIARY loss
    alongside the main classification loss (see train_v2.py's
    --contrastive-weight), not as the primary objective -- standard
    practice for joint contrastive+classification training.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """embeddings: (B, D) pooled representations (will be L2-normalized
        here). labels: (B, C) binary multi-hot."""
        B = embeddings.size(0)
        if B < 2:
            return embeddings.sum() * 0.0  # no pairs possible, zero grad-safe fallback

        z = F.normalize(embeddings, dim=1)
        sim = (z @ z.t()) / self.temperature  # (B, B)

        labels = labels.float()
        intersection = labels @ labels.t()                                   # (B, B)
        union = labels.sum(1, keepdim=True) + labels.sum(1, keepdim=True).t() - intersection
        jaccard = intersection / union.clamp(min=1e-6)                        # (B, B), in [0,1]

        eye = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        jaccard = jaccard.masked_fill(eye, 0.0)  # exclude self as its own positive

        # Use a large-finite sentinel, not -inf: jaccard is already 0 on the
        # diagonal, but 0 * (-inf) = nan in IEEE float, which would poison
        # every row's weighted sum below even though the diagonal term is
        # masked out. A finite sentinel keeps 0 * sentinel = 0 as intended.
        NEG_INF = torch.finfo(sim.dtype).min / 2
        sim = sim.masked_fill(eye, NEG_INF)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        weight_sum = jaccard.sum(dim=1)
        has_signal = weight_sum > 1e-6
        if not has_signal.any():
            return embeddings.sum() * 0.0  # degenerate batch (no two posts share any factor)

        weighted_log_prob = (jaccard * log_prob).sum(dim=1) / weight_sum.clamp(min=1e-6)
        loss = -weighted_log_prob[has_signal].mean()
        return loss


class PairSupConLoss(nn.Module):
    """Standard supervised contrastive loss (Khosla et al. 2020 SupCon) for
    the NLI cross-encoder's per-(post,factor)-pair setting (Idea 2, isolated
    from Idea "v2"'s combined/failed attempt).

    Each training example here is one (post, factor_description) pair with
    a single SCALAR binary label (entailment / not) -- not the 24-dim
    multi-hot post-level label WeightedMultiLabelContrastiveLoss above was
    built for. Reusing that Jaccard-weighted loss directly would be wrong: a
    1-dim "multi-hot" vector makes negative-negative pairs (both label 0)
    degenerate to intersection=0/union=0, silently zero-weighting the
    majority of same-class pairs. This loss instead uses the plain
    same-label indicator (the correct reduction of SupCon to a single binary
    class), so entailment pairs cluster together and non-entailment pairs
    cluster together, regardless of which post/factor they came from.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """embeddings: (B, D) pooled representations. labels: (B,) binary."""
        B = embeddings.size(0)
        if B < 2:
            return embeddings.sum() * 0.0

        z = F.normalize(embeddings, dim=1)
        sim = (z @ z.t()) / self.temperature  # (B, B)

        labels = labels.view(-1, 1).float()
        same = (labels == labels.t()).float()

        eye = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        same = same.masked_fill(eye, 0.0)

        NEG_INF = torch.finfo(sim.dtype).min / 2
        sim = sim.masked_fill(eye, NEG_INF)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        weight_sum = same.sum(dim=1)
        has_signal = weight_sum > 1e-6
        if not has_signal.any():
            return embeddings.sum() * 0.0

        weighted_log_prob = (same * log_prob).sum(dim=1) / weight_sum.clamp(min=1e-6)
        loss = -weighted_log_prob[has_signal].mean()
        return loss
