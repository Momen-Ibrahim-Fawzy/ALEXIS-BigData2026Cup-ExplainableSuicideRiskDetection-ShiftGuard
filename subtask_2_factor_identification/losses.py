"""Multi-label focal loss (Lin et al. 2017, RetinaNet), adapted for Sub2's
24-way sigmoid factor classification.

Untried this project: existing rare-factor handling only reweights the BCE
loss by neg/pos ratio (pos_weight, clipped [1,50]) -- that changes the
*average* gradient magnitude for a class but not *which examples* dominate
it. Focal loss additionally multiplies by (1-p_t)^gamma, which suppresses
loss from already-easy/confident examples and concentrates gradient on hard
misses -- a different, complementary lever for the same extreme imbalance
(rare factors are exactly the case with many confidently-correct negatives
and a handful of hard positives).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossMultiLabel(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        """alpha: optional (num_classes,) tensor, already on the right device,
        giving the positive-class weight per factor (e.g. pw/(pw+1) from the
        same neg/pos ratio used for pos_weight elsewhere in this project).
        gamma: focusing parameter (2.0 is the RetinaNet paper's default)."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)  # = p if target=1 else (1-p)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()
