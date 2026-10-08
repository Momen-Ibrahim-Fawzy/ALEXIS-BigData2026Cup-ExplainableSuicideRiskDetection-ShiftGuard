"""
CORAL (COnsistent RAnk Logits) dual-head classifier for Sub1a (backlog item #2).

Risk levels are ordinal (indicator < ideation < behavior < attempt), not
purely categorical, but every experiment so far (including exp09) trains
with plain 4-way softmax cross-entropy, discarding that structure. Following
the Hierarchical Dual-Head MentalRoBERTa paper (arXiv 2510.20085), this adds
a CORAL ordinal head alongside the standard classification head.

CORAL reduces K-class ordinal regression to K-1 binary "is y > threshold_k"
tasks that share a single projection weight but have their own bias terms,
with the biases constrained to be monotonically decreasing — this
guarantees P(y>0) >= P(y>1) >= P(y>2) by construction (consistent ranks),
unlike training 3 independent binary classifiers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralDualHeadClassifier(nn.Module):
    def __init__(self, model_name, num_classes=4, dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)

        # Standard head (same as the plain Classifier used by exp09 etc.)
        self.standard_head = nn.Linear(h, num_classes)

        # CORAL head: shared scalar score + monotonically-decreasing per-threshold biases
        self.coral_score = nn.Linear(h, 1, bias=False)
        self.coral_bias_base = nn.Parameter(torch.zeros(1))
        self.coral_bias_deltas = nn.Parameter(torch.zeros(num_classes - 2))  # 2 deltas for 3 thresholds

    def pool(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask):
        pooled = self.dropout(self.pool(input_ids, attention_mask))

        standard_logits = self.standard_head(pooled)  # (B, 4)

        score = self.coral_score(pooled).squeeze(-1)  # (B,)
        deltas = F.softplus(self.coral_bias_deltas)  # non-negative, (2,)
        # bias = [base, base - delta0, base - delta0 - delta1] -> strictly decreasing
        bias = torch.cat([
            self.coral_bias_base,
            self.coral_bias_base - torch.cumsum(deltas, dim=0),
        ])  # (3,)
        coral_logits = score.unsqueeze(-1) + bias.unsqueeze(0)  # (B, 3) = logits for P(y>0),P(y>1),P(y>2)

        return standard_logits, coral_logits

    @staticmethod
    def coral_targets(labels, num_classes=4):
        """labels: (B,) int in [0, num_classes-1]. Returns (B, num_classes-1) binary targets
        where target[:, k] = 1 if label > k else 0."""
        thresholds = torch.arange(num_classes - 1, device=labels.device)
        return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()

    @staticmethod
    def coral_logits_to_class_probs(coral_logits):
        """coral_logits: (B, K-1) -> class probs (B, K) via consecutive differences
        of P(y>k) = sigmoid(coral_logits[:,k])."""
        p_gt = torch.sigmoid(coral_logits)  # (B, K-1), decreasing across k
        p_gt_padded = torch.cat([
            torch.ones(p_gt.size(0), 1, device=p_gt.device),
            p_gt,
            torch.zeros(p_gt.size(0), 1, device=p_gt.device),
        ], dim=1)  # (B, K+1): [1, P(y>0), P(y>1), P(y>2), 0]
        class_probs = p_gt_padded[:, :-1] - p_gt_padded[:, 1:]  # (B, K)
        return class_probs.clamp(min=1e-8)

    def combined_probs(self, standard_logits, coral_logits):
        """Average of the standard softmax and the CORAL-derived class distribution."""
        standard_probs = F.softmax(standard_logits, dim=-1)
        coral_probs = self.coral_logits_to_class_probs(coral_logits)
        return 0.5 * standard_probs + 0.5 * coral_probs
