"""
Sub2 classifier with cross-task context (v15): the same FactorClassifier
architecture as model.py, but with a small auxiliary feature vector —
Sub1a's predicted risk-level probability distribution, plus a has_evidence
flag from Sub1b — concatenated before the classification head.

Different mechanism from H-MMoE (v6), which shared one backbone across all
3 tasks and underperformed. Here Sub2 keeps its own dedicated backbone;
only a small already-computed prediction from the other subtasks is
injected as an auxiliary input, testing whether cross-task correlation
(e.g. "suicide means (with access)" correlating with behavior/attempt risk)
helps without diluting task-specific capacity.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from config import MODEL_NAME, NUM_FACTORS

CONTEXT_DIM = 5  # 4 risk-level probs (softmax) + 1 has_evidence flag


class FactorClassifierWithContext(nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout=0.15, context_dim=CONTEXT_DIM, context_embed=32):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, context_embed),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + context_embed, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, NUM_FACTORS),
        )

    def forward(self, input_ids, attention_mask, context, token_type_ids=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        seq = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (seq * mask).sum(1) / mask.sum(1).clamp(min=1)
        ctx = self.context_proj(context)
        combined = torch.cat([pooled, ctx], dim=-1)
        return self.head(combined)  # (B, 24)
