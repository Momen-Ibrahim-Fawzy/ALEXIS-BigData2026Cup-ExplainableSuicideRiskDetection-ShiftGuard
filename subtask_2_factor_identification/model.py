"""Subtask 2: DeBERTa multi-label factor classifier (mean-pooled)."""

import torch
import torch.nn as nn
from transformers import AutoModel
from config import MODEL_NAME, NUM_FACTORS


class FactorClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout=0.15):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, NUM_FACTORS),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        seq = out.last_hidden_state
        # Mean pooling (not [CLS]) — better for multi-label factors
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (seq * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)  # (B, 24)
