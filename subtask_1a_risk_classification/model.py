"""Subtask 1a: DeBERTa-v3-large for 4-class risk classification."""

import torch
import torch.nn as nn
from transformers import AutoModel
from config import MODEL_NAME, NUM_CLASSES


class RiskClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, NUM_CLASSES)

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(self.drop(cls))
