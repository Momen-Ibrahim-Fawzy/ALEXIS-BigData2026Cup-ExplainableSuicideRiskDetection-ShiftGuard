"""Subtask 1b: DeBERTa token classifier for BIO evidence tagging."""

import torch
import torch.nn as nn
from transformers import AutoModel
from config import MODEL_NAME, NUM_BIO


class EvidenceExtractor(nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, NUM_BIO)

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        return self.head(self.drop(out.last_hidden_state))  # (B, seq, 3)
