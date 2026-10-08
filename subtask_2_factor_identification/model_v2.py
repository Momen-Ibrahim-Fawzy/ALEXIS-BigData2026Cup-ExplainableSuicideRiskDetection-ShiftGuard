"""Sub2 v2: combines four components into one architecture --

1. Special risk-level tokens ([RISK_INDICATOR]/[RISK_IDEATION]/[RISK_BEHAVIOR]/
   [RISK_ATTEMPT]) prepended to the input text, so Sub1a's risk classification
   flows through DeBERTa's self-attention across all layers -- mechanistically
   different from the earlier cross-task-context experiment (v15/BACKLOG),
   which concatenated Sub1a's softmax probabilities onto the pooled
   representation only at the very end. That post-hoc version scored 0.4787
   standalone, well below baseline -- worth a genuinely different mechanism,
   though the underlying SIGNAL (risk level -> factor identity) was already
   shown weak in isolation, so temper expectations.
2. SBert semantic-similarity features (see sbert_features.py) concatenated
   before the classification head -- a FIXED, pretrained signal (not learned
   from our 1,635 examples), unlike the from-scratch dual-encoder (v14) that
   failed badly (OOF 0.3801) trying to learn its own embedding space from too
   little data.
3. Distribution-Balanced Loss (losses_v2.py) as the main classification loss.
4. A weighted multi-label contrastive auxiliary loss on the pooled DeBERTa
   representation (losses_v2.py), applied only to the DeBERTa-derived part
   of the representation, not the fixed SBert features.

New token embeddings are initialized to the MEAN of the existing embedding
table (not random) -- avoids destabilizing early training, the same lesson
learned from the DAPT incident earlier this project (a freshly-initialized
head produced NaN under BF16 + aggressive LR).
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from config import MODEL_NAME, NUM_FACTORS

RISK_TOKENS = ["[RISK_INDICATOR]", "[RISK_IDEATION]", "[RISK_BEHAVIOR]", "[RISK_ATTEMPT]"]
RISK_TOKEN_MAP = {"indicator": "[RISK_INDICATOR]", "ideation": "[RISK_IDEATION]",
                   "behavior": "[RISK_BEHAVIOR]", "attempt": "[RISK_ATTEMPT]"}

SBERT_DIM = NUM_FACTORS  # one similarity score per factor


def add_risk_tokens(tokenizer):
    """Call once; returns how many tokens were added (0 if already present)."""
    return tokenizer.add_special_tokens({"additional_special_tokens": RISK_TOKENS})


def init_new_token_embeddings(model, tokenizer, n_new_tokens):
    """Mean-init instead of the HF default (random from a small normal) --
    see module docstring for why."""
    if n_new_tokens == 0:
        return
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        emb = model.embeddings.word_embeddings.weight
        mean_emb = emb[:-n_new_tokens].mean(dim=0)
        emb[-n_new_tokens:] = mean_emb.unsqueeze(0).expand(n_new_tokens, -1)


class FactorClassifierV2(nn.Module):
    def __init__(self, model_name=MODEL_NAME, tokenizer=None, dropout=0.15, sbert_dim=SBERT_DIM):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        if tokenizer is not None:
            n_new = add_risk_tokens(tokenizer)
            init_new_token_embeddings(self.backbone, tokenizer, n_new)

        hidden = self.backbone.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden + sbert_dim, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, NUM_FACTORS),
        )

    def forward(self, input_ids, attention_mask, sbert_sims, token_type_ids=None, **kwargs):
        """sbert_sims: (B, NUM_FACTORS) fixed similarity features.
        Returns (logits, pooled) -- pooled is the DeBERTa-only representation,
        used by the contrastive auxiliary loss (see train_v2.py)."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        seq = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (seq * mask).sum(1) / mask.sum(1).clamp(min=1)

        combined = torch.cat([self.drop(pooled), sbert_sims], dim=1)
        logits = self.head(combined)
        return logits, pooled
