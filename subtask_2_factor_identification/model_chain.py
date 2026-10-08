"""Cheap classifier-chain variant of Sub2's factor classifier.

Real Classifier Chains (Read et al. 2011) train N separate binary models,
each conditioned on the previous models' predictions -- infeasible here at
N=24 with a DeBERTa-v3-large backbone per link (24x the GPU cost of the
current single multi-label head, and can't parallelize across links).

This is the cheap version: ONE shared backbone (same mean-pooled
DeBERTa-v3-large as FactorClassifier), with only the final independent-
sigmoid head replaced by a small GRU that predicts the 24 factors
sequentially, each step conditioned on the pooled post representation AND
the previous step's (teacher-forced during training, self-predicted at
inference) factor value -- a "classifier chain network" in spirit, at
roughly the same cost as the existing single-pass multi-label head.

Motivated by the EDA's factor co-occurrence analysis (factors cluster --
hopelessness is a "hub" connecting most others; e.g. hopelessness+emotion
dysregulation co-occur 307 times, hopelessness+low self-esteem 288 times).
Known risk (see literature check before use): classifier chains are known to
be more sensitive to class imbalance than independent per-label models --
this project's own factors range from 0.5% to 45.6% prevalence, so this
should be validated on true OOF before trusting it, same discipline as
everything else.

Chain order defaults to descending factor prevalence in the training data
(a standard classifier-chain heuristic: put well-supported factors first so
rarer, harder factors get to condition on them, not vice versa) -- computed
from data, not hardcoded, so it stays correct if the data changes.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from config import MODEL_NAME, NUM_FACTORS


def compute_chain_order(label_matrix):
    """label_matrix: (N, NUM_FACTORS) binary. Returns factor indices sorted
    by descending prevalence (most common first)."""
    counts = label_matrix.sum(axis=0)
    return list((-counts).argsort())


class ChainedFactorClassifier(nn.Module):
    def __init__(self, order, model_name=MODEL_NAME, dropout=0.15, chain_hidden=128):
        super().__init__()
        assert sorted(order) == list(range(NUM_FACTORS)), "order must be a permutation of range(NUM_FACTORS)"
        self.order = order
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, chain_hidden), nn.GELU())
        self.rnn_cell = nn.GRUCell(input_size=1, hidden_size=chain_hidden)
        self.out = nn.Linear(chain_hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None, teacher_labels=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        seq = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (seq * mask).sum(1) / mask.sum(1).clamp(min=1)

        h = self.proj(pooled)
        B = pooled.size(0)
        device = pooled.device
        prev_val = torch.zeros(B, 1, device=device)  # chain "start" token

        logits_by_factor = [None] * NUM_FACTORS
        for factor_idx in self.order:
            h = self.rnn_cell(prev_val, h)
            logit = self.out(h)  # (B, 1)
            logits_by_factor[factor_idx] = logit.squeeze(-1)
            if teacher_labels is not None:
                prev_val = teacher_labels[:, factor_idx:factor_idx + 1].float()
            else:
                prev_val = torch.sigmoid(logit).detach()

        return torch.stack(logits_by_factor, dim=1)  # (B, NUM_FACTORS), original factor order
