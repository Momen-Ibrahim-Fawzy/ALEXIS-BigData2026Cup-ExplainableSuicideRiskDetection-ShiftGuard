"""GCN-gated Sub2 factor classifier -- adapted from ML-GCN (Chen et al.,
"Multi-Label Image Recognition with Graph Convolutional Networks", CVPR 2019,
https://github.com/megvii-research/ML-GCN).

Core ML-GCN idea: instead of 24 independent classifier weight vectors (what
FactorClassifier's plain nn.Linear(hidden, 24) is), a 2-layer GCN propagates
information across a label co-occurrence GRAPH to produce classifier vectors
that are explicitly correlated by which factors tend to co-occur -- e.g.
"hopelessness" (the EDA's co-occurrence hub) directly informs the learned
classifier for "emotion dysregulation", "low self-esteem", etc. This is a
different, more principled mechanism than model_chain.py's classifier-chain
head (which imposes an arbitrary sequential order); the GCN instead uses the
real, symmetric co-occurrence structure the EDA found.

Both the graph (adjacency) and the GCN's INPUT features (each factor's name,
embedded via the backbone's own subword embedding table -- not generic GloVe,
so it's grounded in the same vocabulary as the actual post text) are computed
from real project data, not placeholders.

Adjacency re-weighting follows the ML-GCN paper's scheme exactly: binarize
conditional co-occurrence probability P(j|i) at threshold tau, then re-weight
so each label's neighbors contribute total mass p and the self-loop keeps
(1-p) -- this specifically avoids over-smoothing GCNs are prone to. Note the
paper's defaults (tau=0.4, p=0.2) were tuned on datasets with hundreds/
thousands of examples per label; our conditional-probability estimates come
from only 1,635 posts and are noisier -- tau is exposed as a CLI flag and the
resulting graph's edge count is printed so this can be diagnosed rather than
assumed.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from config import MODEL_NAME, NUM_FACTORS, FACTOR_LIST


def build_adjacency(label_matrix, tau=0.3, p=0.2):
    """label_matrix: (N, NUM_FACTORS) binary. Returns the re-weighted (C, C)
    adjacency exactly as in the ML-GCN paper (Eq. 6-7)."""
    co_occur = label_matrix.T @ label_matrix  # (C, C), diagonal = label counts
    label_counts = np.diag(co_occur).copy()
    cond_prob = co_occur / (label_counts[:, None] + 1e-8)  # P(j | i)
    np.fill_diagonal(cond_prob, 0.0)

    binary_adj = (cond_prob >= tau).astype(np.float32)
    n_edges = int(binary_adj.sum())
    print(f"  Adjacency: tau={tau} -> {n_edges} directed edges "
          f"({n_edges / (NUM_FACTORS * (NUM_FACTORS - 1)):.1%} of possible)")

    row_sums = binary_adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # isolated nodes: avoid div-by-zero, self-loop dominates anyway
    reweighted = binary_adj * (p / row_sums)
    np.fill_diagonal(reweighted, 1.0 - p)
    return reweighted.astype(np.float32)


def build_factor_name_embeddings(model_name=MODEL_NAME):
    """Mean-pooled subword embedding (from the backbone's OWN embedding table,
    not external GloVe) for each of the 24 factor name strings -- a fixed,
    non-trained GCN input feature per factor, grounded in the same vocabulary
    as the post text itself."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    emb_table = backbone.embeddings.word_embeddings.weight.detach()  # (vocab, hidden)

    feats = []
    with torch.no_grad():
        for name in FACTOR_LIST:
            ids = tokenizer(name, return_tensors="pt")["input_ids"][0]
            feats.append(emb_table[ids].mean(dim=0))
    return torch.stack(feats), backbone  # (24, hidden); return backbone so caller can reuse it, not load twice


class GCNFactorClassifier(nn.Module):
    def __init__(self, adjacency, factor_name_embeds, backbone=None, model_name=MODEL_NAME,
                 dropout=0.15, gcn_hidden=512):
        super().__init__()
        self.backbone = backbone if backbone is not None else AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.register_buffer("adjacency", torch.as_tensor(adjacency, dtype=torch.float32))
        self.register_buffer("factor_name_embeds", factor_name_embeds.float())
        self.gc1 = nn.Linear(hidden, gcn_hidden, bias=False)
        self.gc2 = nn.Linear(gcn_hidden, hidden, bias=False)
        self.bias = nn.Parameter(torch.zeros(NUM_FACTORS))

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        seq = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (seq * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = self.drop(pooled)

        h0 = self.factor_name_embeds                                    # (24, hidden)
        h1 = F.leaky_relu(self.adjacency @ self.gc1(h0), 0.2)            # (24, gcn_hidden)
        h2 = F.leaky_relu(self.adjacency @ self.gc2(h1), 0.2)            # (24, hidden) -- per-factor classifier vectors

        logits = pooled @ h2.t() + self.bias                             # (B, 24)
        return logits
