"""Explodes posts into (post, threshold-hypothesis) pairs for the ordinal
cumulative-NLI reformulation of Subtask 1a. 1,635 posts x 3 thresholds =
4,905 pairs -- small enough for a direct full 5-fold run (unlike Sub2's
24-factor explosion, which needed a single-split pilot first).

Each pair also carries the GLOBAL row index into the full train/leaderboard
dataframe (not the fold-local index used for OOF matrix bookkeeping), so the
Sub2 auxiliary factor-probability array (aligned to that global order) can be
looked up correctly regardless of which fold-subset a row currently sits in.
"""

import torch
from torch.utils.data import Dataset

from risk_hypotheses import THRESHOLDS, RISK2ORDINAL

NUM_THRESHOLDS = len(THRESHOLDS)


def build_pairs(df, global_idx_col=None, risk_col="risk_norm"):
    """df: has 'post' and (for train/val) risk_col columns.
    global_idx_col: name of the column holding the row's position in the
    ORIGINAL full dataframe (matches aux_features array rows). If None,
    the df's own positional order is used as-is (correct for a leaderboard
    df that isn't a fold-subset of anything).
    Returns (post_text, hypothesis, label, local_row_idx, global_row_idx, thresh_idx).
    label is None when risk_col is not present (leaderboard / inference)."""
    has_labels = risk_col in df.columns
    pairs = []
    for local_idx, (_, row) in enumerate(df.iterrows()):
        global_idx = int(row[global_idx_col]) if global_idx_col else local_idx
        ordinal = RISK2ORDINAL[row[risk_col]] if has_labels else None
        for t_idx, (name, min_ordinal, hypothesis) in enumerate(THRESHOLDS):
            label = (1.0 if ordinal >= min_ordinal else 0.0) if has_labels else 0.0
            pairs.append((str(row["post"]), hypothesis, label, local_idx, global_idx, t_idx))
    return pairs


class NLIOrdinalDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length, aux_features):
        """aux_features: (n_global_rows, aux_dim) float array -- looked up by
        each pair's global_row_idx, NOT its position in this dataset."""
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.aux_features = aux_features

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        post, hyp, label, local_idx, global_idx, t_idx = self.pairs[idx]
        enc = self.tokenizer(post, hyp, max_length=self.max_length, padding="max_length",
                             truncation=True, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float32)
        item["row_idx"] = local_idx
        item["thresh_idx"] = t_idx
        item["aux_features"] = torch.tensor(self.aux_features[global_idx], dtype=torch.float32)
        return item


def pairs_to_matrix(scores, row_idx, thresh_idx, n_rows, n_thresh=NUM_THRESHOLDS):
    import numpy as np
    mat = np.zeros((n_rows, n_thresh), dtype=np.float32)
    mat[row_idx, thresh_idx] = scores
    return mat
