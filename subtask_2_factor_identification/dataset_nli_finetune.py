"""Explodes posts into (post, factor-description) pairs for the NLI
cross-encoder fine-tune. 1,635 posts x 24 factors = 39,240 pairs total --
but these are NOT independent samples (24 pairs from the same post share
the same premise), so splitting MUST happen at the post/user level before
exploding to pairs, never at the pair level, or validation leaks the other
23 pairs from the same post straight into training.
"""

import torch
from torch.utils.data import Dataset

from config import NUM_FACTORS, FACTOR_LIST, FACTOR2ID
from factor_descriptions import FACTOR_DESCRIPTIONS_BY_NAME

CANDIDATE_DESCRIPTIONS = [FACTOR_DESCRIPTIONS_BY_NAME[f] for f in FACTOR_LIST]


def build_pairs(df):
    """df: has 'post' and 'factors_list' columns, ORIGINAL (post-level) rows.
    Returns a flat list of (post_text, factor_description, label, post_row_idx, factor_idx)."""
    pairs = []
    for row_idx, row in df.iterrows():
        present = {f.lower().strip() for f in row["factors_list"]}
        for factor_idx, factor_name in enumerate(FACTOR_LIST):
            label = 1.0 if factor_name in present else 0.0
            pairs.append((str(row["post"]), CANDIDATE_DESCRIPTIONS[factor_idx], label, row_idx, factor_idx))
    return pairs


class NLIPairDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        post, desc, label, row_idx, factor_idx = self.pairs[idx]
        enc = self.tokenizer(post, desc, max_length=self.max_length, padding="max_length",
                             truncation=True, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float32)
        item["row_idx"] = row_idx
        item["factor_idx"] = factor_idx
        return item
