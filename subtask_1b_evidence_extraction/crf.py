"""Minimal linear-chain CRF for BIO tagging (3 tags: O/B-EVD/I-EVD).

Implemented from scratch (no `pytorch-crf` dependency available in this
env) -- standard forward-algorithm training + Viterbi decoding, following
Lafferty et al. 2001 / the well-known AllenNLP-style formulation.

Why this might help Sub1b: the current tagger classifies each token
independently (plain per-token cross-entropy), then a hand-written
`bio_to_spans` heuristic stitches raw predictions into spans and merges
small gaps. A CRF instead learns valid tag-transition scores jointly with
the emissions (e.g. I-EVD should almost never directly follow O), which
should produce cleaner span boundaries at the source -- the thing
phrase-F1 is most sensitive to -- rather than relying entirely on
post-hoc heuristic merging.
"""

import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.randn(num_tags) * 0.1)
        self.end_transitions = nn.Parameter(torch.randn(num_tags) * 0.1)
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)

    def forward(self, emissions, tags, mask):
        """Negative log-likelihood per sample (caller should .mean() over batch).
        emissions: (B, L, C) float
        tags:      (B, L) long
        mask:      (B, L) bool, True = real token, False = padding
        """
        gold_score = self._score(emissions, tags, mask)
        log_partition = self._partition(emissions, mask)
        return log_partition - gold_score

    def _score(self, emissions, tags, mask):
        B, L, C = emissions.shape
        maskf = mask.float()
        batch_idx = torch.arange(B, device=emissions.device)
        score = self.start_transitions[tags[:, 0]] + emissions[batch_idx, 0, tags[:, 0]]
        for i in range(1, L):
            emit = emissions[batch_idx, i, tags[:, i]]
            trans = self.transitions[tags[:, i - 1], tags[:, i]]
            score = score + (trans + emit) * maskf[:, i]
        seq_lens = mask.sum(1).long().clamp(min=1) - 1
        last_tags = tags[batch_idx, seq_lens]
        score = score + self.end_transitions[last_tags]
        return score

    def _partition(self, emissions, mask):
        B, L, C = emissions.shape
        maskf = mask.float()
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, C)
        for i in range(1, L):
            emit = emissions[:, i].unsqueeze(1)                 # (B, 1, C)
            trans = self.transitions.unsqueeze(0)                # (1, C, C)
            alpha_t = torch.logsumexp(alpha.unsqueeze(2) + trans + emit, dim=1)  # (B, C)
            m = maskf[:, i].unsqueeze(1)
            alpha = alpha_t * m + alpha * (1 - m)
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    @torch.no_grad()
    def decode(self, emissions, mask):
        """Viterbi decode. Returns a list of B python lists of tag ids
        (each of length = that sample's real sequence length)."""
        B, L, C = emissions.shape
        maskb = mask.bool()
        history = []
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, C)
        for i in range(1, L):
            broadcast_score = score.unsqueeze(2)                 # (B, C, 1)
            emit = emissions[:, i].unsqueeze(1)                  # (B, 1, C)
            next_score = broadcast_score + self.transitions.unsqueeze(0) + emit  # (B,C,C)
            best_score, best_idx = next_score.max(dim=1)          # (B, C)
            m = maskb[:, i].unsqueeze(1)
            score = torch.where(m, best_score, score)
            history.append(best_idx)
        score = score + self.end_transitions.unsqueeze(0)

        seq_lens = maskb.sum(1).long()
        results = []
        for b in range(B):
            n = max(seq_lens[b].item(), 1)
            best_last = int(score[b].argmax().item())
            tags = [best_last]
            for step in reversed(history[:n - 1]):
                best_last = int(step[b, tags[-1]].item())
                tags.append(best_last)
            tags.reverse()
            results.append(tags)
        return results
