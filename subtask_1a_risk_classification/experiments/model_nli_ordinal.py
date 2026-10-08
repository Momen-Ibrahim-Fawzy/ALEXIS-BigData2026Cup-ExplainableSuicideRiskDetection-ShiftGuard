"""New Subtask 1a lever: reformulate 4-class ordinal risk classification as
NLI cumulative-threshold entailment -- the same underlying mechanism (cross-
encoder transfer from an NLI-pretrained checkpoint, LP-FT) that produced
Subtask 2's biggest gains, applied to 1a for the first time. Combines three
previously-separate ideas into one model:

1. Ordinal-as-cumulative-NLI: 3 binary "at least this severe" hypotheses
   (risk_hypotheses.py), scored by an NLI cross-encoder instead of a
   from-scratch linear head -- CORAL's ordinal-threshold idea (already tried
   here alone, exp25, and came back noise-level) combined with NLI transfer
   (already tried here alone, Sub2, and was this project's biggest win) in a
   combination neither was tried in before.
2. Warm-start from Sub2's own domain-fine-tuned NLI checkpoint (v22 recipe,
   subtask_2_factor_identification/outputs_nli_finetune_kfold_moreepochs)
   instead of the generic off-the-shelf MNLI checkpoint -- already adapted to
   this exact post style, at zero extra pretraining cost.
3. Auxiliary input: Sub2's per-factor entailment probabilities (v24 recipe,
   outputs_nli_moreepochs_seedbag_combined) concatenated before the final
   classifier. Several factors are near-diagnostic for risk level per the
   EDA (e.g. "prior self-harm or suicidal thought/attempt" appears in 88.6%
   of `attempt` posts) -- a similar cross-task-context idea failed once
   before (v15) but using much weaker signal, before the Sub2 NLI series
   existed.

Architecture mirrors DebertaV2ForSequenceClassification's own submodules
(.deberta, .pooler, .dropout) exactly, so a Sub2 NLIFactorClassifier
checkpoint's backbone+pooler weights load directly -- only the final
classifier is new (different shape: hidden + aux_dim -> 1, vs the original
hidden -> 3).
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
NUM_FACTORS = 24  # must match subtask_2_factor_identification/config.py FACTOR_LIST


class NLIOrdinalRiskClassifier(nn.Module):
    def __init__(self, model_name=NLI_MODEL, aux_dim=NUM_FACTORS):
        super().__init__()
        base = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.deberta = base.deberta
        self.pooler = base.pooler
        self.dropout = base.dropout
        hidden = base.config.hidden_size
        self.aux_dim = aux_dim
        self.classifier = nn.Linear(hidden + aux_dim, 1)
        del base

    def forward(self, input_ids, attention_mask, aux_features, token_type_ids=None):
        enc = self.deberta(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled = self.pooler(enc.last_hidden_state)
        pooled = self.dropout(pooled)
        feat = torch.cat([pooled, aux_features], dim=-1)
        return self.classifier(feat).squeeze(-1)

    def load_sub2_warm_start(self, ckpt_path, device):
        """Load a Sub2 NLIFactorClassifier checkpoint (state dict keyed
        'model.deberta.*' / 'model.pooler.*' / 'model.classifier.*') into
        this model's deberta+pooler only. The classifier is a different
        shape here (aux features concatenated, single-threshold output) and
        is intentionally left freshly initialized. Returns the number of
        tensors actually copied, so the caller can sanity-check it's nonzero."""
        raw = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        own = self.state_dict()
        copied = 0
        for k, v in sd.items():
            if not k.startswith("model."):
                continue
            k2 = k[len("model."):]
            if (k2.startswith("deberta.") or k2.startswith("pooler.")) \
                    and k2 in own and own[k2].shape == v.shape:
                own[k2] = v
                copied += 1
        self.load_state_dict(own)
        return copied

    def freeze_backbone(self):
        for p in self.deberta.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.deberta.parameters():
            p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
