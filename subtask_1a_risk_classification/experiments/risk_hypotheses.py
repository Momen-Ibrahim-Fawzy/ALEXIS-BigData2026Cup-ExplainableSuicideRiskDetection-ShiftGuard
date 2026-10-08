"""Cumulative-threshold NLI hypotheses for the ordinal reformulation of
Subtask 1a (indicator < ideation < behavior < attempt).

Instead of a 4-way softmax, risk level is decoded from 3 independent binary
"is this post AT LEAST this severe" entailment questions (the CORAL/ordinal-
regression pattern -- Cao et al. 2020, and the Hierarchical Dual-Head paper
already cited in this project's literature review), scored by an NLI
cross-encoder rather than a from-scratch linear head. Rank = number of
thresholds entailed.

Hypothesis phrasing deliberately matches Sub2's "The person mentions/
expresses X" style (see factor_descriptions.py) since the model checkpoint
being warm-started from was fine-tuned specifically on that phrasing
distribution against real posts from this same dataset.
"""

RISK_LEVELS = ["indicator", "ideation", "behavior", "attempt"]
RISK2ORDINAL = {r: i for i, r in enumerate(RISK_LEVELS)}

# (name, min_ordinal_to_be_positive, hypothesis)
THRESHOLDS = [
    ("at_least_ideation",
     1,
     "The person expresses active suicidal thoughts, a wish to die, or intent "
     "to end their life -- not just general risk factors or warning signs."),
    ("at_least_behavior",
     2,
     "The person describes taking a concrete step toward suicide, such as "
     "making a plan, seeking the means, or preparing to act -- not just "
     "thinking about suicide."),
    ("is_attempt",
     3,
     "The person describes having actually attempted suicide."),
]
THRESHOLD_NAMES = [t[0] for t in THRESHOLDS]
NUM_THRESHOLDS = len(THRESHOLDS)


def decode_ordinal(threshold_flags):
    """threshold_flags: iterable of 3 bools/ints (thresholds crossed, in order).
    Rank = count of crossed thresholds, clipped to [0, 3] (CORAL-style decode)."""
    rank = sum(1 for f in threshold_flags if f)
    return max(0, min(rank, len(RISK_LEVELS) - 1))
