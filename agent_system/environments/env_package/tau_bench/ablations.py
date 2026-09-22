"""Tau ablations: learning rewards and environment advancement are separate."""

VARIANTS = {"full", "a1", "a4", "a5"}


def validate_ablation(variant, *, reward_mode, programmatic_only=False):
    if variant not in VARIANTS:
        raise ValueError(f"unknown Tau ablation: {variant}")
    if variant != "full" and programmatic_only:
        raise ValueError("Tau ablations cannot also mask matcher-required groups")
    if variant == "a4" and reward_mode != "appearance":
        raise ValueError("A4 requires appearance reward with all three teacher votes")
    if variant in {"a1", "a5"} and reward_mode != "frequency_weighted":
        raise ValueError(f"{variant} requires the Full frequency-weighted commit scores")


def learning_rewards(variant, semantic_scores, candidates, transfer_violations):
    """Do not alter semantic scores, teacher hit diagnostics, or group masks."""
    if variant != "a1":
        return list(semantic_scores)
    return [-1.0 if action.kind == "invalid" or violation else 1.0 for action, violation in zip(candidates, transfer_violations, strict=True)]
