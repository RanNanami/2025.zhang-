"""Pure scalar calculations for the existing forgetting rule."""

from __future__ import annotations


def forgetting_score(
    weight: float,
    age: int,
    l_weight: float,
    l_age: float,
) -> float:
    """Return the paper-aligned score with original parenthesization."""

    return l_weight * (1.0 - weight) + l_age * age


def synapse_is_retained(
    weight: float,
    age: int,
    l_weight: float,
    l_age: float,
    threshold: float,
) -> bool:
    """Apply the existing strict retention threshold to one synapse."""

    return forgetting_score(weight, age, l_weight, l_age) < threshold
