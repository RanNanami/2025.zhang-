"""Pure container helpers for the model's existing transient-state boundary.

These functions deliberately know nothing about ``SequentialMemory`` or
diagnostics.  In particular, candidate lists are copied shallowly so candidate
and segment identities remain unchanged.
"""

from __future__ import annotations

from typing import TypeVar


KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")
ItemT = TypeVar("ItemT")


def shallow_copy_dict(values: dict[KeyT, ValueT]) -> dict[KeyT, ValueT]:
    """Return the same shallow copy produced by ``dict.copy()``."""

    return values.copy()


def shallow_copy_list(values: list[ItemT]) -> list[ItemT]:
    """Return the same shallow copy produced by ``list.copy()``."""

    return values.copy()


def shallow_copy_set(values: set[ItemT]) -> set[ItemT]:
    """Return the same shallow copy produced by ``set.copy()``."""

    return values.copy()


def shallow_copy_candidate_map(
    values: dict[int, list[ItemT]],
) -> dict[int, list[ItemT]]:
    """Copy the candidate container while retaining every candidate identity."""

    return {
        column: candidates.copy()
        for column, candidates in values.items()
    }
