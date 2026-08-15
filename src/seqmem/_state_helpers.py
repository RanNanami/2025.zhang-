"""Pure container helpers for the model's existing transient-state boundary.

These functions deliberately know nothing about ``SequentialMemory`` or
diagnostics.  In particular, candidate lists are copied shallowly so candidate
and segment identities remain unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from random import Random
from typing import TypeVar


KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")
ItemT = TypeVar("ItemT")
SnapshotT = TypeVar("SnapshotT")


def shallow_copy_dict(values: dict[KeyT, ValueT]) -> dict[KeyT, ValueT]:
    """Return the same shallow copy produced by ``dict.copy()``."""

    return values.copy()


def shallow_copy_list(values: list[ItemT]) -> list[ItemT]:
    """Return the same shallow copy produced by ``list.copy()``."""

    return values.copy()


def shallow_copy_set(values: set[ItemT]) -> set[ItemT]:
    """Return the same shallow copy produced by ``set.copy()``."""

    return values.copy()


def new_empty_dict() -> dict[object, object]:
    """Return a fresh built-in dict for one reset-state assignment."""

    return {}


def new_empty_list() -> list[object]:
    """Return a fresh built-in list for one reset-state assignment."""

    return []


def new_empty_set() -> set[object]:
    """Return a fresh built-in set for one reset-state assignment."""

    return set()


def shallow_copy_candidate_map(
    values: dict[int, list[ItemT]],
) -> dict[int, list[ItemT]]:
    """Copy the candidate container while retaining every candidate identity."""

    return {
        column: candidates.copy()
        for column, candidates in values.items()
    }


def build_transient_snapshot(
    snapshot_type: Callable[..., SnapshotT],
    *,
    previous_active_cells: dict[int, float],
    previous_winners: dict[int, float],
    last_prediction_candidates: dict[int, list[ItemT]],
    last_prediction_stats: dict[str, int | float],
    last_observe_stats: dict[str, int | float],
    last_symbol_ranking: list[tuple[int, int, str]],
    previous_predicted_sources: set[int],
    previous_burst_only_sources: set[int],
    decode_rng: Random,
    learning_rng: Random,
) -> SnapshotT:
    """Construct the existing snapshot shape in its original field order."""

    return snapshot_type(
        previous_active_cells=shallow_copy_dict(previous_active_cells),
        previous_winners=shallow_copy_dict(previous_winners),
        last_prediction_candidates=shallow_copy_candidate_map(
            last_prediction_candidates
        ),
        last_prediction_stats=shallow_copy_dict(last_prediction_stats),
        last_observe_stats=shallow_copy_dict(last_observe_stats),
        last_symbol_ranking=shallow_copy_list(last_symbol_ranking),
        previous_predicted_sources=shallow_copy_set(
            previous_predicted_sources
        ),
        previous_burst_only_sources=shallow_copy_set(
            previous_burst_only_sources
        ),
        decode_rng_state=decode_rng.getstate(),
        learning_rng_state=learning_rng.getstate(),
    )
