"""Lazy imports for optional Fig.9 diagnostic families.

The strict runner imports this module unconditionally, but each diagnostic
module is imported only when its explicit loader is called. Loader functions
return the original module so call sites keep using the historical functions
and constants without a compatibility copy.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType


def _load(module_name: str) -> ModuleType:
    return importlib.import_module(f"experiments.diagnostics.{module_name}")


@lru_cache(maxsize=None)
def load_oracle_candidate() -> ModuleType:
    """Load post-prediction oracle labels and summaries."""

    return _load("fig9_oracle_candidate")


@lru_cache(maxsize=None)
def load_candidate_score_trace() -> ModuleType:
    """Load read-only candidate score row formatting."""

    return _load("fig9_candidate_score_trace")


@lru_cache(maxsize=None)
def load_candidate_context_oracle() -> ModuleType:
    """Load posthoc candidate/context join analysis and output."""

    return _load("fig9_candidate_context_oracle")


@lru_cache(maxsize=None)
def load_branch_provenance() -> ModuleType:
    """Load read-only branch provenance tracking."""

    return _load("fig9_branch_provenance")


@lru_cache(maxsize=None)
def load_preselection_segments() -> ModuleType:
    """Load read-only preselection trace formatting."""

    return _load("fig9_preselection_segments")


@lru_cache(maxsize=None)
def load_teacher_forced_identity() -> ModuleType:
    """Load teacher-forced post-observation diagnostics."""

    return _load("fig9_teacher_forced_identity")


@lru_cache(maxsize=None)
def load_intracolumn_selector() -> ModuleType:
    """Load read-only intracolumn selection trace helpers."""

    return _load("fig9_intracolumn_selector")


@lru_cache(maxsize=None)
def load_readout_dynamics() -> ModuleType:
    """Load the read-only column-level readout trace implementation."""

    return _load("fig9_readout_dynamics")


@lru_cache(maxsize=None)
def load_match_overlap() -> ModuleType:
    """Load read-only match-overlap capture and formatting."""

    return _load("fig9_match_overlap")


@lru_cache(maxsize=None)
def load_segment_context_composition() -> ModuleType:
    """Load segment-context composition tracing and output."""

    return _load("fig9_segment_context_composition")


@lru_cache(maxsize=None)
def load_context_trajectory() -> ModuleType:
    """Load context trajectory decomposition tracking."""

    return _load("fig9_context_trajectory")


@lru_cache(maxsize=None)
def load_temporal_context() -> ModuleType:
    """Load temporal-context read-only tracking."""

    return _load("fig9_temporal_context")


@lru_cache(maxsize=None)
def load_autonomous_context_provenance() -> ModuleType:
    """Load autonomous-context provenance tracking."""

    return _load("fig9_autonomous_context_provenance")


@lru_cache(maxsize=None)
def load_segment_reinforcement() -> ModuleType:
    """Load Scenario reinforcement trace formatting."""

    return _load("fig9_segment_reinforcement")
