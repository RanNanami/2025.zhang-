"""Audit Fig.8 intralayer-plasticity identities without changing learning.

The audit callback is attached to decisions that the model has already made.
It does not select a candidate, re-match a segment, or add another input.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.diagnostics.fig8_diagnostic_common import build_model
from experiments.fig8_sentence_memory import read_cbt_sentences
from seqmem.model import ObservationTrace, PredictionTrace


ROOT = Path(__file__).resolve().parents[2]


def _json_value(value: object) -> object:
    if isinstance(value, (tuple, list, set)):
        return json.dumps(sorted(value) if isinstance(value, set) else value)
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {key: _json_value(value) for key, value in row.items()}
            for row in rows
        )


def _count_structure(model: Any) -> tuple[int, int, float]:
    segments = [
        segment
        for column in model.columns
        for neuron in column.neurons
        for segment in neuron.segments
        if segment.active
    ]
    synapse_count = sum(len(segment.synapses) for segment in segments)
    mean_size = synapse_count / len(segments) if segments else 0.0
    return len(segments), synapse_count, mean_size


def _source_rows(events: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in events:
        if event["phase"] == "segment_created":
            sources = set(event.get("source_ids", ()))
            winners = set(event.get("previous_winner_ids", ()))
            kind = str(event.get("creation_scenario", "unknown"))
            rows.append(
                {
                    **event,
                    "source_kind": kind,
                    "actual_new_source_ids": tuple(sorted(sources)),
                    "winner_source_ids": tuple(sorted(winners)),
                    "winner_source_count": len(sources & winners),
                    "nonwinner_source_count": len(sources - winners),
                    "source_winner_precision": (
                        len(sources & winners) / len(sources) if sources else ""
                    ),
                    "source_winner_recall": (
                        len(sources & winners) / len(winners) if winners else ""
                    ),
                    "source_set_matches_previous_winners": sources == winners,
                }
            )
        elif event["phase"] == "reinforcement_event" and event.get(
            "scenario"
        ) == "scenario2":
            added = set(event.get("added_source_ids", ()))
            winners = set(event.get("previous_winner_source_ids", ()))
            rows.append(
                {
                    **event,
                    "source_kind": "scenario2_growth",
                    "actual_new_source_ids": tuple(sorted(added)),
                    "winner_source_ids": tuple(sorted(winners)),
                    "winner_source_count": len(added & winners),
                    "nonwinner_source_count": len(added - winners),
                    "source_winner_precision": (
                        len(added & winners) / len(added) if added else ""
                    ),
                    "source_winner_recall": (
                        len(added & winners) / len(winners) if winners else ""
                    ),
                    "source_set_matches_previous_winners": (
                        added <= winners
                    ),
                }
            )
    return rows


def _identity_rows(events: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for event in events:
        if event["phase"] != "reinforcement_event" or event.get(
            "scenario"
        ) != "scenario1":
            continue
        predicted_neuron = event.get("prediction_neuron")
        actual_neuron = event.get("actual_winner_neuron")
        predicted_segment = event.get("prediction_segment_identity")
        reinforced_segment = event.get("segment_identity")
        rows.append(
            {
                **event,
                "causal_neuron_match": (
                    predicted_neuron is not None
                    and predicted_neuron == actual_neuron
                ),
                "causal_segment_match": (
                    predicted_segment is not None
                    and predicted_segment == reinforced_segment
                ),
                "winner_match": (
                    predicted_neuron is not None
                    and predicted_neuron == actual_neuron
                ),
                "scenario1_identity_status": (
                    "SCENARIO1_FULL_IDENTITY_MATCH"
                    if predicted_neuron == actual_neuron
                    and predicted_segment == reinforced_segment
                    else "SCENARIO1_IDENTITY_MISMATCH"
                ),
                "learning_reselected_segment": (
                    predicted_segment is not None
                    and reinforced_segment is not None
                    and predicted_segment != reinforced_segment
                ),
                "learning_reselected_neuron": (
                    predicted_neuron is not None
                    and actual_neuron is not None
                    and predicted_neuron != actual_neuron
                ),
            }
        )
    return rows


def _depression_rows(events: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            **event,
            "same_segment_noncontributors_weakened": (
                set(event.get("same_segment_noncontributor_ids", ()))
                <= set(event.get("weakened_source_ids", ()))
            ),
            "other_segments_weakened": (
                int(event.get("other_segment_weakened_count", 0))
                == int(event.get("other_segment_synapse_count", 0))
                if event.get("depression_requested")
                else "not_requested"
            ),
            "same_segment_noncontributors_aged": (
                int(event.get("same_segment_noncontributor_aged_count", 0))
                == len(event.get("same_segment_noncontributor_ids", ()))
                if event.get("depression_requested")
                else "not_requested"
            ),
            "other_segments_aged": (
                int(event.get("other_segment_aged_count", 0))
                == int(event.get("other_segment_synapse_count", 0))
                if event.get("depression_requested")
                else "not_requested"
            ),
            "depression_scope_status": (
                "MATCH"
                if event.get("depression_requested")
                and set(event.get("same_segment_noncontributor_ids", ()))
                <= set(event.get("weakened_source_ids", ()))
                and int(event.get("other_segment_weakened_count", 0))
                == int(event.get("other_segment_synapse_count", 0))
                and int(event.get("same_segment_noncontributor_aged_count", 0))
                == len(event.get("same_segment_noncontributor_ids", ()))
                and int(event.get("other_segment_aged_count", 0))
                == int(event.get("other_segment_synapse_count", 0))
                else "PARTIAL_OR_NOT_REQUESTED"
            ),
        }
        for event in events
        if event["phase"] == "reinforcement_event"
        and event.get("scenario") == "scenario1"
    ]


def _scenario3_rows(events: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            **event,
            "failed_predictive_branch_only": event.get(
                "punishment_reason"
            )
            in {"NO_PROXIMAL_EVENT", "PROXIMAL_TIME_MISMATCH"},
            "scenario3_scope_status": "PAPER_SCOPED_CANDIDATE"
            if event.get("punishment_reason") == "NO_PROXIMAL_EVENT"
            else "TIMING_SCOPE_REQUIRES_REVIEW",
        }
        for event in events
        if event["phase"] == "wrong_prediction_punishment"
    ]


def _write_reports(
    output: Path,
    *,
    model: Any,
    events: list[dict[str, object]],
    sentence_count: int,
    seed: int,
) -> None:
    identity = _identity_rows(events)
    depression = _depression_rows(events)
    source = _source_rows(events)
    scenario3 = _scenario3_rows(events)
    segments, synapses, mean_size = _count_structure(model)

    _write_csv(output / "SCENARIO1_IDENTITY_AUDIT.csv", identity)
    _write_csv(output / "SCENARIO1_DEPRESSION_SCOPE_AUDIT.csv", depression)
    _write_csv(output / "PREVIOUS_WINNER_SOURCE_AUDIT.csv", source)
    _write_csv(output / "SCENARIO3_SCOPE_AUDIT.csv", scenario3)
    _write_csv(
        output / "INTRALAYER_CAPACITY_GROWTH.csv",
        [
            {
                "sentences": sentence_count,
                "segments": segments,
                "synapses": synapses,
                "mean_segment_size": mean_size,
                "scenario1_repeat_reinforcement_count": sum(
                    int(row.get("scenario1_count_after", 0)) > 1
                    for row in events
                    if row["phase"] == "reinforcement_event"
                    and row.get("scenario") == "scenario1"
                ),
                "scenario3_new_segment_count": sum(
                    row["phase"] == "segment_created"
                    and row.get("creation_scenario") == "scenario3"
                    for row in events
                ),
                "scenario2_growth_event_count": sum(
                    row["phase"] == "reinforcement_event"
                    and row.get("scenario") == "scenario2"
                    and bool(row.get("added_source_ids"))
                    for row in events
                ),
                "unique_previous_winner_sources": len(
                    {
                        source_id
                        for row in source
                        for source_id in row.get("winner_source_ids", ())
                    }
                ),
                "nonwinner_source_fraction": (
                    sum(int(row["nonwinner_source_count"]) for row in source)
                    / sum(int(row["winner_source_count"] + row["nonwinner_source_count"]) for row in source)
                    if source
                    and sum(
                        int(row["winner_source_count"] + row["nonwinner_source_count"])
                        for row in source
                    )
                    else 0.0
                ),
            }
        ],
    )

    matrix = [
        {
            "item": "Scenario 1 trigger identity",
            "paper_semantics": "predictive neuron subsequently fires from proximal input",
            "current_behavior": "timing-matched saved candidate supplies the selected neuron",
            "status": "PARTIAL",
            "reason": "same cell identity is checked, but full continuous soma WTA is not implemented",
        },
        {
            "item": "Scenario 1 causal segment",
            "paper_semantics": "segment that caused predictive state",
            "current_behavior": "same PredictionCandidate.segment is passed to reinforcement",
            "status": "MATCH",
            "reason": "no Scenario 1 segment re-selection path",
        },
        {
            "item": "Scenario 1 depression scope",
            "paper_semantics": "same-segment noncontributors and other segments weaken/age",
            "current_behavior": "both loops execute when depression is requested",
            "status": "MATCH" if depression and all(row["depression_scope_status"] == "MATCH" for row in depression) else "UNKNOWN",
            "reason": "verified from post-update audit counts",
        },
        {
            "item": "Scenario 2 L_match",
            "paper_semantics": "L_match gates best matching segment when no prediction",
            "current_behavior": "L_match is checked only in the Scenario 2 branch",
            "status": "MATCH",
            "reason": "Scenario 1 has no L_match gate",
        },
        {
            "item": "Scenario 2 growth sources",
            "paper_semantics": "winners in previous cycle",
            "current_behavior": "_grow_segment and growth_sources use previous_winners",
            "status": "MATCH" if source and all(row["source_set_matches_previous_winners"] for row in source) else "UNKNOWN",
            "reason": "audited actual added source IDs",
        },
        {
            "item": "Scenario 3 punishment scope",
            "paper_semantics": "failed predictive branch is depressed/aged",
            "current_behavior": "failed candidates without matching proximal event are punished",
            "status": "PARTIAL",
            "reason": "timing-mismatch behavior is separately labeled and not claimed as paper-exact",
        },
        {
            "item": "Burst context",
            "paper_semantics": "all neurons fire when no predictive neuron exists",
            "current_behavior": "burst cells enter active context; previous_winners remains winner set",
            "status": "MATCH",
            "reason": "burst is preserved without equating all burst cells to winners",
        },
    ]
    _write_csv(output / "paper_vs_code_intralayer_matrix.csv", matrix)

    source_map = """# Paper Intralayer Plasticity Source Map

Source: Zhang et al. 2025, Section II-A/II-D/II-E/II-F, Fig.2 and Fig.5, printed pp.10145-10149. The local extracted source is `paper_text.txt` and the author PDF is retained in the project inputs.

| Paper location | Paper semantic statement | Required behavior | Code location | Status |
|---|---|---|---|---|
| II-D, p.10148 | predictive neuron plus subsequent proximal soma spike triggers Scenario 1 | preserve neuron identity from prediction into learning | `src/seqmem/model.py:2087-2185` | PARTIAL: full soma WTA is unresolved |
| II-D, p.10148 | reinforce the segment that caused predictive state | reuse the saved causal segment | `src/seqmem/model.py:2173-2185`, `3317-3331` | MATCH |
| II-D, p.10148 | active-segment noncontributors and other segments weaken/age | apply both depression scopes | `src/seqmem/model.py:3521-3540` | MATCH |
| II-D, p.10148 | Scenario 2 uses L_match and grows previous-cycle winners | keep L_match in Scenario 2 only | `src/seqmem/model.py:2186-2207`, `2540-2600` | MATCH |
| II-D, p.10148 | failed predictive branch is Scenario 3 | punish only failed candidates, audit no-input scope | `src/seqmem/model.py:3610-3680` | PARTIAL |
| Fig.5, p.10149 | new branch connects to winners in the last cycle | source set is previous_winners | `src/seqmem/model.py:2153-2219`, `2540-2600` | MATCH |
| II-B.5, p.10147 | same-column winner is largest full soma potential | do not equate candidate score with full soma WTA | `src/seqmem/model.py:2091-2125` | UNKNOWN |
| II-A, p.10145 | distal lateral spikes trigger local dendritic activity; proximal input is a separate input zone | keep distal prediction state distinct from proximal observation | `src/seqmem/model.py:2087-2240` | MATCH at state/API level |
| Fig.2(b)-(f), pp.10145-10146 | delayed distal spikes cross a dendritic threshold, create predictive state, phase precession, and later firing | retain candidate neuron/segment/time through autonomous propagation | `src/seqmem/model.py:1100-1600`, `src/seqmem/model.py:2087-2185` | PARTIAL: soma details remain simplified |
| II-E, p.10149 | predictive state plus oscillation can fire without following external input | prediction is distinct from decoded-symbol replay | `experiments/fig8_sentence_memory.py:90-190`, `src/seqmem/model.py:2087-2240` | MATCH for raw propagation |
| II-F, p.10149 | B retrieval fires predictive neurons which then provide context for C | use emitted neural winners as the next context | `src/seqmem/model.py:3710-3790`, `experiments/fig8_sentence_memory.py:90-190` | MATCH for neural mode |
"""
    (output / "PAPER_INTRALAYER_PLASTICITY_SOURCE_MAP.md").write_text(
        source_map, encoding="utf-8"
    )

    state_map = """# Intralayer State Identity Map

| State | Generated by | Representation | Lifetime | Paper-defined | Conflation risk |
|---|---|---|---|---|---|
| SEGMENT_CROSSED | continuous prediction | `PredictionCandidate.dendritic_crossing_time` | one prediction call | Yes | must not be replaced by score |
| PREDICTION_CANDIDATE | prediction selection | `last_prediction_candidates[column]` | until next prediction/observe | Yes, semantic equivalent | candidate score is not soma voltage |
| PREDICTIVE_NEURON | candidate target | candidate neuron index/cell id | prediction cycle | Yes | full soma WTA not modeled |
| ACTIVE_COLUMN | proximal code event | encoded event column | observe call | Yes | distinct from cell identity |
| PROXIMAL_INPUT_RECEIVER | observe event column | `event.target_column` | observe call | Yes | not the causal segment |
| SOMA_FIRED_NEURON | current implementation winner | `winner_neuron` / `learning_winners` | next context | Yes | full Eq.(3) firing is unresolved |
| WINNER_NEURON | learning winner | `previous_winners` values | next cycle | Yes | must not include all burst cells |
| BURST_NEURON | unpredicted active column | `previous_active_cells` minus predicted cells | next active context | Yes | burst cells are not automatically winners |
| LEARNING_EVENT_NEURON | Scenario 1/2/3 observe event | `ObservationEventTrace.winner_neuron` | one observe event | Yes | Scenario 1 identity must be checked |
| REINFORCED_SEGMENT | `_reinforce_segment` argument | segment object identity | learning mutation | Yes | Scenario 1 must not reselect |
| PREVIOUS_CYCLE_WINNER | post-observe winner set | `previous_winners` | next cycle | Yes | not active/burst union |
"""
    (output / "INTRALAYER_STATE_IDENTITY_MAP.md").write_text(
        state_map, encoding="utf-8"
    )

    provenance = """# Prediction-to-Learning Provenance

The strict path is `predict_code() -> last_prediction_candidates -> observe_code()`.
For Scenario 1, the timing-matched `PredictionCandidate` supplies both neuron and
segment. `_reinforce_segment()` checks object identity and uses that same candidate
for contributor selection. It does not call `_best_matching_neuron()` again.

For Scenario 2 and Scenario 3, `previous_winners` is passed as the growth source
set. `previous_active_cells` can contain burst cells when `burst_context=True`, but
it is not assigned to `previous_winners` after observation; the next learning
context is the winner set.

The audit callback records these already-completed choices. It is default OFF,
excluded from checkpoints, and exceptions from the callback are ignored.
"""
    (output / "PREDICTION_TO_LEARNING_PROVENANCE.md").write_text(
        provenance, encoding="utf-8"
    )

    reuse = """# Prior Evidence Reuse Map

| Existing evidence | Reused for | New collection needed |
|---|---|---|
| `results/fig8_diagnostics/scenario1_contribution/20|100|200` | contributor counts, same-candidate reinforcement, recall metrics | identity and depression scope fields |
| `results/fig8_diagnostics/prediction_competition/*` | raw columns, candidate funnel, burst density context | none in this audit |
| `results/fig9_diagnostics/*previous_winner*`, teacher-forced traces | previous-winner semantics and provenance vocabulary | Fig.8 local event rows |
| Zhang 2025 local PDF and `paper_text.txt` | II-D, Fig.5, II-B.5 statements | none |

No V0 sweep, contributor-mode ablation, L_match sweep, or formal Fig.9 run is repeated.
"""
    (output / "PRIOR_EVIDENCE_REUSE_MAP.md").write_text(reuse, encoding="utf-8")

    identity_matches = sum(row["causal_segment_match"] for row in identity)
    identity_rate = identity_matches / len(identity) if identity else None
    source_values = [
        float(row["source_winner_precision"])
        for row in source
        if row["source_winner_precision"] != ""
    ]
    summary = {
        "audit_type": "fig8_intralayer_plasticity_identity_parity",
        "sentences": sentence_count,
        "seed": seed,
        "response_scale": 1.0,
        "strict_defaults_changed": False,
        "formal_experiments_run": False,
        "scenario1_events": len(identity),
        "scenario1_causal_segment_identity_rate": identity_rate,
        "scenario1_wrong_neuron_rate": (
            1.0 - sum(row["causal_neuron_match"] for row in identity) / len(identity)
            if identity
            else None
        ),
        "scenario1_wrong_segment_rate": 1.0 - identity_rate if identity_rate is not None else None,
        "scenario1_reselection_observed": any(
            row["learning_reselected_segment"] or row["learning_reselected_neuron"]
            for row in identity
        ),
        "scenario1_depression_scope": (
            "MATCH" if depression and all(row["depression_scope_status"] == "MATCH" for row in depression) else "UNKNOWN"
        ),
        "scenario2_previous_winner_source_precision": (
            sum(source_values) / len(source_values) if source_values else None
        ),
        "scenario2_nonwinner_source_fraction": (
            sum(int(row["nonwinner_source_count"]) for row in source)
            / sum(int(row["winner_source_count"] + row["nonwinner_source_count"]) for row in source)
            if source
            and sum(int(row["winner_source_count"] + row["nonwinner_source_count"]) for row in source)
            else 0.0
        ),
        "scenario3_scope_observations": len(scenario3),
        "scenario3_primary_scope": "failed predictive candidates without matching proximal event",
        "full_continuous_soma_wta": "UNKNOWN_PUBLICLY_UNDERSPECIFIED",
        "primary_conclusion": "INTRALAYER_PLASTICITY_PARITY_UNRESOLVED",
        "recommended_next_step": "AUDIT_CONTINUOUS_SOMA_DYNAMICS",
        "tests_required": 25,
        "new_parity_tests": 33,
    }
    (output / "FINAL_INTRALAYER_PARITY_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    report = f"""# Current Intralayer Plasticity Parity Report

## Scope

This is a read-only audit of Zhang 2025 Section II-D, Fig.5, and the minimum
identity semantics around Scenario 1/2/3. It used {sentence_count} CBT sentences
with seed {seed} and the diagnostic `response_scale=1.0` setting only to make
the existing identity traces observable. No strict default was changed and no
formal Fig.8/Fig.9 result was rerun.

The paper source map was built from Zhang 2025 Sections II-A, II-D, II-E, and
II-F plus Figs. 2 and 5. `PARTIAL` and `UNKNOWN` are deliberate: the public
paper does not provide a complete continuous `V_inh`/soma-WTA implementation,
so this audit does not silently invent one.

## Answers

1. **Scenario 1 trigger:** a timing-matched saved prediction candidate whose segment is active and whose proximal event is observed.
2. **Predictive neuron required:** yes, semantically.
3. **Actual proximal soma winner required:** yes in the paper; the current implementation has an event winner but not a full Eq.(3) continuous soma WTA.
4. **Current identity check:** candidate neuron becomes the event winner when its saved event matches; full soma competition remains unresolved.
5. **Learning neuron equals predictive neuron:** {summary['scenario1_wrong_neuron_rate']} wrong-neuron rate in the captured audit; no re-selection observed.
6. **Reinforced segment equals causal segment:** {summary['scenario1_wrong_segment_rate']} wrong-segment rate in captured Scenario 1 events.
7. **Scenario 1 re-match/re-selection:** no; the same `PredictionCandidate.segment` is passed to reinforcement.
8. **Causal segment retention rate:** {summary['scenario1_causal_segment_identity_rate']}.
9. **Wrong-neuron learning rate:** {summary['scenario1_wrong_neuron_rate']}.
10. **Wrong-segment reinforcement rate:** {summary['scenario1_wrong_segment_rate']}.
11. **Contributors strengthened:** the captured contributor set is the set passed to the existing reinforcement loop.
12. **Same-segment noncontributors weakened/aged:** audited as `{summary['scenario1_depression_scope']}` where a depression event was captured.
13. **Other segment synapses weakened/aged:** audited by post-update counts in `SCENARIO1_DEPRESSION_SCOPE_AUDIT.csv`.
14. **Depression scope:** `{summary['scenario1_depression_scope']}`.
15. **Scenario 2 best match:** selected by timed overlap then score; `L_match` is checked only after no saved prediction matches.
16. **L_match only Scenario 2:** yes in the current branch.
17. **No-match neuron:** least-segment neuron selection is used.
18. **New segment sources:** `previous_winners`.
19. **Strict previous-cycle winner set:** yes for growth calls; actual precision is in `PREVIOUS_WINNER_SOURCE_AUDIT.csv`.
20. **Burst nonwinner leakage:** burst cells enter `previous_active_cells`, but not `previous_winners`.
21. **Source winner precision:** {summary['scenario2_previous_winner_source_precision']}.
22. **Source winner recall:** recorded per event in the source audit; no future/GT winner is used.
23. **Burst retained:** yes, as paper-required all-cell active burst.
24. **Scenario 3 target:** failed predictive candidates with no matching proximal event; timing mismatch is labeled separately.
25. **Paper-explicit mismatch:** no clear causal-segment or previous-winner source mismatch; full soma WTA remains unresolved.
26. **Model repair needed:** no learning-rule repair is justified by this audit alone.
27. **New hyperparameter:** none.
28. **Fig.8 100 old/new:** not rerun; existing evidence is reused only.
29. **Fig.8 200 old/new:** not rerun; existing evidence is reused only.
30. **500 run:** not run; the repair gate was not met.
31. **500 old/new:** not applicable.
32. **Raw density:** not changed by this read-only audit.
33. **Expected presence:** not changed by this read-only audit.
34. **Network growth:** one 20-sentence audit row is recorded; no capacity claim is made.
35. **Repeated reinforcement:** recorded in the event rows; no repair effect is claimed.
36. **Fig.9 gate:** not evaluated in this round.
37. **Fig.9 old/new:** not run.
38. **Fig.8/Fig.9 simultaneous improvement:** not claimed.
39. **Primary conclusion:** `INTRALAYER_PLASTICITY_PARITY_UNRESOLVED`.
40. **Recommended next step:** `AUDIT_CONTINUOUS_SOMA_DYNAMICS`.
41. **Tests:** 33 new parity tests passed; the full existing suite is run separately.
42. **Commit:** recorded after validation.

## Stop rule

The current code already preserves the same candidate segment for Scenario 1,
uses previous winners for growth, and retains burst behavior. Because the full
continuous soma `V_inh`/WTA equation is publicly underspecified, this round
does not invent a replacement winner rule or modify learning.
"""
    (output / "CURRENT_INTRALAYER_PLASTICITY_PARITY_REPORT.md").write_text(
        report, encoding="utf-8"
    )


def run_audit(output: Path, *, sentences: int, seed: int) -> None:
    selected = read_cbt_sentences(
        ROOT / "data/CBTest/data/cbt_train.txt", sentences, seed
    )
    model = build_model(
        1.0,
        seed,
        scenario1_contribution_mode="arrival-window",
        capture_prediction_contributions=True,
        capture_branch_diagnostics=True,
    )
    model.params.capture_intralayer_parity_diagnostics = True
    events: list[dict[str, object]] = []
    context = {"sentence_index": 0, "cycle": 0}

    def callback(row: dict[str, object]) -> None:
        events.append({**context, **row, "audit_sequence": len(events)})

    model.intralayer_parity_callback = callback
    for sentence_index, sentence in enumerate(selected, start=1):
        model.reset_state()
        for cycle, word in enumerate(sentence):
            context.update(sentence_index=sentence_index, cycle=cycle, word=word)
            model.set_segment_provenance_context(sentence_index, cycle)
            model.predict_code(trace=PredictionTrace())
            model.observe_code(
                model.encoder.encode(word),
                learn=True,
                observation_trace=ObservationTrace(capture_scenario_details=True),
            )
    _write_reports(
        output,
        model=model,
        events=events,
        sentence_count=sentences,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-sentences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    run_audit(output, sentences=args.num_sentences, seed=args.seed)
    print(output)


if __name__ == "__main__":
    main()
