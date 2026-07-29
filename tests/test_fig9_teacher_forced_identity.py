from __future__ import annotations

import csv
import hashlib
import inspect
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from experiments.diagnostics.analyze_fig9_teacher_forced_identity import (
    CLASSIFICATIONS,
    align_identity_rows,
    analyze,
    bootstrap_rows,
    classify_identity,
)
from experiments.diagnostics.fig9_competitive_inhibition import (
    CompetitionSettings,
)
from experiments.diagnostics.fig9_teacher_forced_identity import (
    DIAGNOSTIC_MARKERS,
    build_teacher_forced_observation_rows,
    stable_observation_reference_id,
    winner_set,
)
from experiments.fig9 import TaxiRecord
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    load_strict_checkpoint,
    run_strict_stream,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_csv(path: Path) -> list[dict[str, str]]:
    runtime_fields = {
        "prediction_runtime_seconds",
        "decode_runtime_seconds",
        "competition_runtime_seconds",
        "observe_runtime",
        "runtime_seconds",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                key: value
                for key, value in row.items()
                if key not in runtime_fields
            }
            for row in csv.DictReader(handle)
        ]


def _reference(
    *,
    neuron: str = "2",
    scenario: str = "scenario1",
    segment_id: str = "teacher-segment",
) -> dict[str, object]:
    return {
        "actual_record_index": 5,
        "actual_timestamp": "2014-07-01 02:30:00",
        "field": "passenger",
        "target_column": 100,
        "observed_winner_available": True,
        "observed_winner_neurons": neuron,
        "selected_segment_id": segment_id,
        "created_segment_id": (
            segment_id if scenario == "scenario3" else ""
        ),
        "observe_scenario": scenario,
        "scenario3": scenario == "scenario3",
        "creation_source_fingerprint": "teacher-source",
        "creation_source_cell_ids": "1 2",
        "current_source_fingerprint": "teacher-current",
        "segment_creation_transition_index": 4,
        "source_support_class": "predicted-supported",
    }


def _funnel(outcome: str = "TARGET_SEGMENT_EMITTED") -> dict[str, object]:
    return {
        "input_index": 5,
        "target_timestamp": "2014-07-01 02:30:00",
        "horizon_step": 1,
        "target_column_outcomes": json.dumps(
            [{"field": "passenger", "column": 100, "outcome": outcome}]
        ),
    }


def _segment(
    *,
    neuron: int = 2,
    candidate: bool = True,
    emitted: bool = True,
    segment_id: str = "teacher-segment",
) -> dict[str, object]:
    return {
        "input_index": 5,
        "target_timestamp": "2014-07-01 02:30:00",
        "horizon_step": 1,
        "target_column": 100,
        "target_neuron": neuron,
        "crossed_threshold": True,
        "became_prediction_candidate": candidate,
        "emitted_after_competition": emitted,
        "segment_provenance_id": segment_id,
        "creation_source_fingerprint": "teacher-source",
        "current_source_fingerprint": "teacher-current",
        "creation_transition_index": 4,
        "source_support_class": "predicted-supported",
    }


class TeacherForcedIdentityUnitTests(unittest.TestCase):
    def test_reference_id_is_stable_and_uses_column(self) -> None:
        arguments = {
            "stream_label": "original",
            "actual_record_index": 8,
            "actual_timestamp": "2014-07-01 04:00:00",
            "target_column": 10,
        }
        reference_id = stable_observation_reference_id(**arguments)
        self.assertEqual(
            reference_id,
            stable_observation_reference_id(**arguments),
        )
        arguments["target_column"] = 11
        self.assertNotEqual(
            reference_id,
            stable_observation_reference_id(**arguments),
        )

    def test_winner_set_supports_single_multiple_and_missing(self) -> None:
        self.assertEqual(winner_set({"observed_winner_neurons": "3"}), {3})
        self.assertEqual(
            winner_set({"observed_winner_neurons": "3 7"}),
            {3, 7},
        )
        self.assertEqual(winner_set({}), set())

    def test_classification_covers_absent_below_and_no_firing(self) -> None:
        shared = {
            "reference_available": True,
            "target_column_candidate": False,
            "target_column_emitted": False,
            "reference_neuron_crossed": False,
            "reference_neuron_candidate": False,
            "reference_neuron_emitted": False,
        }
        self.assertEqual(
            classify_identity(
                funnel_outcome="TARGET_NO_SEGMENT_INSPECTED",
                **shared,
            ),
            CLASSIFICATIONS["A"],
        )
        self.assertEqual(
            classify_identity(
                funnel_outcome="TARGET_SEGMENT_BELOW_THRESHOLD",
                **shared,
            ),
            CLASSIFICATIONS["B"],
        )
        self.assertEqual(
            classify_identity(
                funnel_outcome=(
                    "TARGET_SEGMENT_CROSSED_BUT_NO_VALID_FIRING_TIME"
                ),
                **shared,
            ),
            CLASSIFICATIONS["C"],
        )

    def test_missing_reference_is_safe(self) -> None:
        rows = align_identity_rows(
            observation_rows=[],
            funnel_rows=[_funnel()],
            segment_rows=[_segment()],
            policy_label="batched",
        )
        self.assertEqual(rows[0]["classification"], CLASSIFICATIONS["I"])

    def test_multiple_winner_match_accepts_any_member(self) -> None:
        rows = align_identity_rows(
            observation_rows=[_reference(neuron="2 7")],
            funnel_rows=[_funnel()],
            segment_rows=[_segment(neuron=7)],
            policy_label="batched",
        )
        self.assertEqual(rows[0]["classification"], CLASSIFICATIONS["H"])
        self.assertEqual(rows[0]["winner_set_size"], 2)

    def test_correct_column_wrong_neuron_classification(self) -> None:
        rows = align_identity_rows(
            observation_rows=[_reference(neuron="2")],
            funnel_rows=[_funnel()],
            segment_rows=[_segment(neuron=7)],
            policy_label="batched",
        )
        self.assertEqual(rows[0]["classification"], CLASSIFICATIONS["G"])
        self.assertTrue(rows[0]["correct_column_wrong_neuron"])

    def test_reference_candidate_suppressed_classification(self) -> None:
        rows = align_identity_rows(
            observation_rows=[_reference()],
            funnel_rows=[
                _funnel(
                    "TARGET_SEGMENT_BECAME_CANDIDATE_BUT_SUPPRESSED_INTERCOLUMN"
                )
            ],
            segment_rows=[_segment(candidate=True, emitted=False)],
            policy_label="batched",
        )
        self.assertEqual(rows[0]["classification"], CLASSIFICATIONS["F"])

    def test_reference_present_but_not_candidate_classification(self) -> None:
        rows = align_identity_rows(
            observation_rows=[_reference()],
            funnel_rows=[
                _funnel("TARGET_SEGMENT_CROSSED_BUT_LOST_INTERNAL_SELECTION")
            ],
            segment_rows=[_segment(candidate=False, emitted=False)],
            policy_label="batched",
        )
        self.assertEqual(rows[0]["classification"], CLASSIFICATIONS["E"])

    def test_exact_segment_and_source_similarity(self) -> None:
        rows = align_identity_rows(
            observation_rows=[_reference()],
            funnel_rows=[_funnel()],
            segment_rows=[_segment()],
            policy_label="batched",
            creation_sources_by_segment={"teacher-segment": {1, 2}},
        )
        self.assertTrue(rows[0]["exact_segment_match"])
        self.assertEqual(
            rows[0]["creation_source_fingerprint_jaccard"],
            1.0,
        )
        self.assertTrue(rows[0]["same_creation_transition"])

    def test_created_after_observation_has_no_exact_match_requirement(self) -> None:
        rows = align_identity_rows(
            observation_rows=[_reference(scenario="scenario3")],
            funnel_rows=[_funnel()],
            segment_rows=[_segment(segment_id="older-segment")],
            policy_label="batched",
        )
        self.assertTrue(
            rows[0][
                "teacher_forced_segment_created_only_after_observation"
            ]
        )
        self.assertFalse(rows[0]["exact_segment_match_applicable"])
        self.assertEqual(rows[0]["exact_segment_match"], "")

    def test_timestamp_join_and_target_index_validation(self) -> None:
        rows = align_identity_rows(
            observation_rows=[_reference()],
            funnel_rows=[_funnel()],
            segment_rows=[_segment()],
            policy_label="batched",
        )
        self.assertTrue(rows[0]["target_index_mapping_valid"])
        changed = _reference()
        changed["actual_timestamp"] = "2014-07-01 03:00:00"
        missing = align_identity_rows(
            observation_rows=[changed],
            funnel_rows=[_funnel()],
            segment_rows=[_segment()],
            policy_label="batched",
        )
        self.assertEqual(missing[0]["classification"], CLASSIFICATIONS["I"])

    def test_same_reference_is_reused_by_multiple_rollouts(self) -> None:
        first = _funnel()
        second = _funnel()
        second["input_index"] = 4
        second["horizon_step"] = 2
        second_segment = _segment()
        second_segment["input_index"] = 4
        second_segment["horizon_step"] = 2
        rows = align_identity_rows(
            observation_rows=[_reference()],
            funnel_rows=[first, second],
            segment_rows=[_segment(), second_segment],
            policy_label="batched",
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["reference_available"] for row in rows))

    def test_rollout_bootstrap_is_repeatable(self) -> None:
        rows = [
            {
                "input_index": index,
                "reference_available": True,
                "target_column_emitted_recall": index % 2 == 0,
                "reference_neuron_candidate_recall": True,
                "reference_neuron_emitted_recall": index % 2 == 0,
                "correct_column_wrong_neuron": index % 2 != 0,
            }
            for index in range(8)
        ]
        self.assertEqual(
            bootstrap_rows(rows, samples=20, seed=3),
            bootstrap_rows(rows, samples=20, seed=3),
        )

    def test_analysis_empty_trace_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = analyze(
                run_dir=root / "missing",
                output_dir=root / "analysis",
                policy_label="batched",
                bootstrap_samples=2,
                bootstrap_seed=0,
            )
            self.assertEqual(summary["target_columns"], 0)
            self.assertTrue(
                (root / "analysis" / "teacher_forced_identity_summary.json").exists()
            )

    def test_analysis_and_capture_do_not_call_prediction_or_observe(self) -> None:
        sources = (
            inspect.getsource(align_identity_rows),
            inspect.getsource(build_teacher_forced_observation_rows),
        )
        for source in sources:
            for forbidden in (
                "predict_code(",
                ".observe(",
                ".observe_code(",
                "learn_actual_code(",
            ):
                self.assertNotIn(forbidden, source)

    def test_all_required_safety_markers_are_true(self) -> None:
        self.assertTrue(all(DIAGNOSTIC_MARKERS.values()))


class TeacherForcedIdentityIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.data_path = cls.root / "tiny.csv"
        cls.data_path.write_text(
            "timestamp,passenger_count\n",
            encoding="utf-8",
        )
        start = datetime(2014, 7, 1)
        cls.records = [
            TaxiRecord(
                start + timedelta(minutes=30 * index),
                1000.0 + index * 20.0,
            )
            for index in range(18)
        ]
        settings = CompetitionSettings(
            mode="competitive_raw",
            inhibition_strength=0.1,
            inhibition_tau=0.02,
            simultaneous_policy="batched",
            simultaneous_bin_width=0.005,
        )
        cls.results = {}
        for label, enabled in (("off", False), ("on", True)):
            directory = cls.root / label
            cls.results[label] = run_strict_stream(
                records=cls.records,
                data_path=cls.data_path,
                stream_label="original",
                output_dir=directory,
                config=Fig9StrictConfig(warmup=6),
                limit=0,
                print_fingerprint=False,
                competition_settings=settings,
                oracle_candidate_diagnostic=True,
                branch_provenance_diagnostic=True,
                branch_provenance_level="candidate",
                preselection_segment_diagnostic=True,
                preselection_segment_level="crossing",
                checkpoint_path=directory / "checkpoint.pkl",
                checkpoint_at_index=10,
                teacher_forced_winner_diagnostic=enabled,
                teacher_forced_winner_level="segment",
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_on_off_preserves_predictions_model_and_rng(self) -> None:
        off = self.root / "off"
        on = self.root / "on"
        self.assertEqual(
            _sha256(off / "original_predictions.csv"),
            _sha256(on / "original_predictions.csv"),
        )
        self.assertEqual(
            self.results["off"]["final_model_fingerprint"],
            self.results["on"]["final_model_fingerprint"],
        )
        self.assertEqual(
            self.results["off"]["final_rng_fingerprint"],
            self.results["on"]["final_rng_fingerprint"],
        )

    def test_on_off_preserves_existing_diagnostic_traces(self) -> None:
        off = self.root / "off"
        on = self.root / "on"
        self.assertEqual(
            _stable_csv(off / "competition_trace.csv"),
            _stable_csv(on / "competition_trace.csv"),
        )
        for name in (
            "oracle_candidate_trace.csv",
            "branch_candidate_trace.csv",
            "branch_segment_trace.csv",
            "preselection_segment_trace.csv",
            "preselection_group_trace.csv",
            "preselection_replacement_trace.csv",
        ):
            self.assertEqual(
                (off / name).read_bytes(),
                (on / name).read_bytes(),
                name,
            )
        self.assertEqual(
            _stable_csv(off / "preselection_funnel_trace.csv"),
            _stable_csv(on / "preselection_funnel_trace.csv"),
        )

    def test_reference_rows_have_one_winner_and_real_scenarios(self) -> None:
        path = self.root / "on" / "teacher_forced_observation_trace.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        self.assertTrue(
            all(int(row["observed_winner_count_in_column"]) == 1 for row in rows)
        )
        self.assertTrue(
            {row["observe_scenario"] for row in rows}
            <= {"scenario1", "scenario2", "scenario3"}
        )
        self.assertTrue(
            all(row["future_observation_does_not_affect_past_prediction"] == "True"
                for row in rows)
        )

    def test_scenario_segment_capture_and_created_semantics(self) -> None:
        path = self.root / "on" / "teacher_forced_observation_trace.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row["observe_scenario"] in {"scenario1", "scenario2"}:
                self.assertEqual(
                    row["selected_segment_id"],
                    row["reinforced_segment_id"],
                )
            if row["observe_scenario"] == "scenario3" and row["created_segment_id"]:
                self.assertEqual(
                    row["selected_segment_id"],
                    row["created_segment_id"],
                )

    def test_checkpoint_format_and_strict_protocol_stay_clean(self) -> None:
        checkpoint = load_strict_checkpoint(
            self.root / "on" / "checkpoint.pkl"
        )
        self.assertEqual(checkpoint["checkpoint_format"], "fig9-strict-v1")
        protocol = json.loads(
            (self.root / "on" / "original_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(
            any("teacher" in key.lower() for key in protocol)
        )

    def test_off_has_no_teacher_trace_and_on_analysis_joins(self) -> None:
        self.assertFalse(
            (self.root / "off" / "teacher_forced_observation_trace.csv").exists()
        )
        summary = analyze(
            run_dir=self.root / "on",
            output_dir=self.root / "analysis",
            policy_label="batched",
            bootstrap_samples=10,
            bootstrap_seed=2,
        )
        self.assertGreater(summary["target_columns"], 0)
        self.assertEqual(summary["target_index_mapping_valid_rate"], 1.0)

    def test_reference_ids_survive_checkpoint_reload(self) -> None:
        path = self.root / "on" / "teacher_forced_observation_trace.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        expected = stable_observation_reference_id(
            stream_label=row["stream_label"],
            actual_record_index=int(row["actual_record_index"]),
            actual_timestamp=row["actual_timestamp"],
            target_column=int(row["target_column"]),
        )
        self.assertEqual(row["observation_reference_id"], expected)
        self.assertEqual(
            load_strict_checkpoint(
                self.root / "on" / "checkpoint.pkl"
            )["checkpoint_format"],
            "fig9-strict-v1",
        )
