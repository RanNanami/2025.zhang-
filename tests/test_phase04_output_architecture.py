from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from experiments.fig9.outputs import (
    StrictStreamOutputPaths,
    write_initial_stream_artifacts,
    write_optional_stream_results,
    write_runtime_artifacts,
)
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    model_long_term_fingerprint,
    model_rng_fingerprint,
    parse_args,
    transient_fingerprint,
)
from seqmem.encoding import SSTDDiscreteEncoder
from seqmem.model import MemoryParams, SequentialMemory


ROOT = Path(__file__).resolve().parents[1]
STRICT_PATH = ROOT / "experiments/fig9_strict_reproduction.py"
OUTPUT_PATH = ROOT / "experiments/fig9/outputs.py"

CHECKPOINT_SOURCE_HASHES = {
    "save_strict_checkpoint": "ef1dbf66c568ef08b5f2e514e9be3318c1005a21dc4aeae8fa335651462a2e11",
    "checkpoint_metadata_path": "eacc58fdf07f27e5048f2e1e09637889c781b7c0458c90c80daa2be5b7aca5c4",
    "write_checkpoint_metadata": "cd3872d90304e1db6ce42a12eee428d3ff4853fb2b672f789b1ad049e875dc40",
    "_StrictCheckpointUnpickler": "9c7f19e9afd6264d1e9cb73bd4b64aa6489ba96bf836b173863c1dd20234dc04",
    "load_strict_checkpoint": "84f39213024e96e50272a2e2584d2ccb9117bf06c51bb7c171531e8be60e8558",
    "validate_checkpoint_l_match": "6b8ffc0038ebc21d348aa4c8c225eb99cdddf34fb57870c9fe82fb8df161d80e",
    "branch_checkpoint_sidecar_path": "c63749f84fb4d3b5bd1763b5b0c53d5ed2d862fbd30cdd4e77b8085eb1ddaaad",
}
NATIVE_SOURCE_HASHES = {
    "close_native_crash_logging": "f18d7868df89c0fbacdc848fefec42b7d3806963d8fa6c50d75fc6f43d508256",
    "_periodic_native_traceback": "d656c892afceba052231374ad511a7ac87b02db1d80b66e9f7a1df5a64d97b82",
    "install_native_crash_logging": "a34523cf1df2b4dc5a21de5c008cdcc1922cf8a6b9e5f3235a036e805a7e4b5d",
    "write_native_crash_breadcrumb": "d3a160cb619a824be5db22c12914dc6d5630e0a19d0ea45f970c1270dc97d1fe",
    "write_native_runtime_debug_breadcrumb": "db9ba5f2331434993e36ce3dcdd546aab582e4454a02f3c39515ba52081bc081",
    "PruneBreadcrumbRecorder": "b0bab6bc06deed25159814c7b50cf5ebac8935aad94911a9ed9c431c507bcc9c",
}
CHECKPOINT_PAYLOAD_KEYS = {
    "checkpoint_format",
    "git_commit_sha",
    "encoder",
    "model",
    "fingerprint",
    "config",
    "L_match",
    "stream_label",
    "data_path",
    "data_file_sha256",
    "prefix_end_index",
    "prefix_end_timestamp",
    "common_prefix_hash",
    "perturbation_split_timestamp",
    "limit",
    "next_index",
    "predictions",
    "targets",
    "rows",
    "recent_errors",
    "missing",
    "raw_event_counts",
    "raw_column_counts",
    "density_rows",
    "long_sequence_rows",
    "branch_provenance",
    "diagnostic_state",
}


def _top_level_source_hashes(path: Path) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    return {
        node.name: hashlib.sha256(
            ast.get_source_segment(source, node).encode("utf-8")
        ).hexdigest()
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def _checkpoint_payload_keys() -> set[str]:
    tree = ast.parse(STRICT_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "save_strict_checkpoint"
    )
    assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "payload" for target in node.targets)
    )
    assert isinstance(assignment.value, ast.Dict)
    return {
        key.value
        for key in assignment.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


class Phase04OutputArchitectureTests(unittest.TestCase):
    def _model(self) -> SequentialMemory:
        encoder = SSTDDiscreteEncoder(num_columns=12, k=3, seed=7)
        model = SequentialMemory(
            encoder=encoder,
            num_neurons_per_column=4,
            params=MemoryParams(l_match=2, forgetting_threshold=50.0),
            tie_break_seed=7,
        )
        model.predict_code()
        model.observe("A")
        return model

    def _science_state(self, model: SequentialMemory) -> tuple[str, str, str]:
        return (
            model_long_term_fingerprint(model),
            model_rng_fingerprint(model),
            transient_fingerprint(model),
        )

    def test_output_module_has_no_scientific_import_or_parameter(self) -> None:
        tree = ast.parse(OUTPUT_PATH.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertFalse(any(name.startswith("seqmem") for name in imports))
        for function in (
            write_initial_stream_artifacts,
            write_optional_stream_results,
            write_runtime_artifacts,
        ):
            self.assertTrue(
                {"model", "encoder", "rng"}.isdisjoint(
                    inspect.signature(function).parameters
                )
            )

    def test_output_enabled_and_disabled_science_fingerprints_match(self) -> None:
        model = self._model()
        disabled = self._science_state(model)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = StrictStreamOutputPaths(root, "original")
            write_initial_stream_artifacts(
                paths,
                [{"target": 1.25, "prediction": 2.5}],
                {"mape": 0.5},
                {"L_match": 4},
            )
            write_optional_stream_results(
                paths,
                interval_rows=[{"start": 0, "end": 10}],
                debug_payload={"record": 3},
                debug_output_path=None,
            )
            write_runtime_artifacts(root, {"finished": True})
        self.assertEqual(disabled, self._science_state(model))

    def test_strict_config_and_cli_defaults_are_unchanged(self) -> None:
        self.assertEqual(
            {
                "horizon": 5,
                "warmup": 5904,
                "rolling_window": 400,
                "seed": 0,
                "weekday_columns": 30,
                "time_columns": 58,
                "passenger_columns": 482,
                "k": 10,
                "neurons_per_column": 32,
                "l_match": 4,
                "forgetting_threshold": 65.0,
                "passenger_min": 0.0,
                "passenger_max": 40000.0,
                "response_scale": None,
                "continuous_dynamics": True,
                "integration_step": 0.005,
                "burst_context": True,
                "intracolumn_inhibition": True,
                "propagation_mode": "raw",
                "scenario1_rule": "direct-reinforce-predicted-segment",
                "use_future_covariates": False,
                "reencode_decoded_value": False,
                "rollout_learning": False,
                "restore_transient_state": True,
                "strict_label": "strict",
                "continuous_prediction_impl": "reference",
            },
            asdict(Fig9StrictConfig()),
        )
        with patch.object(sys, "argv", ["fig9_strict_reproduction.py"]):
            args = parse_args()
        self.assertEqual(5, args.prediction_horizon)
        self.assertEqual(5904, args.warmup)
        self.assertEqual(4, args.l_match)
        self.assertEqual("reference", args.continuous_impl)
        self.assertEqual("off", args.competition_mode)

    def test_protocol_serialization_is_byte_compatible(self) -> None:
        summary = {"z": 2, "a": {"value": 1}}
        protocol = {"seed": 0, "L_match": 4}
        with tempfile.TemporaryDirectory() as temporary:
            paths = StrictStreamOutputPaths(Path(temporary), "original")
            write_initial_stream_artifacts(paths, [], summary, protocol)
            self.assertEqual(
                json.dumps(summary, indent=2, sort_keys=True)
                .replace("\n", os.linesep)
                .encode("utf-8"),
                paths.summary.read_bytes(),
            )
            self.assertEqual(
                json.dumps(protocol, indent=2, sort_keys=True)
                .replace("\n", os.linesep)
                .encode("utf-8"),
                paths.protocol.read_bytes(),
            )

    def test_prediction_csv_is_byte_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = StrictStreamOutputPaths(Path(temporary), "original")
            write_initial_stream_artifacts(
                paths,
                [{"target": 1.25, "prediction": 2.5}],
                {},
                {},
            )
            self.assertEqual(
                b"target,prediction\r\n1.25,2.5\r\n",
                paths.predictions.read_bytes(),
            )

    def test_core_to_experiments_imports_remain_zero(self) -> None:
        violations = []
        for path in (ROOT / "src/seqmem").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else []
                )
                if any(
                    name == "experiments" or name.startswith("experiments.")
                    for name in modules
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual([], violations)

    def test_diagnostic_lazy_loading_boundary_does_not_regress(self) -> None:
        code = """
import sys
import experiments.fig9_strict_reproduction
print(';'.join(sorted(name for name in sys.modules if name.startswith('experiments.diagnostics'))))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(
            "experiments.diagnostics;experiments.diagnostics.fig9_competitive_inhibition",
            completed.stdout.strip(),
        )

    def test_checkpoint_paths_schema_and_implementation_are_unchanged(self) -> None:
        hashes = _top_level_source_hashes(STRICT_PATH)
        self.assertEqual(
            CHECKPOINT_SOURCE_HASHES,
            {name: hashes[name] for name in CHECKPOINT_SOURCE_HASHES},
        )
        self.assertEqual(CHECKPOINT_PAYLOAD_KEYS, _checkpoint_payload_keys())

    def test_native_crash_hooks_are_unchanged(self) -> None:
        hashes = _top_level_source_hashes(STRICT_PATH)
        self.assertEqual(
            NATIVE_SOURCE_HASHES,
            {name: hashes[name] for name in NATIVE_SOURCE_HASHES},
        )


if __name__ == "__main__":
    unittest.main()
