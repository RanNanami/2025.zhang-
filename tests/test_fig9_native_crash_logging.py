from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from seqmem import dynamics as dynamics_module
from seqmem.dynamics import DSDynamicsParams
from experiments.fig9_strict_reproduction import (
    Fig9StrictConfig,
    build_fig9_encoder,
    build_strict_model,
    check_module_state_integrity,
    close_native_crash_logging,
    install_native_crash_logging,
    module_state_integrity_snapshot,
    native_environment,
    run_strict_stream,
    write_native_crash_breadcrumb,
    working_set_memory_mb,
)
from experiments.fig9 import TaxiRecord
from datetime import datetime, timedelta


class Fig9NativeCrashLoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        close_native_crash_logging()

    def test_dedicated_faulthandler_file_and_breadcrumb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = install_native_crash_logging(root)
            model = build_strict_model(
                build_fig9_encoder(Fig9StrictConfig()), Fig9StrictConfig()
            )
            write_native_crash_breadcrumb(
                current_index=210,
                rollout_step=2,
                phase="unit_test",
                model=model,
            )
            content = path.read_text(encoding="utf-8")
            self.assertIn("NATIVE_ENVIRONMENT", content)
            self.assertIn("PERIODIC_TRACEBACK disabled", content)
            self.assertIn("FIG9_NATIVE_PHASE", content)
            self.assertIn('"current_index": 210', content)
            self.assertIn('"rollout_step": 2', content)
            environment = json.loads(
                (root / "native_environment.json").read_text(encoding="utf-8")
            )
            for key in ("python", "platform", "zlib_runtime", "packages"):
                self.assertIn(key, environment)
            close_native_crash_logging()

    def test_working_set_measurement_is_positive(self) -> None:
        self.assertGreater(working_set_memory_mb(), 0.0)

    def test_debug_end_index_cannot_advance_more_than_ten(self) -> None:
        records = [
            TaxiRecord(
                datetime(2014, 7, 1) + timedelta(minutes=30 * index),
                100.0 + index,
            )
            for index in range(20)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "tiny.csv"
            data.write_text("timestamp,passenger_count\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "limited to 10 records"):
                run_strict_stream(
                    records=records,
                    data_path=data,
                    stream_label="original",
                    output_dir=root / "run",
                    config=Fig9StrictConfig(warmup=30),
                    limit=20,
                    print_fingerprint=False,
                    debug_end_index=11,
                )

    def test_environment_collection_does_not_require_optional_imports(self) -> None:
        payload = native_environment()
        self.assertIn("numpy", payload["packages"])
        self.assertIn("pandas", payload["packages"])
        self.assertIn("scipy", payload["packages"])

    def test_module_state_integrity_records_expected_types(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            install_native_crash_logging(root, native_runtime_debug=True)
            config = Fig9StrictConfig()
            model = build_strict_model(build_fig9_encoder(config), config)
            before = module_state_integrity_snapshot(model)
            event = check_module_state_integrity(
                model=model,
                phase="run_start",
                current_index=0,
                rollout_step=None,
                stream="original",
            )
            after = module_state_integrity_snapshot(model)
            self.assertEqual(before, after)
            self.assertIsNotNone(event)
            self.assertEqual(event["invalid_reasons"], [])
            log = (root / "native_runtime_debug.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertIn("MODULE_STATE_INTEGRITY_CHECK", log)
            self.assertIn('"exp_is_expected": true', log)
            close_native_crash_logging()

    def test_module_state_integrity_detects_exp_rebinding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            install_native_crash_logging(root, native_runtime_debug=True)
            config = Fig9StrictConfig()
            model = build_strict_model(build_fig9_encoder(config), config)
            check_module_state_integrity(
                model=model,
                phase="run_start",
                current_index=0,
                rollout_step=None,
                stream="original",
            )
            original_exp = dynamics_module._EXP
            try:
                dynamics_module._EXP = DSDynamicsParams()
                with self.assertRaisesRegex(
                    RuntimeError, "GLOBAL_STATE_CORRUPTION_DETECTED"
                ):
                    check_module_state_integrity(
                        model=model,
                        phase="rollout_step_enter",
                        current_index=210,
                        rollout_step=2,
                        stream="original",
                    )
            finally:
                dynamics_module._EXP = original_exp
            runtime_log = (root / "native_runtime_debug.jsonl").read_text(
                encoding="utf-8"
            )
            crash_log = (root / "native_crash_faulthandler.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("DYNAMICS_EXP_REBOUND", runtime_log)
            self.assertIn("GLOBAL_STATE_CORRUPTION_DETECTED", crash_log)
            close_native_crash_logging()


if __name__ == "__main__":
    unittest.main()
