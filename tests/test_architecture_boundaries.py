from __future__ import annotations

import ast
import importlib
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from seqmem.model import (
    MemoryParams,
    MiniColumn,
    Neuron,
    Segment,
    SequentialMemory,
    Synapse,
)


def imported_modules(path: Path) -> set[str]:
    """返回 Python 文件中静态声明的 import 模块名。"""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class ArchitectureBoundaryTests(unittest.TestCase):
    """锁定重构期间不能跨越的依赖和 pickle 兼容边界。"""

    def test_seqmem_does_not_import_experiments_or_diagnostics(self) -> None:
        """核心模型不能反向依赖实验入口或诊断模块。"""

        for path in (ROOT / "src" / "seqmem").rglob("*.py"):
            with self.subTest(path=path):
                modules = imported_modules(path)
                self.assertFalse(
                    any(
                        module == "experiments"
                        or module.startswith("experiments.")
                        or ".diagnostics" in module
                        for module in modules
                    )
                )

    def test_strict_entrypoint_does_not_import_historical_package(self) -> None:
        """strict 入口不能依赖未来隔离出的 historical 包。"""

        path = ROOT / "experiments" / "fig9_strict_reproduction.py"
        modules = imported_modules(path)
        self.assertFalse(
            any(
                module == "experiments.historical"
                or module.startswith("experiments.historical.")
                for module in modules
            )
        )

    def test_importing_diagnostics_preserves_memory_defaults(self) -> None:
        """加载诊断模块不得修改全局 strict/model 默认参数。"""

        before = asdict(MemoryParams())
        importlib.import_module("experiments.diagnostics.fig8_diagnostic_common")
        self.assertEqual(asdict(MemoryParams()), before)

    def test_pickle_class_module_paths_remain_stable(self) -> None:
        """旧 pickle checkpoint 依赖这些完整类路径。"""

        for cls in (
            Synapse,
            Segment,
            Neuron,
            MiniColumn,
            MemoryParams,
            SequentialMemory,
        ):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls.__module__, "seqmem.model")


if __name__ == "__main__":
    unittest.main()
