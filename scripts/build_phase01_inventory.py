"""Build read-only Phase 0/1 repository audit artifacts.

The script parses source files and inspects artifact metadata. It deliberately
does not import experiment or model modules, execute a runner, or mutate any
scientific state.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


SNAPSHOT_COMMIT = "1864f46aee5cc0aee814fd4e6154b06f8bfeb08f"
PHASE_REPORT_PREFIX = "reports/refactor/phase01_20260814_233129/"
CRITICAL_BACKUP_PREFIXES = (
    "results/fig9_diagnostics/lmatch_stack_2x2_20260810/250/STRICT_RAW_L4/",
    "results/fig9_diagnostics/lmatch_stack_2x2_20260810/250/STRICT_RAW_L2/",
    "results/temporal_confirmation_semantics_20260814_phaseC_20_final_v2/",
)
CRITICAL_BACKUP_FILES = {
    "results/temporal_confirmation_semantics_20260814_phaseC_20_final_v2_conclusions.zip"
}
CHECKPOINT_CLASS_NAMES = {
    "Synapse",
    "Segment",
    "Neuron",
    "MiniColumn",
    "MemoryParams",
    "SequentialMemory",
}


@dataclass
class PythonFile:
    path: str
    module: str
    role: str
    loc: int
    imports: list[tuple[str, int, str]]
    classes: list[str]
    functions: list[str]
    module_docstring: bool
    documented_definitions: int
    definitions: int
    comment_lines: int


def git_lines(root: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return [line for line in result.stdout.splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def module_name(path: str) -> str:
    candidate = Path(path)
    parts = list(candidate.with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def file_role(path: str) -> str:
    if path.startswith("src/seqmem/"):
        return "core"
    if path.startswith("tests/"):
        return "test"
    if path.startswith("experiments/diagnostics/"):
        return "diagnostic"
    if path.startswith("experiments/historical/"):
        return "historical"
    if path.startswith("experiments/fig"):
        return "experiment"
    if path.startswith("scripts/") or path.startswith("tools/"):
        return "infrastructure"
    return "support"


def resolve_relative_import(source_module: str, level: int, imported: str | None) -> str:
    if level == 0:
        return imported or ""
    package = source_module.split(".")[:-1]
    keep = max(0, len(package) - level + 1)
    package = package[:keep]
    if imported:
        package.extend(imported.split("."))
    return ".".join(package)


def parse_python_file(root: Path, path: str) -> PythonFile:
    source = (root / path).read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=path)
    imports: list[tuple[str, int, str]] = []
    classes: list[str] = []
    functions: list[str] = []
    documented = 0
    definitions = 0
    source_module = module_name(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno, "import"))
        elif isinstance(node, ast.ImportFrom):
            target = resolve_relative_import(source_module, node.level, node.module)
            imports.append((target, node.lineno, "from"))
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
            definitions += 1
            documented += int(ast.get_docstring(node) is not None)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
            definitions += 1
            documented += int(ast.get_docstring(node) is not None)
    comments = sum(1 for line in source.splitlines() if line.lstrip().startswith("#"))
    return PythonFile(
        path=path,
        module=source_module,
        role=file_role(path),
        loc=len(source.splitlines()),
        imports=imports,
        classes=classes,
        functions=functions,
        module_docstring=ast.get_docstring(tree) is not None,
        documented_definitions=documented,
        definitions=definitions,
        comment_lines=comments,
    )


def local_import_target(target: str, modules: set[str]) -> str | None:
    candidate = target
    while candidate:
        if candidate in modules:
            return candidate
        candidate = candidate.rpartition(".")[0]
    return None


def build_python_inventory(root: Path, report: Path) -> dict[str, object]:
    paths = sorted(git_lines(root, "ls-files", "*.py"))
    parsed: dict[str, PythonFile] = {}
    parse_errors: list[tuple[str, str]] = []
    for path in paths:
        try:
            parsed[path] = parse_python_file(root, path)
        except (OSError, SyntaxError) as exc:
            parse_errors.append((path, f"{type(exc).__name__}: {exc}"))

    module_to_path = {item.module: path for path, item in parsed.items() if item.module}
    modules = set(module_to_path)
    edges: list[dict[str, object]] = []
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    for source_path, item in parsed.items():
        for raw_target, line, kind in item.imports:
            target_module = local_import_target(raw_target, modules)
            if target_module is None:
                continue
            target_path = module_to_path[target_module]
            outgoing[source_path].add(target_path)
            incoming[target_path].add(source_path)
            source_role = item.role
            target_role = parsed[target_path].role
            violation = ""
            if source_role == "core" and target_role != "core":
                violation = "CORE_DEPENDS_OUTWARD"
            elif source_path == "experiments/fig9_strict_reproduction.py" and target_role == "diagnostic":
                violation = "STRICT_DEPENDS_ON_DIAGNOSTIC"
            elif source_role != "historical" and target_role == "historical":
                violation = "ACTIVE_DEPENDS_ON_HISTORICAL"
            edges.append(
                {
                    "source_path": source_path,
                    "source_module": item.module,
                    "source_role": source_role,
                    "target_path": target_path,
                    "target_module": target_module,
                    "target_role": target_role,
                    "line": line,
                    "import_kind": kind,
                    "architecture_flag": violation,
                }
            )

    test_paths = {path for path, item in parsed.items() if item.role == "test"}
    direct_test_targets = set().union(*(outgoing[path] for path in test_paths)) if test_paths else set()
    indirect_test_targets: set[str] = set()
    queue: deque[str] = deque(direct_test_targets)
    while queue:
        current = queue.popleft()
        if current in indirect_test_targets:
            continue
        indirect_test_targets.add(current)
        queue.extend(outgoing[current] - indirect_test_targets)

    inventory_rows: list[dict[str, object]] = []
    for path, item in sorted(parsed.items()):
        inventory_rows.append(
            {
                "path": path,
                "module": item.module,
                "role": item.role,
                "loc": item.loc,
                "class_count": len(item.classes),
                "function_count": len(item.functions),
                "local_import_count": len(outgoing[path]),
                "imported_by_count": len(incoming[path]),
                "direct_test_reference": path in direct_test_targets,
                "indirect_test_reference": path in indirect_test_targets,
                "checkpoint_sensitive": bool(CHECKPOINT_CLASS_NAMES.intersection(item.classes)),
                "module_docstring": item.module_docstring,
                "documented_definition_ratio": (
                    f"{item.documented_definitions / item.definitions:.3f}"
                    if item.definitions
                    else ""
                ),
                "comment_lines": item.comment_lines,
            }
        )
    write_csv(
        report / "PYTHON_FILE_INVENTORY.csv",
        list(inventory_rows[0]),
        inventory_rows,
    )
    write_csv(
        report / "IMPORT_DEPENDENCY_EDGES.csv",
        list(edges[0]) if edges else ["source_path", "target_path"],
        edges,
    )

    largest = sorted(parsed.values(), key=lambda item: (-item.loc, item.path))[:20]
    large_lines = [
        "# Large Python Modules",
        "",
        "Line counts use Python `splitlines()` over tracked files at the frozen snapshot.",
        "",
        "| Rank | Lines | Role | File | Classes | Functions |",
        "| ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for rank, item in enumerate(largest, 1):
        large_lines.append(
            f"| {rank} | {item.loc} | {item.role} | `{item.path}` | "
            f"{len(item.classes)} | {len(item.functions)} |"
        )
    large_lines.extend(
        [
            "",
            "## Risk interpretation",
            "",
            "`experiments/fig9_strict_reproduction.py` and `src/seqmem/model.py` are the two",
            "highest-risk modules. Their size reflects multiple responsibilities and long-lived",
            "checkpoint paths. Neither is a suitable first extraction target. The next largest",
            "files are predominantly diagnostic analyzers and their tests; they are easier to",
            "isolate later because they are not checkpoint class owners.",
        ]
    )
    write_text(report / "LARGE_MODULE_REPORT.md", "\n".join(large_lines))

    core_outward = [edge for edge in edges if edge["architecture_flag"] == "CORE_DEPENDS_OUTWARD"]
    strict_diag = [edge for edge in edges if edge["architecture_flag"] == "STRICT_DEPENDS_ON_DIAGNOSTIC"]
    active_historical = [edge for edge in edges if edge["architecture_flag"] == "ACTIVE_DEPENDS_ON_HISTORICAL"]
    dependency_lines = [
        "# Import Dependency Audit",
        "",
        f"- Tracked Python files parsed: `{len(parsed)}`",
        f"- Parse errors: `{len(parse_errors)}`",
        f"- Local import edges: `{len(edges)}`",
        f"- Core to non-core violations: `{len(core_outward)}`",
        f"- Fig.9 strict to diagnostics edges: `{len(strict_diag)}`",
        f"- Active to historical edges: `{len(active_historical)}`",
        "",
        "## Findings",
        "",
        "The `src/seqmem` core has no import edge into `experiments`, diagnostics, or",
        "historical code. This boundary is already healthy and should become an enforced",
        "architecture test.",
        "",
        "The Fig.9 strict runner directly imports many diagnostic modules. Even though",
        "their CLI flags default off, this makes strict startup, auditability, and failure",
        "isolation depend on nonpaper code. Future runner decomposition should invert this",
        "relationship after golden baselines exist; this phase only records it.",
        "",
        "## Strict diagnostic dependencies",
        "",
    ]
    dependency_lines.extend(
        f"- `{edge['target_path']}` (line {edge['line']})" for edge in strict_diag
    )
    dependency_lines.extend(["", "## Parse errors", ""])
    dependency_lines.extend(
        f"- `{path}`: {message}" for path, message in parse_errors
    )
    if not parse_errors:
        dependency_lines.append("None.")
    write_text(report / "IMPORT_DEPENDENCY_AUDIT.md", "\n".join(dependency_lines))

    return {
        "count": len(parsed),
        "largest": [{"path": item.path, "loc": item.loc} for item in largest],
        "core_outward": len(core_outward),
        "strict_diagnostic_edges": len(strict_diag),
        "active_historical_edges": len(active_historical),
        "parsed": parsed,
        "edges": edges,
    }


def count_text_rows(path: Path) -> int | None:
    if path.suffix.lower() not in {".csv", ".txt", ".json"}:
        return None
    if path.stat().st_size > 512 * 1024 * 1024:
        return None
    with path.open("rb") as handle:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))


def dataset_metadata(path: str) -> tuple[str, str, str, str]:
    if path == "data/CBTest/data/cbt_train.txt":
        return "Fig8", "PAPER_DATASET", "CBT source corpus", "Facebook CBT/ParlAI-compatible copy"
    if path == "data/paper_nyc_taxi.csv":
        return "Fig9", "PAPER_CANDIDATE", "strict original stream", "reference [58] processed NYC taxi candidate"
    if path == "data/paper_nyc_taxi_perturb.csv":
        return "Fig9", "PAPER_CANDIDATE", "strict changed stream", "derived perturbation aligned to reference [58]"
    if path == "data/ETTh1.csv":
        return "Transfer", "APPLICATION_TRANSFER", "ETTh1 stress test", "public ETT benchmark"
    if path == "data/weather/weather.csv":
        return "Transfer", "APPLICATION_TRANSFER", "Weather stress test", "public Weather benchmark"
    if "chinese_poetry" in path or "fig8c" in path:
        return "Fig8c", "NONPAPER_APPROXIMATION", "poem approximation", "public chinese-poetry-derived data"
    if path == "data/nyc_taxi.csv":
        return "Fig9", "HISTORICAL_INPUT", "legacy taxi input", "local historical asset"
    return "Support", "SUPPORT_FILE", "dataset metadata/support", "repository file"


def build_dataset_manifest(root: Path, report: Path) -> dict[str, str]:
    rows: list[dict[str, object]] = []
    fig7_sources = [
        root / "experiments/fig7_sequence_prediction.py",
        root / "experiments/fig7_repeated_trials.py",
    ]
    fig7_hash = sha256_files(fig7_sources)
    rows.append(
        {
            "path": "GENERATED_IN_MEMORY:fig7_reference58_sequence_stream",
            "experiment_family": "Fig7",
            "classification": "PAPER_PROTOCOL_GENERATED",
            "role": "synthetic sequence/noise stream generator",
            "source_claim": "reference [58] sequences and generator semantics",
            "bytes": sum(path.stat().st_size for path in fig7_sources),
            "rows_or_lines": "generated",
            "sha256": fig7_hash,
            "tracked": True,
            "notes": "Hash covers both tracked Fig.7 generator/runner source files.",
        }
    )
    hashes = {"fig7_generated_protocol": fig7_hash}
    for relative in sorted(git_lines(root, "ls-files", "data")):
        path = root / relative
        family, classification, role, source = dataset_metadata(relative)
        digest = sha256_file(path)
        rows.append(
            {
                "path": relative,
                "experiment_family": family,
                "classification": classification,
                "role": role,
                "source_claim": source,
                "bytes": path.stat().st_size,
                "rows_or_lines": count_text_rows(path),
                "sha256": digest,
                "tracked": True,
                "notes": "",
            }
        )
        if relative in {
            "data/CBTest/data/cbt_train.txt",
            "data/paper_nyc_taxi.csv",
            "data/paper_nyc_taxi_perturb.csv",
            "data/ETTh1.csv",
            "data/weather/weather.csv",
        }:
            hashes[relative] = digest
    write_csv(report / "DATASET_MANIFEST.csv", list(rows[0]), rows)
    return hashes


def artifact_classification(path: str) -> tuple[str, str, str]:
    lower = path.lower()
    backed_up = path in CRITICAL_BACKUP_FILES or any(
        path.startswith(prefix) for prefix in CRITICAL_BACKUP_PREFIXES
    )
    if backed_up:
        return "CRITICAL", "BACKED_UP_EXTERNAL_ZIP", "KEEP_CANONICAL"
    name = Path(path).name.lower()
    important_name = (
        name == "checkpoint.pkl"
        or name.endswith("protocol.json")
        or name.endswith("summary.json")
        or name.endswith("predictions.csv")
        or name.endswith("report.md")
    )
    if important_name and not any(token in lower for token in ("smoke", "debug", "isolation", "dryrun")):
        action = "KEEP_CHECKPOINT" if name == "checkpoint.pkl" else "KEEP_REPORT"
        return "IMPORTANT", "PRESENT_IN_WORKTREE", action
    if any(token in lower for token in ("__pycache__", ".pstats", ".tmp", "pytest_cache")):
        return "TEMPORARY", "PRESENT_IN_WORKTREE", "TEMP_DELETE_CANDIDATE"
    if any(token in lower for token in ("smoke", "debug", "isolation", "trace", ".log", ".png", ".gz")):
        return "REGENERABLE", "PRESENT_IN_WORKTREE", "REGENERABLE"
    if lower.endswith(".zip"):
        return "REGENERABLE", "PRESENT_IN_WORKTREE", "DUPLICATE_CANDIDATE"
    return "IMPORTANT", "PRESENT_IN_WORKTREE", "ARCHIVE_HISTORICAL"


def build_artifact_manifest(root: Path, report: Path) -> dict[str, int]:
    untracked = sorted(git_lines(root, "ls-files", "--others", "--exclude-standard"))
    original = [
        path
        for path in untracked
        if not path.startswith(PHASE_REPORT_PREFIX)
        and path != "scripts/build_phase01_inventory.py"
    ]
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for relative in original:
        path = root / relative
        classification, backup_status, action = artifact_classification(relative)
        counts[classification] += 1
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size if path.is_file() else "",
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                "classification": classification,
                "backup_status": backup_status,
                "suggested_future_action": action,
                "reason": (
                    "selected canonical untracked artifact"
                    if classification == "CRITICAL"
                    else "heuristic inventory; requires human review before deletion"
                ),
            }
        )
    write_csv(report / "PRE_REFACTOR_ARTIFACT_MANIFEST.csv", list(rows[0]), rows)
    return dict(counts)


def infer_experiment_family(path: str) -> str:
    lower = path.lower()
    if "fig7" in lower:
        return "Fig7"
    if "fig8" in lower or "temporal_confirmation" in lower:
        return "Fig8"
    if "fig9" in lower or "lmatch" in lower or "runtime_" in lower:
        return "Fig9"
    if "weather" in lower or "etth" in lower or "transfer" in lower:
        return "Transfer"
    return "Other"


def result_credibility(path: str) -> str:
    lower = path.lower()
    if any(token in lower for token in ("historical", "compensated", "proximal", "bug", "failed")):
        return "D"
    if any(token in lower for token in ("diagnostic", "oracle", "teacher", "smoke", "debug", "ablation", "transfer")):
        return "C"
    if any(token in lower for token in ("strict", "lmatch_stack_2x2")):
        return "B"
    return "UNCLASSIFIED"


def build_results_manifest(root: Path, report: Path) -> int:
    results = root / "results"
    aggregate: dict[Path, dict[str, object]] = defaultdict(
        lambda: {"size": 0, "mtime": 0.0, "names": set()}
    )
    for current, _, files in os.walk(results):
        current_path = Path(current)
        for name in files:
            path = current_path / name
            try:
                stat = path.stat()
            except OSError:
                continue
            relative_dir = current_path.relative_to(root)
            node = relative_dir
            while str(node).startswith("results"):
                aggregate[node]["size"] = int(aggregate[node]["size"]) + stat.st_size
                aggregate[node]["mtime"] = max(float(aggregate[node]["mtime"]), stat.st_mtime)
                cast_names = aggregate[node]["names"]
                assert isinstance(cast_names, set)
                cast_names.add(name.lower())
                if node == Path("results"):
                    break
                node = node.parent
    candidates = [
        path
        for path in aggregate
        if path != Path("results")
        and (
            path.parent == Path("results")
            or path.parent == Path("results/fig9_diagnostics")
            or any(
                name in aggregate[path]["names"]
                for name in ("checkpoint.pkl", "protocol.json", "original_protocol.json", "summary.json")
            )
        )
    ]
    rows: list[dict[str, object]] = []
    for relative in sorted(candidates, key=lambda item: item.as_posix()):
        names = aggregate[relative]["names"]
        assert isinstance(names, set)
        path_text = relative.as_posix()
        credibility = result_credibility(path_text)
        temporary = any(token in path_text.lower() for token in ("smoke", "debug", "tmp", "isolation"))
        if temporary:
            action = "REGENERABLE"
        elif "checkpoint.pkl" in names:
            action = "KEEP_CHECKPOINT"
        elif credibility in {"B", "C"}:
            action = "KEEP_REPORT"
        elif credibility == "D":
            action = "ARCHIVE_HISTORICAL"
        else:
            action = "UNKNOWN"
        rows.append(
            {
                "path": path_text,
                "directory": relative.name,
                "size_bytes_recursive": aggregate[relative]["size"],
                "mtime_latest": datetime.fromtimestamp(float(aggregate[relative]["mtime"])).isoformat(),
                "experiment_family": infer_experiment_family(path_text),
                "known_commit": "",
                "protocol_available": any(name.endswith("protocol.json") for name in names),
                "checkpoint": "checkpoint.pkl" in names,
                "report": any(name.endswith(".md") or "report" in name for name in names),
                "raw_prediction": any("prediction" in name and name.endswith(".csv") for name in names),
                "temporary": temporary,
                "credibility": credibility,
                "regeneration_cost": "HIGH" if "checkpoint.pkl" in names else ("LOW" if temporary else "MEDIUM"),
                "suggested_future_action": action,
            }
        )
    write_csv(report / "RESULTS_MANIFEST.csv", list(rows[0]), rows)
    return len(rows)


def build_diagnostics_manifest(
    root: Path, report: Path, python_audit: dict[str, object]
) -> int:
    parsed = python_audit["parsed"]
    assert isinstance(parsed, dict)
    edges = python_audit["edges"]
    assert isinstance(edges, list)
    strict_targets = {
        edge["target_path"]
        for edge in edges
        if edge["source_path"] == "experiments/fig9_strict_reproduction.py"
    }
    rows: list[dict[str, object]] = []
    for relative in sorted(git_lines(root, "ls-files", "experiments/diagnostics/*.py")):
        source = (root / relative).read_text(encoding="utf-8", errors="replace")
        lower = source.lower()
        item = parsed.get(relative)
        purpose = ""
        if item is not None:
            try:
                purpose = ast.get_docstring(ast.parse(source)) or ""
            except SyntaxError:
                pass
        oracle = "oracle" in relative.lower() or "oracle" in lower
        teacher = "teacher" in relative.lower() or "teacher-forced" in lower or "teacher_forced" in lower
        gt = oracle or teacher or "ground_truth" in lower or "expected_word" in lower
        side_effect = "learn=true" in lower or ".observe(" in lower or "observe_code(" in lower
        trace_only = any(token in relative.lower() for token in ("analyze_", "inspect_", "audit_", "compare_"))
        rows.append(
            {
                "path": relative,
                "purpose": purpose.splitlines()[0][:240] if purpose else "No module docstring",
                "classification": "NONPAPER_DIAGNOSTIC",
                "imported_by_strict_runner": relative in strict_targets,
                "potential_model_side_effects": side_effect,
                "trace_only_or_analyzer": trace_only,
                "oracle": oracle,
                "teacher_forced": teacher,
                "ground_truth_reference": gt,
                "future_location": "experiments/diagnostics",
                "suggested_action": "KEEP_ISOLATED" if relative in strict_targets else "KEEP_OR_ARCHIVE",
            }
        )
    write_csv(report / "DIAGNOSTICS_MANIFEST.csv", list(rows[0]), rows)
    return len(rows)


def literal_text(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return "<dynamic>"


def build_cli_inventory(root: Path, report: Path) -> int:
    rows: list[dict[str, object]] = []
    option_counts: Counter[str] = Counter()
    pending: list[dict[str, object]] = []
    for relative in sorted(git_lines(root, "ls-files", "experiments/*.py", "experiments/**/*.py")):
        source = (root / relative).read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "add_argument":
                continue
            options = [
                value.value
                for value in node.args
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            ]
            if not options:
                continue
            option = next((value for value in options if value.startswith("--")), options[0])
            kwargs = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
            lower = option.lower()
            diagnostic = (
                "/diagnostics/" in relative
                or any(token in lower for token in ("diagnostic", "trace", "oracle", "teacher", "debug", "profile"))
            )
            paper_facing = (
                relative
                in {
                    "experiments/fig7_sequence_prediction.py",
                    "experiments/fig7_repeated_trials.py",
                    "experiments/fig8_sentence_memory.py",
                    "experiments/fig9_strict_reproduction.py",
                }
                and not diagnostic
            )
            row = {
                "script": relative,
                "line": node.lineno,
                "option": option,
                "aliases": "|".join(options),
                "default": literal_text(kwargs.get("default")),
                "paper_facing": paper_facing,
                "diagnostic": diagnostic,
                "legacy": "historical" in relative or "legacy" in lower,
                "duplicated": False,
                "silent_override_risk": "HIGH" if diagnostic and paper_facing else "LOW",
                "future_config_group": "DiagnosticConfig" if diagnostic else infer_experiment_family(relative) + "ProtocolConfig",
            }
            pending.append(row)
            option_counts[option] += 1
    for row in pending:
        row["duplicated"] = option_counts[str(row["option"])] > 1
        rows.append(row)
    write_csv(report / "CLI_OPTION_INVENTORY.csv", list(rows[0]), rows)
    return len(rows)


def source_span(root: Path, relative: str, names: list[str]) -> list[tuple[str, int, int]]:
    tree = ast.parse((root / relative).read_text(encoding="utf-8"), filename=relative)
    found: list[tuple[str, int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            found.append((node.name, node.lineno, node.end_lineno or node.lineno))
    return found


def build_responsibility_maps(root: Path, report: Path) -> None:
    model_names = [
        "Synapse", "Segment", "Neuron", "MiniColumn", "PredictionCandidate",
        "TransientStateSnapshot", "MemoryParams", "SequentialMemory",
    ]
    spans = source_span(root, "src/seqmem/model.py", model_names)
    span_text = {name: f"{start}-{end}" for name, start, end in spans}
    model = f"""# Model Responsibility Map

`src/seqmem/model.py` is both the checkpoint compatibility owner and the main
scientific state machine. Its class module paths must remain stable until an
explicit checkpoint migration exists.

| Responsibility | Current symbols/range | State or sensitivity | First-phase decision |
| --- | --- | --- | --- |
| Persistent graph entities | `Synapse` {span_text.get('Synapse')}; `Segment` {span_text.get('Segment')}; `Neuron` {span_text.get('Neuron')}; `MiniColumn` {span_text.get('MiniColumn')} | Pickle class paths, weights, delay, age, active flags | Do not move |
| Prediction evidence | `PredictionCandidate` {span_text.get('PredictionCandidate')} and trace dataclasses | Candidate identity, crossing time, PSP metadata | Keep colocated initially |
| Transient state | `TransientStateSnapshot` {span_text.get('TransientStateSnapshot')} | Active cells, winners, candidates, RNG-neutral evaluation | Extract helper functions only after baselines |
| Protocol parameters | `MemoryParams` {span_text.get('MemoryParams')} | Defaults directly affect dynamics and checkpoint replay | Do not rename or reorder fields |
| Model facade/state machine | `SequentialMemory` {span_text.get('SequentialMemory')} | Prediction, observation, learning, pruning, propagation, RNG | Preserve module/class path |
| Prediction | `predict_code`, continuous prediction helpers, candidate selection | Floating accumulation order and tie-break sensitive | Late extraction |
| Decoding | `decode_*_from_prediction`, `_decode_candidates` | Decode RNG and ranking sensitive | Candidate for helper extraction after golden tests |
| Learning | `observe_code`, matching/growth/reinforcement/punishment | Scenario counts and graph mutation | Late extraction |
| Forgetting | `_prune_neuron`, `_live_incoming` | Segment/synapse identity and age | Medium-to-high checkpoint risk |
| Runtime diagnostics | callbacks and trace emitters | Must be read-only but currently mixed with core flow | Detach only with trajectory parity tests |

## Checkpoint-sensitive class paths

- `seqmem.model.Synapse`
- `seqmem.model.Segment`
- `seqmem.model.Neuron`
- `seqmem.model.MiniColumn`
- `seqmem.model.MemoryParams`
- `seqmem.model.SequentialMemory`

The proposed architecture keeps `model.py` as a compatibility facade. Pure
helpers may later move behind it, while object ownership and pickle-visible
paths remain unchanged.
"""
    write_text(report / "MODEL_RESPONSIBILITY_MAP.md", model)

    fig9_names = [
        "Fig9StrictConfig", "StrictRunState", "save_strict_checkpoint",
        "load_strict_checkpoint", "build_fig9_encoder", "build_strict_model",
        "rollout_raw_autonomous", "run_strict_stream", "parse_args", "run_main",
    ]
    spans = source_span(root, "experiments/fig9_strict_reproduction.py", fig9_names)
    lines = {name: f"{start}-{end}" for name, start, end in spans}
    fig9 = f"""# Fig.9 Runner Responsibility Map

The strict runner is a 6,000-line orchestration module. It imports stable Fig.9
helpers and many nonpaper diagnostics at module import time.

| Responsibility | Current symbol/range | Scientific or operational risk | Proposed treatment |
| --- | --- | --- | --- |
| Strict configuration | `Fig9StrictConfig` {lines.get('Fig9StrictConfig')} | Protocol/default risk | Keep public shape stable; later map CLI to config explicitly |
| Resume state | `StrictRunState` {lines.get('StrictRunState')} | Checkpoint compatibility | Do not move before migration tests |
| Native crash logging | lines 584-1129 | Runtime-only, file-handle sensitive | Future infrastructure extraction |
| Checkpoint IO | `save_strict_checkpoint` {lines.get('save_strict_checkpoint')}; `load_strict_checkpoint` {lines.get('load_strict_checkpoint')} | Highest operational compatibility risk | Extract only as wrappers preserving payload |
| Encoder/model construction | `build_fig9_encoder` {lines.get('build_fig9_encoder')}; `build_strict_model` {lines.get('build_strict_model')} | Protocol and RNG initialization | Add exact fingerprint gate before moving |
| Fingerprinting/protocol | lines 1506-1720 | Pure audit utilities | Best early extraction candidate |
| Autonomous rollout | `rollout_raw_autonomous` {lines.get('rollout_raw_autonomous')} | Core scientific trajectory | Do not extract early |
| Stream training/evaluation | `run_strict_stream` {lines.get('run_strict_stream')} | Combines prediction, observation, learning, diagnostics, output, resume | Decompose late and incrementally |
| CLI | `parse_args` {lines.get('parse_args')} | Hundreds of paper/diagnostic options share one namespace | Inventory now; separate config later without default changes |
| Top-level orchestration | `run_main` {lines.get('run_main')} | Dataset, streams, branches, profiling, summaries | Extract after runner regression gate |
| Diagnostic hooks | imports at lines 50-162 and branches in `run_strict_stream` | Strict depends on nonpaper modules | Future adapter boundary; defaults remain off |

## State transition timing

`rollout_raw_autonomous` snapshots transient state, advances raw predicted cell
identities for each horizon step, and restores the observation trajectory.
Actual observation then enters `model.observe_code(..., learn=True)` inside
`run_strict_stream`, updating long-term graph state and `previous_winners`.
Diagnostic traces must distinguish autonomous rollout state from the actual
observation state and must never feed oracle/teacher references into either.
"""
    write_text(report / "FIG9_RUNNER_RESPONSIBILITY_MAP.md", fig9)


def build_credibility_reports(root: Path, report: Path) -> None:
    common_fields = [
        "artifact_or_claim", "classification", "protocol_status", "mechanism_status",
        "numerical_status", "known_gap", "evidence", "allowed_claim",
    ]
    fig7 = [
        {
            "artifact_or_claim": "Corrected repeated sequence trials",
            "classification": "B",
            "protocol_status": "Substantially aligned with paper/reference [58]",
            "mechanism_status": "Event-driven approximation with corrected burst/winner separation",
            "numerical_status": "0.938+/-0.114; 0.955+/-0.056; 0.965+/-0.022 final accuracy",
            "known_gap": "Slower and less reliable convergence than paper curves near 1.0",
            "evidence": "RESULTS_STATUS.md; results/fig7_strict_corrected",
            "allowed_claim": "Protocol-aligned partial numerical reproduction",
        },
        {
            "artifact_or_claim": "Locally retrained HTM/TDNN/LSTM/ELM",
            "classification": "C",
            "protocol_status": "Nonpaper local comparison",
            "mechanism_status": "Different implementations/training",
            "numerical_status": "Not author baseline values",
            "known_gap": "Cannot establish paper baseline reproduction",
            "evidence": "experiments/diagnostics/fig7_nonpaper_baselines.py",
            "allowed_claim": "Diagnostic comparison only",
        },
    ]
    fig8 = [
        {
            "artifact_or_claim": "CBT neural autonomous retrieval 1000",
            "classification": "B",
            "protocol_status": "Cue 6/retrieve 4 and raw cell propagation aligned",
            "mechanism_status": "One-layer continuous/event approximation",
            "numerical_status": "Mean Levenshtein 3.981 versus paper about 0.2 at 10,000 symbols",
            "known_gap": "Large absolute mismatch and smaller local scale",
            "evidence": "FIG8_NEURAL_RETRIEVAL.md; RESULTS_STATUS.md",
            "allowed_claim": "Protocol correction completed; numerical reproduction failed",
        },
        {
            "artifact_or_claim": "Resource sweep",
            "classification": "C",
            "protocol_status": "Controlled local diagnostic",
            "mechanism_status": "Same local model with varied capacity",
            "numerical_status": "Qualitative resource trend only",
            "known_gap": "Absolute errors exceed paper",
            "evidence": "results/fig8_strict_fixedset_resource_sweep",
            "allowed_claim": "Nonpaper capacity diagnostic",
        },
        {
            "artifact_or_claim": "Historical decoded-symbol proximal replay",
            "classification": "D",
            "protocol_status": "Violates autonomous contextual retrieval",
            "mechanism_status": "Decoded word re-entered as proximal input",
            "numerical_status": "Historical only",
            "known_gap": "Burst contamination and wrong state trajectory",
            "evidence": "FIG8_NEURAL_RETRIEVAL.md",
            "allowed_claim": "Invalid historical baseline/ablation",
        },
        {
            "artifact_or_claim": "Temporal confirmation 20-sentence diagnostic",
            "classification": "C",
            "protocol_status": "Nonpaper response_scale=1.0 diagnostic",
            "mechanism_status": "Read-only semantic instrumentation around learning",
            "numerical_status": "Current 0.0 vs unique identity 0.45 on 20 sentences",
            "known_gap": "Small diagnostic fixture; not strict default",
            "evidence": "results/temporal_confirmation_semantics_20260814_phaseC_20_final_v2",
            "allowed_claim": "Local mechanism evidence only",
        },
    ]
    fig9 = [
        {
            "artifact_or_claim": "Strict raw L_match=4 250 original",
            "classification": "B",
            "protocol_status": "Paper-constrained raw rollout; dataset candidate not proven author asset",
            "mechanism_status": "Strict one-layer continuous approximation",
            "numerical_status": "MAPE 0.505450; coverage 1.0; paper about 0.10",
            "known_gap": "Large numerical gap; 2000/full-year incomplete",
            "evidence": "results/fig9_diagnostics/lmatch_stack_2x2_20260810/250/STRICT_RAW_L4",
            "allowed_claim": "Strict 250-record partial attempt, not full reproduction",
        },
        {
            "artifact_or_claim": "Real L_match=2 ablation",
            "classification": "C",
            "protocol_status": "Nonpaper diagnostic; strict default unchanged",
            "mechanism_status": "Real matching semantics changed intentionally",
            "numerical_status": "MAPE 0.590201 on 250",
            "known_gap": "Not paper L_match=4",
            "evidence": "results/fig9_diagnostics/lmatch_stack_2x2_20260810/250/STRICT_RAW_L2",
            "allowed_claim": "Ablation only",
        },
        {
            "artifact_or_claim": "Competitive inhibition / selector / oracle diagnostics",
            "classification": "C",
            "protocol_status": "Explicit nonpaper modes",
            "mechanism_status": "Ranking, competition, or GT-labelled read-only analysis",
            "numerical_status": "Best historical diagnostic MAPE around 0.408656",
            "known_gap": "Cannot replace strict default or establish paper result",
            "evidence": "experiments/diagnostics; FIG9_* diagnostic reports",
            "allowed_claim": "Mechanism diagnosis only",
        },
        {
            "artifact_or_claim": "Historical growth-only/proximal-replay 0.098",
            "classification": "D",
            "protocol_status": "Violates corrected strict retrieval/learning protocol",
            "mechanism_status": "Compensated historical path",
            "numerical_status": "Numerically close but scientifically invalid for strict claim",
            "known_gap": "Uses proximal replay/growth-only behavior",
            "evidence": "RESULTS_STATUS.md; results/fig9_historical_compensated",
            "allowed_claim": "Invalid historical result",
        },
        {
            "artifact_or_claim": "ETTh1 and Weather transfer",
            "classification": "C",
            "protocol_status": "Application transfer, not Zhang Fig.9 protocol",
            "mechanism_status": "Adapted multivariate/local baselines",
            "numerical_status": "Dataset-specific transfer metrics",
            "known_gap": "Does not validate paper reproduction",
            "evidence": "ETTH1_FIG9_TRANSFER.md; WEATHER_FIG9_TRANSFER.md",
            "allowed_claim": "Application/transfer evidence only",
        },
    ]
    write_csv(report / "FIG7_CREDIBILITY_MATRIX.csv", common_fields, fig7)
    write_csv(report / "FIG8_CREDIBILITY_MATRIX.csv", common_fields, fig8)
    write_csv(report / "FIG9_CREDIBILITY_MATRIX.csv", common_fields, fig9)

    conflicts = """# Credibility Conflicts

1. `RESULTS_STATUS.md` ends with "passes the full unit-test suite", while the
   current Windows single-process discovery can terminate with native exit
   `0xC0000005`. Isolated module runs pass, but that is a different claim.
2. Old Fig.8 result names containing `strict` may use decoded-symbol proximal
   replay. Only the audited neural-retrieval artifacts support the corrected
   contextual-retrieval protocol.
3. Historical Fig.9 values near `0.098` used compensated/growth-only/proximal
   behavior and cannot be cited as a strict reproduction result.
4. The local NYC taxi stream structurally matches the cited reference but the
   exact Zhang preprocessing asset and aggregation are unpublished.
5. Fig.9 strict imports nonpaper diagnostic modules at startup. Defaults are
   off, but the dependency weakens architectural separation and runtime fault
   isolation.
6. ETTh1 and Weather experiments evaluate transfer behavior. They do not raise
   the credibility grade of Fig.9 paper reproduction.

No report should say "complete reproduction" while these conflicts remain.
"""
    write_text(report / "CREDIBILITY_CONFLICTS.md", conflicts)

    draft = """# Reproduction Credibility Draft

## Scale

- `A`: protocol, mechanism, data provenance, and numerical result are strongly verified.
- `B`: protocol is substantially aligned, but numerical or unpublished-detail gaps remain.
- `C`: nonpaper diagnostic, approximation, ablation, or application transfer.
- `D`: historical output known to violate the corrected protocol or claim boundary.

## Fig.7: B

The corrected stream and sequence protocol are substantially aligned. Final
ten-trial accuracy is high, but convergence is slower and less reliable than
the paper. This supports a partial protocol-aligned reproduction, not complete
numerical agreement. Locally retrained reference models remain grade C.

## Fig.8: B for corrected CBT protocol, C/D for auxiliaries

Autonomous neural retrieval now propagates actual predictive cells and does
not replay decoded words. The 1000-sentence distance remains 3.981, far from
the paper's approximate 0.2 at 10,000 symbols. Capacity, response-scale,
competition, and temporal-confirmation runs are grade C diagnostics. The old
proximal-replay claim is grade D.

## Fig.9: B for strict 250, C/D elsewhere

The strict 250-record L_match=4 result uses raw autonomous propagation and has
MAPE 0.505450 with complete coverage. The paper is near 0.10, the exact author
taxi asset is not verified, and 2000/full-year strict runs are incomplete.
Competition, oracle, selector, real L_match=2, ETTh1, and Weather work are grade
C. Historical compensated 0.098 output is grade D.

## Claim boundary

ETTh1 and Weather demonstrate application-transfer behavior only. They cannot
be used as evidence that Zhang Fig.9 has been reproduced. The strongest
defensible project statement is that key protocols and mechanisms have been
implemented and audited, while important numerical gaps and unpublished paper
details remain.
"""
    write_text(report / "REPRODUCTION_CREDIBILITY_DRAFT.md", draft)


def build_regression_anchor_manifest(root: Path, report: Path) -> None:
    anchors = [
        {
            "artifact": "results/fig9_diagnostics/lmatch_stack_2x2_20260810/250/STRICT_RAW_L4",
            "family": "Fig9", "classification": "B", "commit": "unknown in artifact",
            "dataset_hash_available": True, "protocol_available": True, "seed_available": True,
            "fingerprint_available": True, "metric": "MAPE=0.5054500721931258",
            "usable_as_anchor": True, "reason": "Complete 45/45 predictions and model/RNG fingerprints",
        },
        {
            "artifact": "results/fig9_diagnostics/lmatch_stack_2x2_20260810/250/STRICT_RAW_L2",
            "family": "Fig9", "classification": "C", "commit": "unknown in artifact",
            "dataset_hash_available": True, "protocol_available": True, "seed_available": True,
            "fingerprint_available": True, "metric": "MAPE=0.5902013451316054",
            "usable_as_anchor": True, "reason": "Explicit nonpaper L_match ablation anchor",
        },
        {
            "artifact": "results/temporal_confirmation_semantics_20260814_phaseC_20_final_v2",
            "family": "Fig8", "classification": "C", "commit": SNAPSHOT_COMMIT,
            "dataset_hash_available": False, "protocol_available": True, "seed_available": True,
            "fingerprint_available": True, "metric": "current=0.0; unique identity=0.45",
            "usable_as_anchor": True, "reason": "Trace on/off fingerprints match; diagnostic fixture only",
        },
        {
            "artifact": "results/fig8_strict_neural_retrieval_1000.csv",
            "family": "Fig8", "classification": "B", "commit": "not embedded",
            "dataset_hash_available": False, "protocol_available": False, "seed_available": False,
            "fingerprint_available": False, "metric": "Mean distance=3.981",
            "usable_as_anchor": False, "reason": "Useful scientific result, insufficient exact-state provenance",
        },
        {
            "artifact": "results/fig7_strict_corrected",
            "family": "Fig7", "classification": "B", "commit": "not embedded",
            "dataset_hash_available": True, "protocol_available": False, "seed_available": True,
            "fingerprint_available": False, "metric": "0.938/0.955/0.965",
            "usable_as_anchor": False, "reason": "Use as major-stage metric gate, not exact state baseline",
        },
    ]
    write_csv(report / "EXISTING_REGRESSION_ANCHORS.csv", list(anchors[0]), anchors)


def build_comment_audit(report: Path, python_audit: dict[str, object]) -> None:
    parsed = python_audit["parsed"]
    assert isinstance(parsed, dict)
    low_doc = sorted(
        (
            item
            for item in parsed.values()
            if item.definitions >= 10 and item.documented_definitions / item.definitions < 0.2
        ),
        key=lambda item: (-item.definitions, item.path),
    )[:25]
    lines = [
        "# Comment and Docstring Audit",
        "",
        "This is a read-only inventory. No scientific source comments were rewritten.",
        "",
        "## Priorities",
        "",
        "Retain and strengthen comments that explain paper provenance, state mutation, RNG",
        "sensitivity, floating-point ordering, checkpoint constraints, and nonpaper diagnostic",
        "boundaries. Remove narration and stale performance claims only in the phase that",
        "touches the associated code, so comment edits remain reviewable beside behavior gates.",
        "",
        "## Large files with low definition-docstring coverage",
        "",
        "| File | Definitions | Documented | Module docstring | Comment lines |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for item in low_doc:
        lines.append(
            f"| `{item.path}` | {item.definitions} | {item.documented_definitions} | "
            f"{item.module_docstring} | {item.comment_lines} |"
        )
    lines.extend(
        [
            "",
            "## Misleading-claim risk",
            "",
            "Repository-level reports that say all tests pass or imply complete numerical",
            "reproduction require correction independently of code comments. Diagnostic and",
            "oracle modules should carry an explicit nonpaper marker near their module",
            "docstring and CLI definition in a later documentation-only phase.",
        ]
    )
    write_text(report / "COMMENT_DOCSTRING_AUDIT.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/refactor/phase01_20260814_233129"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    report = args.report if args.report.is_absolute() else root / args.report
    report.mkdir(parents=True, exist_ok=True)

    python_audit = build_python_inventory(root, report)
    dataset_hashes = build_dataset_manifest(root, report)
    artifact_counts = build_artifact_manifest(root, report)
    results_count = build_results_manifest(root, report)
    diagnostics_count = build_diagnostics_manifest(root, report, python_audit)
    cli_count = build_cli_inventory(root, report)
    build_responsibility_maps(root, report)
    build_credibility_reports(root, report)
    build_regression_anchor_manifest(root, report)
    build_comment_audit(report, python_audit)

    print(
        json.dumps(
            {
                "python_files": python_audit["count"],
                "dataset_hashes": dataset_hashes,
                "artifact_counts": artifact_counts,
                "result_directories": results_count,
                "diagnostic_modules": diagnostics_count,
                "cli_options": cli_count,
                "core_outward_imports": python_audit["core_outward"],
                "strict_diagnostic_imports": python_audit["strict_diagnostic_edges"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
