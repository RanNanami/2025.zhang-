"""Print machine-readable resume metadata for a strict Fig.9 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.diagnostics.fig9_branch_provenance import BranchProvenanceRegistry
from experiments.fig9_strict_reproduction import (
    branch_checkpoint_sidecar_path,
    load_strict_checkpoint,
    model_long_term_fingerprint,
    model_rng_fingerprint,
)


def inspect_checkpoint(path: Path) -> dict[str, object]:
    payload = load_strict_checkpoint(path)
    model = payload["model"]
    provenance = payload.get("branch_provenance")
    source = "embedded_checkpoint"
    if not isinstance(provenance, dict):
        sidecar = branch_checkpoint_sidecar_path(path)
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        source = "sidecar"
    registry = BranchProvenanceRegistry.from_checkpoint_payload(model, provenance)
    return {
        "checkpoint_path": str(path.resolve()),
        "L_match": int(payload.get("L_match", payload["config"].l_match)),
        "limit": int(payload["limit"]),
        "next_index": int(payload["next_index"]),
        "model_fingerprint": model_long_term_fingerprint(model),
        "rng_fingerprint": model_rng_fingerprint(model),
        "provenance_source": source,
        "provenance_binding": registry.binding_summary(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_checkpoint(args.checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
