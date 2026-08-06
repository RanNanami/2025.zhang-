"""Run one native process with durable merged logging and failure metadata."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def _progress_index(progress_path: Path) -> int | None:
    if progress_path.exists():
        lines = progress_path.read_text(encoding="utf-8").splitlines()
        if lines:
            try:
                return int(json.loads(lines[-1])["current_index"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
    return None


def _checkpoint_metadata_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(checkpoint_path.stem + ".metadata.json")


def _checkpoint_index(checkpoint_path: Path) -> int | None:
    """Read only the atomic metadata; never unpickle a child model here."""

    metadata_path = _checkpoint_metadata_path(checkpoint_path)
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            if payload.get("atomic_complete") and payload.get("checkpoint_sha256") == digest:
                return int(payload["next_index"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return None


def _native_phase(native_log: Path) -> tuple[int | None, str | None]:
    last_index: int | None = None
    last_phase: str | None = None
    if not native_log.exists():
        return last_index, last_phase
    try:
        for line in native_log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("FIG9_NATIVE_PHASE "):
                payload = json.loads(line.split(" ", 1)[1])
                last_index = int(payload["current_index"])
                last_phase = str(payload["phase"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return last_index, last_phase


def _native_prune_phase(native_log: Path) -> tuple[int | None, str | None]:
    last_index: int | None = None
    last_phase: str | None = None
    if not native_log.exists():
        return last_index, last_phase
    try:
        for line in native_log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("FIG9_NATIVE_PRUNE "):
                payload = json.loads(line.split(" ", 1)[1])
                last_index = int(payload["current_index"])
                last_phase = str(payload["phase"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return last_index, last_phase


def _exception_text(lines: deque[str], exit_code: int) -> str:
    content = "".join(lines)
    marker = content.rfind("Traceback (most recent call last):")
    if marker >= 0:
        return content[marker:].strip()
    return f"Native process terminated with exit code {exit_code}; no Python traceback was emitted."


def _console_write(text: str) -> None:
    """Echo diagnostics without letting a legacy Windows code page kill the runner."""

    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        encoded = text.encode(sys.stdout.encoding or "utf-8", errors="replace")
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is not None:
            buffer.write(encoded)
        else:
            sys.stdout.write(text.encode("utf-8", errors="replace").decode("utf-8"))
    sys.stdout.flush()


def run(
    command: list[str],
    *,
    log: Path | None = None,
    result: Path,
    failure: Path,
    checkpoint: Path,
    procdump: Path | None = None,
    dump_dir: Path | None = None,
    stdout_log: Path | None = None,
    stderr_log: Path | None = None,
    combined_log: Path | None = None,
) -> int:
    started = time.perf_counter()
    combined_log = combined_log or log or Path("run.log")
    stdout_log = stdout_log or combined_log
    stderr_log = stderr_log or combined_log
    for path in {stdout_log, stderr_log, combined_log, result, failure, checkpoint}:
        path.parent.mkdir(parents=True, exist_ok=True)
    combined_log.touch(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    launch_dir = result.parent
    (launch_dir / "command.json").write_text(
        json.dumps(
            {
                "command": command,
                "working_directory": os.getcwd(),
                "python_args": command[1:],
                "environment": {
                    key: env[key]
                    for key in sorted(env)
                    if key
                    in {
                        "PYTHONFAULTHANDLER",
                        "PYTHONUNBUFFERED",
                        "PYTHONPATH",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    }
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    tail: deque[str] = deque(maxlen=300)
    exit_code = -1
    dump_process: subprocess.Popen[str] | None = None
    try:
        with (
            stdout_log.open("a", encoding="utf-8", buffering=1) as stdout_handle,
            stderr_log.open("a", encoding="utf-8", buffering=1) as stderr_handle,
            combined_log.open("a", encoding="utf-8", buffering=1) as combined_handle,
        ):
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            if procdump is not None:
                assert dump_dir is not None
                dump_dir.mkdir(parents=True, exist_ok=True)
                dump_process = subprocess.Popen(
                    [
                        str(procdump),
                        "-accepteula",
                        "-e",
                        "1",
                        "-ma",
                        str(process.pid),
                        str(dump_dir),
                    ],
                    stdout=combined_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
            assert process.stdout is not None and process.stderr is not None
            write_lock = threading.Lock()

            combined_path = combined_log.resolve()

            def forward(stream, destination, destination_path: Path, label: str) -> None:
                for line in stream:
                    if label == "stderr":
                        tail.append(line)
                    with write_lock:
                        destination.write(line)
                        destination.flush()
                        if destination_path.resolve() != combined_path:
                            combined_handle.write(line)
                            combined_handle.flush()
                    # Keep child diagnostics visible without PowerShell's
                    # native stderr error-record conversion.
                    _console_write(line)

            threads = [
                threading.Thread(
                    target=forward,
                    args=(process.stdout, stdout_handle, stdout_log, "stdout"),
                    daemon=True,
                ),
                threading.Thread(
                    target=forward,
                    args=(process.stderr, stderr_handle, stderr_log, "stderr"),
                    daemon=True,
                ),
            ]
            for thread in threads:
                thread.start()
            exit_code = process.wait()
            for thread in threads:
                thread.join()
            process.stdout.close()
            process.stderr.close()
    except BaseException as exc:
        tail.append(f"Supervisor exception: {type(exc).__name__}: {exc}\n")
        exit_code = -1
    finally:
        if dump_process is not None and dump_process.poll() is None:
            dump_process.terminate()
            try:
                dump_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                dump_process.kill()

    elapsed = time.perf_counter() - started
    progress_index = _progress_index(checkpoint.parent / "progress.jsonl")
    checkpoint_index = _checkpoint_index(checkpoint)
    native_index, native_phase = _native_phase(
        checkpoint.parent / "native_crash_faulthandler.log"
    )
    prune_index, prune_phase = _native_prune_phase(
        checkpoint.parent / "native_crash_faulthandler.log"
    )
    payload = {
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "command": command,
        "checkpoint_path": str(checkpoint.resolve()),
        # Progress is observational only.  The recoverable completion boundary
        # is the last atomically published checkpoint metadata record.
        "last_completed_index": (
            checkpoint_index if checkpoint_index is not None else progress_index
        ),
        "last_progress_index": progress_index,
        "checkpoint_next_index": checkpoint_index,
        "last_native_phase_index": native_index,
        "last_native_phase": native_phase,
        "last_native_prune_index": prune_index,
        "last_native_prune_phase": prune_phase,
        "native_crash_log_path": str(
            (checkpoint.parent / "native_crash_faulthandler.log").resolve()
        ),
        "dump_directory": str(dump_dir.resolve()) if dump_dir else None,
    }
    result.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if exit_code != 0:
        payload["exception"] = _exception_text(tail, exit_code)
        failure.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _console_write("\n===== child stderr tail (last 300 lines) =====\n")
        _console_write("".join(tail))
        return 1
    if failure.exists():
        failure.unlink()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--failure-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--procdump", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--stdout-log", type=Path)
    parser.add_argument("--stderr-log", type=Path)
    parser.add_argument("--combined-log", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a command is required after --")
    if bool(args.procdump) != bool(args.dump_dir):
        parser.error("--procdump and --dump-dir must be supplied together")
    raise SystemExit(
        run(
            command,
            log=args.log,
            result=args.result_json,
            failure=args.failure_json,
            checkpoint=args.checkpoint,
            procdump=args.procdump,
            dump_dir=args.dump_dir,
            stdout_log=args.stdout_log,
            stderr_log=args.stderr_log,
            combined_log=args.combined_log,
        )
    )


if __name__ == "__main__":
    main()
