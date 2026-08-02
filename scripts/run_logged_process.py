"""Run one native process with durable merged logging and failure metadata."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
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


def _checkpoint_index(checkpoint_path: Path) -> int | None:
    if checkpoint_path.exists():
        try:
            from experiments.fig9_strict_reproduction import load_strict_checkpoint

            return int(load_strict_checkpoint(checkpoint_path)["next_index"])
        except Exception:
            pass
    return None


def _exception_text(lines: deque[str], exit_code: int) -> str:
    content = "".join(lines)
    marker = content.rfind("Traceback (most recent call last):")
    if marker >= 0:
        return content[marker:].strip()
    return f"Native process terminated with exit code {exit_code}; no Python traceback was emitted."


def run(
    command: list[str],
    *,
    log: Path,
    result: Path,
    failure: Path,
    checkpoint: Path,
    procdump: Path | None = None,
    dump_dir: Path | None = None,
) -> int:
    started = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    log.touch(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    tail: deque[str] = deque(maxlen=250)
    exit_code = -1
    dump_process: subprocess.Popen[str] | None = None
    try:
        with log.open("a", encoding="utf-8", buffering=1) as handle:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
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
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )
            assert process.stdout is not None
            for line in process.stdout:
                tail.append(line)
                handle.write(line)
                handle.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            exit_code = process.wait()
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
    payload = {
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "command": command,
        "checkpoint_path": str(checkpoint.resolve()),
        # A successful child atomically writes its final checkpoint after the
        # last progress interval. On failure, progress is the best record of
        # work completed after the most recent recoverable checkpoint.
        "last_completed_index": (
            checkpoint_index
            if exit_code == 0 and checkpoint_index is not None
            else progress_index if progress_index is not None else checkpoint_index
        ),
        "last_progress_index": progress_index,
        "checkpoint_next_index": checkpoint_index,
        "native_crash_log_path": str(
            (checkpoint.parent / "native_crash_faulthandler.log").resolve()
        ),
        "dump_directory": str(dump_dir.resolve()) if dump_dir else None,
    }
    result.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if exit_code != 0:
        payload["exception"] = _exception_text(tail, exit_code)
        failure.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 1
    if failure.exists():
        failure.unlink()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--failure-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--procdump", type=Path)
    parser.add_argument("--dump-dir", type=Path)
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
        )
    )


if __name__ == "__main__":
    main()
