#!/usr/bin/env python3
"""Repository-local launcher for NEGF TDP PS-SCBA."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


SOFTWARE = "NEGF TDP PS-SCBA"
ROOT = Path(__file__).resolve().parent


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, check=False,
                                capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _config_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dirty() -> bool:
    return bool(_git("status", "--porcelain", "--untracked-files=normal"))


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log, self.lock = stream, log, threading.Lock()

    def write(self, value):
        with self.lock:
            self.stream.write(value)
            self.log.write(value)
            # Do not call ``self.flush()`` here: it acquires the same lock and
            # would deadlock on the first launcher line.
            self.stream.flush()
            self.log.flush()
        return len(value)

    def flush(self):
        with self.lock:
            self.stream.flush()
            self.log.flush()

    def isatty(self):
        return False


@contextmanager
def _launch_context(command: str, output: Path, *, acquire_lock: bool = True):
    output.mkdir(parents=True, exist_ok=True)
    (output / ".matplotlib").mkdir(exist_ok=True)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))

    import fcntl
    lock_path = ROOT / "runs" / ".negf-tdp-ps-scba.launch.lock"
    lock_handle = None
    if acquire_lock:
        lock_path.parent.mkdir(exist_ok=True)
        lock_handle = open(lock_path, "w", encoding="utf-8")
    try:
        if lock_handle is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another launcher holds {lock_path}.") from exc
        with open(output / "launcher.log", "a", encoding="utf-8") as log:
            stdout, stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
            with redirect_stdout(stdout), redirect_stderr(stderr):
                print("=" * 72)
                print(SOFTWARE)
                print(f"mode={command}  started={datetime.now(timezone.utc).isoformat()}")
                print(f"repository={ROOT}")
                print(f"output={output}")
                print(f"pid={os.getpid()}")
                print(f"git={_git('describe', '--tags', '--always', '--dirty') or 'unavailable'}")
                try:
                    yield
                except BaseException as exc:
                    print(
                        f"failed={datetime.now(timezone.utc).isoformat()} "
                        f"type={type(exc).__name__} message={exc}"
                    )
                    raise
                finally:
                    print(f"finished={datetime.now(timezone.utc).isoformat()}")
    finally:
        if lock_handle is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                lock_handle.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runner.py", description=SOFTWARE)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="schema and synthetic checks")
    check.add_argument("config")
    test = sub.add_parser("test", help="run the unit test suite")
    test.add_argument("pytest_args", nargs=argparse.REMAINDER)
    for name in ("run", "profile", "validate", "sweep", "compare-preparation"):
        command = sub.add_parser(name)
        command.add_argument("config")
        if name == "validate":
            command.add_argument("--session")
        if name == "sweep":
            command.add_argument("--directory")
    resume = sub.add_parser("resume")
    resume.add_argument("run_directory")
    resume.add_argument("--validation-session")
    plot = sub.add_parser("plot")
    plot.add_argument("run_directory")
    status = sub.add_parser("status")
    status.add_argument("path")
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("root")
    return parser


def _output_for(args) -> Path:
    if args.command == "resume":
        return Path(args.run_directory).resolve()
    return ROOT / "runs" / f"negf-tdp-ps-scba_{args.command}_{_timestamp()}"


def _run_cli(argv: list[str]) -> int:
    from psscba.frontend.cli import main as cli_main
    return int(cli_main(argv) or 0)


def _run_logged_subprocess(command: list[str]) -> None:
    """Stream a child process through the launcher's branded log.

    Redirecting ``sys.stdout`` does not redirect file descriptor 1 used by a
    child process.  Without this small pump, pytest output disappears into
    ``nohup``'s ``/dev/null`` even though the launcher itself is logging.
    """
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def _write_marker(config: str, session: Path) -> None:
    marker = ROOT / "runs" / ".validation-ready.json"
    campaign_path = session / "campaign.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8")) if campaign_path.exists() else {}
    payload = {
        "software": SOFTWARE,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_description": _git("describe", "--tags", "--always", "--dirty"),
        "git_dirty": _dirty(),
        "config": str(Path(config).resolve()),
        "config_sha256": _config_hash(config),
        "validation_directory": str(session.resolve()),
        "validation_status": campaign.get("status", "unknown"),
        "diagnostic_failures": campaign.get("failure_summary", {}).get("diagnostic_count", 0),
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, marker)


def _require_validated(config: str) -> None:
    marker = ROOT / "runs" / ".validation-ready.json"
    if not marker.exists():
        raise RuntimeError("Sweep refused: complete validation is required first.")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("validation_status") not in ("complete", "complete_with_diagnostic_failures"):
        raise RuntimeError("Sweep refused: validation did not pass its production gates.")
    if payload.get("git_dirty") or _dirty():
        raise RuntimeError("Sweep refused: the validated Git tree must be clean.")
    if payload.get("git_commit") != _git("rev-parse", "HEAD"):
        raise RuntimeError("Sweep refused: Git commit differs from validation.")
    if payload.get("config_sha256") != _config_hash(config):
        raise RuntimeError("Sweep refused: configuration differs from validation.")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "test":
        return subprocess.run([sys.executable, "-m", "pytest", *args.pytest_args], cwd=ROOT).returncode
    # ``check`` is deliberately lock-free.  It only parses the configuration
    # and runs lightweight synthetic checks, so it must remain usable while a
    # long-running solver (possibly started on another host sharing the run
    # filesystem) owns the campaign lock.
    if args.command == "check":
        # Checks create their own run record but do not contend with a solver.
        # This makes a check safe to launch alongside an existing campaign and
        # keeps its nohup output under runs/ like every other command.
        output = _output_for(args)
        with _launch_context(args.command, output, acquire_lock=False):
            return _run_cli(["validate", args.config])
    if args.command == "status":
        return _run_cli(["status", args.path])
    if args.command == "plot":
        return _run_cli(["plot", args.run_directory])
    if args.command == "aggregate":
        return _run_cli(["aggregate", args.root])

    output = _output_for(args)
    with _launch_context(args.command, output):
        if args.command == "profile":
            return _run_cli(["profile", args.config])
        if args.command == "validate":
            # A failed rerun must not leave an older readiness marker usable.
            # The marker is generated state, so remove only this exact file.
            marker = ROOT / "runs" / ".validation-ready.json"
            try:
                marker.unlink()
            except FileNotFoundError:
                pass
            _run_logged_subprocess([sys.executable, "-m", "pytest", "-q"])
            _run_cli(["profile", args.config])
            session = Path(args.session).resolve() if args.session else output
            _run_cli(["validate", args.config, "--full", "--session", str(session)])
            _write_marker(args.config, session)
            print(f"validation_marker={ROOT / 'runs' / '.validation-ready.json'}")
            return 0
        if args.command == "sweep":
            _require_validated(args.config)
            command = ["sweep", args.config, "--directory", args.directory or str(output)]
            return _run_cli(command)
        if args.command == "run":
            return _run_cli(["run", args.config])
        if args.command == "compare-preparation":
            return _run_cli(["compare-preparation", args.config])
        if args.command == "resume":
            command = ["resume", args.run_directory]
            if args.validation_session:
                command += ["--validation-session", args.validation_session]
            return _run_cli(command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except RuntimeError as exc:
        # Keep launcher failures readable in a terminal and in nohup logs;
        # users should not need to parse a Python traceback for a held lock or
        # a validation precondition.
        print(f"{SOFTWARE}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
