from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import io
import json
import os
from pathlib import Path
import resource
import socket
import threading
import time
from typing import Any, Callable, Mapping

from psscba.frontend.configuration import RunConfig
from psscba.frontend.storage import atomic_json, atomic_text
from psscba.frontend.profiling import estimate_case_memory_bytes
from psscba.backend.model.units import time_to_ps
from psscba.types import OuterIteration


ProgressCallback = Callable[[str, Mapping[str, Any]], None]


PHASES: tuple[tuple[str, str], ...] = (
    ("initialization", "Initialization / stationary kernel"),
    ("pole_basis_reuse", "Validated AAA pole-basis reuse"),
    ("mpm_fit", "MPM rational-kernel fit"),
    ("aaa_fit", "Fresh causal AAA fit"),
    ("endpoint_poles", "Same-kernel physical endpoint poles"),
    ("square_cache_build", "Square-pulse residue cache"),
    ("square_residue_validation", "Square-residue validation"),
    ("amplitude_a", "Retarded lead amplitude A"),
    ("amplitude_c", "Interaction amplitude C"),
    ("amplitude_source_grid", "A/C source-grid refinement"),
    ("history_b", "Cumulative lead history B"),
    ("history_d", "Cumulative interaction history D"),
    ("psi", "Mixed lesser transform Psi"),
    ("retarded_reconstruction", "Retarded two-time reconstruction"),
    ("lesser_sources", "Generic lesser-source factorization"),
    ("streaming_projection", "Uniform stationary projection"),
    ("dense_validation", "Dense projector validation"),
    ("scba_candidate", "Projected Hartree-Fock candidate"),
    ("mixing_checkpoint", "Kernel mixing and checkpoint"),
    ("discarded_sector_diagnostics", "Discarded-sector diagnostics"),
    ("currents", "Physical currents and continuity"),
    ("qa_rendering", "QA and publication rendering"),
)
PHASE_INDEX = {name: index + 1 for index, (name, _) in enumerate(PHASES)}
PHASE_LABEL = dict(PHASES)


def _duration(seconds: float | None) -> str:
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0.0:
        return "--:--:--"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _gib(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 1024**3:.1f} GiB"


def _scalar(value: float | None) -> str:
    return "pending" if value is None else f"{float(value):.3e}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resource_sample() -> dict[str, int | None]:
    rss = None
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        rss = pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        pass
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if os.uname().sysname != "Darwin":
        peak *= 1024
    available = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    return {"rss_bytes": rss, "peak_rss_bytes": peak, "available_memory_bytes": available}


def write_convergence_csv(path: Path, history: tuple[OuterIteration, ...] | list[OuterIteration]) -> None:
    fields = list(asdict(OuterIteration(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for item in history:
        writer.writerow(asdict(item))
    atomic_text(path, output.getvalue())


def read_tracking_history(path: str | Path) -> list[dict[str, Any]]:
    """Read complete JSONL records, ignoring a torn final line."""
    records: list[dict[str, Any]] = []
    target = Path(path)
    if not target.exists():
        return records
    for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def stale_heartbeat(progress_path: str | Path, *, now: float | None = None, factor: float = 2.5) -> bool:
    path = Path(progress_path)
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("state") not in ("running", "interrupted"):
        return False
    written = datetime.fromisoformat(payload["timestamp"]).timestamp()
    interval = float(payload.get("heartbeat_seconds", 60.0))
    return (time.time() if now is None else now) - written > factor * interval


class RunTracker:
    """Thread-safe, scalar-only live state for one run."""

    def __init__(
        self,
        run_dir: str | Path,
        config: RunConfig,
        history=(),
        *,
        clock=time.monotonic,
        mirror_stdout: bool = False,
    ):
        self.run_dir = Path(run_dir)
        self.config = config
        self.history = list(history)
        self.clock = clock
        prior_progress: dict[str, Any] = {}
        try:
            prior_progress = json.loads(
                (self.run_dir / "progress.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            pass
        prior_elapsed = float(prior_progress.get("elapsed_seconds") or 0.0)
        self._prior_peak_rss = max(
            int(prior_progress.get("peak_rss_bytes") or 0),
            max(
                (int(item.peak_rss_bytes) for item in self.history if item.peak_rss_bytes is not None),
                default=0,
            ),
        )
        self.started_wall = str(prior_progress.get("started_at") or utc_now())
        # A resumed tracker continues the persisted case wall-clock rather
        # than resetting elapsed time to zero.
        self.started = clock() - prior_elapsed
        self.phase_started = self.started
        self.phase = "initialization"
        self.outer_step: int | None = self.history[-1].iteration + 1 if self.history else 0
        self.batch_completed: int | None = None
        self.batch_total: int | None = None
        self.batch_unit: str | None = None
        self.message: str | None = None
        self.state = "running"
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned = False
        self._last_sample: dict[str, Any] = {}
        self.mirror_stdout = bool(mirror_stdout)
        self._last_live_snapshot = self.started
        self.last_current_nat: dict[str, float] = {}
        self.last_current_uA: dict[str, float] = {}
        self.observable_time_nat: float | None = None
        self.observable_time_ps: float | None = None
        self.pending_residual_retarded: float | None = None
        self.pending_residual_lesser: float | None = None
        self.stage_metrics: dict[str, Any] = {}

    def _emit(self, message: str) -> None:
        """Append one durable human-readable event to this job only."""
        line = f"{utc_now()}  {message}"
        path = self.run_dir / "solver.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        if self.mirror_stdout:
            print(line, flush=True)

    def log_message(self, message: str) -> None:
        """Write one explicit job-local informational line."""
        self._emit(message)

    def _append_event(self, kind: str, **details: Any) -> None:
        """Append one structured, flushed event for post-run stage accounting."""
        sample = resource_sample() if self.config.tracking.resource_sampling else {}
        payload = {
            "format": "psscba-progress",
            "timestamp": utc_now(),
            "elapsed_seconds": self.clock() - self.started,
            "kind": kind,
            "phase": self.phase,
            "outer_step": self.outer_step,
            **sample,
            **details,
        }
        path = self.run_dir / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _write_header(self) -> None:
        model = self.config.model
        protocol = self.config.protocol
        projection = self.config.projection
        numerics = self.config.numerics
        time_unit_ps = float(time_to_ps(1.0, model.gamma_eV))
        mode = "noninteracting_exact (g=0)" if model.coupling_meV == 0.0 else "self_consistent_SCBA"
        duration = (
            f" | pulse={protocol.duration:g} hbar/Gamma"
            if protocol.duration is not None else ""
        )
        self._emit("╭─ PS-SCBA · projected stationary self-consistency")
        self._emit(
            "│ CASE  "
            f"protocol={protocol.name}{duration} | W/Gamma={model.bandwidth:g} | "
            f"omega0/Gamma={model.phonon_energy:g} | g={model.coupling_meV:g} meV "
            f"({model.coupling:g} Gamma)"
        )
        self._emit(
            "│ GRID  "
            f"t=[{projection.time_min:g},{projection.time_max:g}] hbar/Gamma "
            f"({projection.time_min * time_unit_ps:g},{projection.time_max * time_unit_ps:g}) ps "
            f"dt={projection.time_step:g} hbar/Gamma "
            f"({projection.time_step * time_unit_ps:g} ps) | E=[{numerics.energy_min:g},"
            f"{numerics.energy_max:g}] Gamma dE={numerics.energy_step:g}"
        )
        self._emit(
            "│ SCF   "
            f"mode={mode} | loss=max(R_sigma^R,R_sigma^<) | target={numerics.outer_tolerance:.3e} | "
            f"mixing alpha={numerics.outer_mixing:g} | max_steps={numerics.outer_max_iter}"
        )
        self._emit(
            "│ CURR  production=pole/Filon A/C/Psi | dense two-time data is used only "
            "for PS-SCBA filtering/validation"
        )
        self._emit(
            "╰─ TRACK "
            f"live={self.config.tracking.live_refresh_seconds:g}s heartbeat={self.config.tracking.heartbeat_seconds:g}s | "
            "unfinished-step residuals always refer to the last completed SCF step"
        )

    @property
    def callback(self) -> ProgressCallback:
        return self.update

    def start(self) -> "RunTracker":
        if not self.config.tracking.enabled:
            return self
        self._write_header()
        self._append_event("run_started")
        self.snapshot(append_history=True)
        self._thread = threading.Thread(target=self._heartbeat_loop, name="psscba-heartbeat", daemon=True)
        self._thread.start()
        return self

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.config.tracking.heartbeat_seconds):
            self.snapshot(append_history=True, display=True)

    def update(self, phase: str, details: Mapping[str, Any] | None = None) -> None:
        details = {} if details is None else dict(details)
        with self._lock:
            changed = phase != self.phase
            if changed:
                self._append_event(
                    "phase_finished",
                    phase=self.phase,
                    duration_seconds=self.clock() - self.phase_started,
                )
                self.phase_started = self.clock()
            self.phase = phase
            self.outer_step = details.get("outer_step", self.outer_step)
            self.batch_completed = details.get("batch_completed")
            self.batch_total = details.get("batch_total")
            self.batch_unit = details.get("batch_unit")
            self.message = details.get("message")
            for key in (
                "reused", "validation", "fallback_reason", "terms", "origin_iteration",
                "cache_hit", "grid_points", "residue_method", "duration_seconds",
                "residue_validation", "max_terms", "residue_refit",
                "residue_refit_iteration", "sigma_scaled_error",
            ):
                if key in details:
                    self.stage_metrics[key] = details[key]
            if details.get("residual_retarded") is not None:
                self.pending_residual_retarded = float(details["residual_retarded"])
            if details.get("residual_lesser") is not None:
                self.pending_residual_lesser = float(details["residual_lesser"])
            lead = details.get("lead")
            if lead in ("L", "R"):
                if details.get("current_nat") is not None:
                    self.last_current_nat[lead] = float(details["current_nat"])
                if details.get("current_uA") is not None:
                    self.last_current_uA[lead] = float(details["current_uA"])
            for name in ("L", "R"):
                if details.get(f"current_{name}_nat") is not None:
                    self.last_current_nat[name] = float(details[f"current_{name}_nat"])
                if details.get(f"current_{name}_uA") is not None:
                    self.last_current_uA[name] = float(details[f"current_{name}_uA"])
            if details.get("observable_time_nat") is not None:
                self.observable_time_nat = float(details["observable_time_nat"])
                self.observable_time_ps = float(details.get(
                    "observable_time_ps",
                    time_to_ps(self.observable_time_nat, self.config.model.gamma_eV),
                ))
        if changed:
            self._append_event("phase_started", phase=phase, details=details)
            batch = ""
            if self.batch_completed is not None and self.batch_total is not None:
                batch = f" {self.batch_completed}/{self.batch_total} {self.batch_unit or 'items'}"
            stage = PHASE_INDEX.get(phase)
            stage_text = f"{stage:02d}/{len(PHASES):02d}" if stage is not None else "--/--"
            iteration = 1 + int(self.outer_step or 0)
            self._emit(
                f"▶ STAGE {stage_text} | SCF {iteration:04d}/{self.config.numerics.outer_max_iter:04d} "
                f"| {PHASE_LABEL.get(phase, phase)}{batch}"
            )
            self.snapshot(append_history=True)
        elif self.config.tracking.enabled:
            now = self.clock()
            if now - self._last_live_snapshot >= self.config.tracking.live_refresh_seconds:
                self._last_live_snapshot = now
                self.snapshot(display=True)

    def completed_step(self, record: OuterIteration, history) -> None:
        with self._lock:
            self.history = list(history)
            self.outer_step = record.iteration
            self.pending_residual_retarded = None
            self.pending_residual_lesser = None
        write_convergence_csv(self.run_dir / "convergence.csv", self.history)
        if self.config.plotting.enabled and (
            len(self.history) % self.config.tracking.convergence_refresh_steps == 0
        ):
            from psscba.backend.plotting import plot_outer_convergence

            plot_outer_convergence(
                self.config,
                tuple(self.history),
                self.run_dir / "plots" / "live" / "outer_convergence.svg",
            )
        loss = max(record.residual_retarded, record.residual_lesser)
        target = self.config.numerics.outer_tolerance
        exact_noninteracting = self.config.model.coupling_meV == 0.0
        verdict = "EXACT g=0" if exact_noninteracting else (
            "TOLERANCE MET" if loss < target else "continue"
        )
        self._emit(
            f"◆ SCF COMPLETE {record.iteration + 1:04d}/{self.config.numerics.outer_max_iter:04d} "
            f"| loss={loss:.6e} target={target:.3e} ratio={loss / target:.3e} "
            f"| R_R={record.residual_retarded:.6e} R_<={record.residual_lesser:.6e} "
            f"| alpha={self.config.numerics.outer_mixing:g} | {verdict}"
        )
        self._emit(
            f"  STATE | dPG_R={_scalar(record.projected_green_retarded_change)} "
            f"dPG_<={_scalar(record.projected_green_lesser_change)} "
            f"N={record.projected_occupation:.8f} "
            f"Hartree={record.sigma_hartree_installed if record.sigma_hartree_installed is not None else float('nan'):.6e}"
        )
        self._emit(
            f"  COST  | step={_duration(record.step_duration_seconds)} "
            f"cumulative={_duration(record.cumulative_duration_seconds)} "
            f"RSS={_gib(record.rss_bytes)} peak={_gib(record.peak_rss_bytes)}"
        )
        self._append_event(
            "scf_step_completed",
            iteration=record.iteration,
            loss=loss,
            residual_retarded=record.residual_retarded,
            residual_lesser=record.residual_lesser,
            step_duration_seconds=record.step_duration_seconds,
        )
        self.snapshot(append_history=True)

    def _payload(self) -> dict[str, Any]:
        sample = resource_sample() if self.config.tracking.resource_sampling else {}
        if sample.get("peak_rss_bytes") is not None:
            self._prior_peak_rss = max(
                self._prior_peak_rss, int(sample["peak_rss_bytes"])
            )
            sample["peak_rss_bytes"] = self._prior_peak_rss
        self._last_sample = sample
        estimate = estimate_case_memory_bytes(self.config)
        available = sample.get("available_memory_bytes")
        warning = bool(
            available is not None
            and available < self.config.tracking.memory_warning_factor * estimate
        )
        if warning and not self._warned:
            self._emit(
                "MEMORY WARNING | available host memory "
                f"{available / 1024**3:.1f} GiB is below "
                f"{self.config.tracking.memory_warning_factor:.2f} x the estimated "
                f"{estimate / 1024**3:.1f} GiB case peak; policy=warn, continuing."
            )
            self._warned = True
        latest = asdict(self.history[-1]) if self.history else None
        latest_loss = None if latest is None else max(
            float(latest["residual_retarded"]), float(latest["residual_lesser"])
        )
        losses = [
            max(float(item.residual_retarded), float(item.residual_lesser))
            for item in self.history
        ]
        best_loss = min(losses) if losses else None
        batch_fraction = None
        batch_eta = None
        if self.batch_completed is not None and self.batch_total:
            batch_fraction = max(0.0, min(1.0, self.batch_completed / self.batch_total))
            phase_elapsed = self.clock() - self.phase_started
            if batch_fraction > 0.0:
                batch_eta = phase_elapsed * (1.0 - batch_fraction) / batch_fraction
        phase_number = PHASE_INDEX.get(self.phase)
        return {
            "format": "psscba-event",
            "state": self.state,
            "phase": self.phase,
            "outer_step": self.outer_step,
            "batch_completed": self.batch_completed,
            "batch_total": self.batch_total,
            "batch_unit": self.batch_unit,
            "message": self.message,
            "timestamp": utc_now(),
            "started_at": self.started_wall,
            "elapsed_seconds": self.clock() - self.started,
            "phase_elapsed_seconds": self.clock() - self.phase_started,
            "heartbeat_seconds": self.config.tracking.heartbeat_seconds,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            **sample,
            "estimated_case_peak_bytes": estimate,
            "memory_warning": warning,
            "residuals_are_from_last_completed_step": True,
            "last_completed": latest,
            "last_completed_loss": latest_loss,
            "best_completed_loss": best_loss,
            "loss_definition": "max(residual_retarded, residual_lesser)",
            "loss_tolerance": self.config.numerics.outer_tolerance,
            "loss_to_tolerance_ratio": (
                None if latest_loss is None else latest_loss / self.config.numerics.outer_tolerance
            ),
            "mixing_rate": self.config.numerics.outer_mixing,
            "outer_max_iterations": self.config.numerics.outer_max_iter,
            "outer_iteration_display": 1 + int(self.outer_step or 0),
            "phase_number": phase_number,
            "phase_total": len(PHASES),
            "phase_label": PHASE_LABEL.get(self.phase, self.phase),
            "batch_fraction": batch_fraction,
            "batch_eta_seconds": batch_eta,
            "scba_mode": (
                "noninteracting_exact" if self.config.model.coupling_meV == 0.0
                else "self_consistent"
            ),
            "current_evaluator": "pole_filon_A_C_Psi",
            "residual_retarded": None if latest is None else latest.get("residual_retarded"),
            "residual_lesser": None if latest is None else latest.get("residual_lesser"),
            "pending_residual_retarded": self.pending_residual_retarded,
            "pending_residual_lesser": self.pending_residual_lesser,
            "projected_green_retarded_change": (
                None if latest is None else latest.get("projected_green_retarded_change")
            ),
            "projected_green_lesser_change": (
                None if latest is None else latest.get("projected_green_lesser_change")
            ),
            "projected_occupation": None if latest is None else latest.get("projected_occupation"),
            "sigma_hartree_installed": None if latest is None else latest.get("sigma_hartree_installed"),
            "sigma_hartree_candidate": None if latest is None else latest.get("sigma_hartree_candidate"),
            "mpm_tolerance": None if latest is None else latest.get("mpm_tolerance"),
            "step_duration_seconds": None if latest is None else latest.get("step_duration_seconds"),
            "cumulative_duration_seconds": None if latest is None else latest.get("cumulative_duration_seconds"),
            "observable_time_nat": self.observable_time_nat,
            "observable_time_ps": self.observable_time_ps,
            "current_L_nat": self.last_current_nat.get("L"),
            "current_R_nat": self.last_current_nat.get("R"),
            "current_L_uA": self.last_current_uA.get("L"),
            "current_R_uA": self.last_current_uA.get("R"),
            "stage_metrics": dict(self.stage_metrics),
            "pole_basis_reused": self.stage_metrics.get("reused"),
            "pole_basis_fallback_reason": self.stage_metrics.get("fallback_reason"),
            "pole_fit_terms": self.stage_metrics.get("terms"),
            "square_cache_hit": self.stage_metrics.get("cache_hit"),
            "square_cache_grid_points": self.stage_metrics.get("grid_points"),
            "square_residue_method": self.stage_metrics.get("residue_method"),
        }

    def _dashboard_line(self, payload: Mapping[str, Any]) -> str:
        completed = payload.get("batch_completed")
        total = payload.get("batch_total")
        if completed is not None and total:
            fraction = 100.0 * float(payload.get("batch_fraction") or 0.0)
            work = (
                f"{completed}/{total} {payload.get('batch_unit') or 'items'} "
                f"({fraction:5.1f}%) eta={_duration(payload.get('batch_eta_seconds'))}"
            )
        else:
            work = f"elapsed-stage={_duration(payload.get('phase_elapsed_seconds'))}"
        time_text = (
            "t=--"
            if payload.get("observable_time_ps") is None
            else f"t={float(payload['observable_time_ps']):.3f} ps"
        )
        current_text = (
            f"J_L={_scalar(payload.get('current_L_uA'))} µA "
            f"J_R={_scalar(payload.get('current_R_uA'))} µA"
        )
        loss = payload.get("last_completed_loss")
        if payload.get("scba_mode") == "noninteracting_exact":
            loss_text = "SCBA=exact(g=0); residual=N/A"
        else:
            loss_text = "loss(last)=pending" if loss is None else (
                f"loss(last)={loss:.3e} x{payload['loss_to_tolerance_ratio']:.2e} target"
            )
        return (
            f"● LIVE | SCF {payload['outer_iteration_display']:04d}/"
            f"{payload['outer_max_iterations']:04d} | stage "
            f"{payload.get('phase_number') or 0:02d}/{payload['phase_total']:02d} "
            f"{payload['phase_label']} | {work} | {loss_text} | "
            f"{time_text} | {current_text} | "
            f"alpha={payload['mixing_rate']:g} | RSS={_gib(payload.get('rss_bytes'))} "
            f"peak={_gib(payload.get('peak_rss_bytes'))} | total={_duration(payload['elapsed_seconds'])}"
        )

    def snapshot(self, *, append_history: bool = False, display: bool = False) -> dict[str, Any]:
        if not self.config.tracking.enabled:
            return {}
        with self._lock:
            payload = self._payload()
            atomic_json(self.run_dir / "progress.json", payload)
            if append_history:
                path = self.run_dir / "tracking_history.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            if display:
                self._emit(self._dashboard_line(payload))
        return payload

    def close(self, state: str) -> None:
        self.state = state
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        payload = self.snapshot(append_history=True)
        self._append_event("run_finished", state=state)
        self._emit(
            f"■ RUN {state.upper()} | completed_steps={len(self.history)} | "
            f"best_loss={payload.get('best_completed_loss') if payload.get('best_completed_loss') is not None else 'n/a'} "
            f"| elapsed={_duration(payload.get('elapsed_seconds'))}"
        )

    def __enter__(self) -> "RunTracker":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        state = "complete" if exc_type is None else (
            "interrupted" if issubclass(exc_type, (KeyboardInterrupt, SystemExit)) else "failed"
        )
        self.close(state)
        return False
