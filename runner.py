from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from backend.minimal_poles import MINIPOLE_COMMIT
from backend.observables import current_all, square_time_grid
from backend.system_classes import LeadParams, System
from backend.units import current_to_uA, energy_mev_to_gamma, time_to_ps


# =============================================================================
# Global configuration
# =============================================================================

GAMMA: float = 0.01  # eV
VERBOSE: bool = False
USE_FAKE_SOLVER: bool = False

ALPHA_DEFAULT: str = "L"
T_MAX = 6.0
N_T = 201

N_W_SCBA = 2001
OMEGA_INT_N_X = 1001
OMEGA_INT_N_OMEGA = 1001

N0_DEFAULT: float | None = None
MPM_TOL: float = 1e-8
MPM_N_IW: int = 512
MPM_BETA_FIT: float = 80.0
MPM_AAA_INITIAL_TERMS: int = 120
MPM_AAA_MAX_TERMS: int = 240
MPM_AAA_GROWTH_FACTOR: float = 1.5
STATIONARY_MODE = "weak_born"
PULSE_PROTOCOL = "square"
SQUARE_DURATION = 3.0  # hbar/Gamma

W_GRID = np.array([1.0, 2.5, 5.0, 10.0, 20.0, 100.0])
# Physical electron-phonon couplings in meV.  They are converted exactly once
# at the runner/backend boundary; System always stores dimensionless g/Gamma.
GQ_GRID = np.array([0.0, 0.01, 0.5, 1.0, 2.5, 5.0])

PARALLEL = False
MAX_WORKERS = 1
CONTINUE_ON_FAILURE = True

USE_TEX: bool = False
SAVE_SVG: bool = True
SHOW_PLOTS: bool = False
SAVE_NPY: bool = True

RUN_BASE = Path(".")


# =============================================================================
# Small logging helpers
# =============================================================================

class TimestampedWriter:
    def __init__(self, stream) -> None:
        self.stream = stream
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.stream.write(f"{line}\n")

        self.stream.flush()
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self.stream.write(f"{self._buffer}\n")
            self._buffer = ""
        self.stream.flush()


def make_run_dir(base: Path = RUN_BASE, protocol: str = PULSE_PROTOCOL) -> Path:
    if protocol not in ("downward", "upward", "square"):
        raise ValueError(f"Unknown pulse protocol {protocol!r}.")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{protocol}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(exist_ok=False)
    (run_dir / "figures").mkdir(exist_ok=False)
    return run_dir


def worker_log_path(log_dir: Path, i: int, j: int, W: float, g_q: float) -> Path:
    return log_dir / f"worker_W{W:.3f}_gq{g_q:.3f}.log"


def master_log_path(run_dir: Path) -> Path:
    return run_dir / "master.log"


def figure_path(fig_dir: Path, alpha: str, W: float, g_q: float) -> Path:
    return fig_dir / f"J_{alpha}_W{W:.3f}_gq{g_q:.3f}.svg"


def write_run_metadata(run_dir: Path) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(),
        "GAMMA": GAMMA,
        "VERBOSE": VERBOSE,
        "USE_FAKE_SOLVER": USE_FAKE_SOLVER,
        "ALPHA_DEFAULT": ALPHA_DEFAULT,
        "T_MAX": T_MAX,
        "N_T": N_T,
        "N_W_SCBA": N_W_SCBA,
        "OMEGA_INT_N_X": OMEGA_INT_N_X,
        "OMEGA_INT_N_OMEGA": OMEGA_INT_N_OMEGA,
        "N0": N0_DEFAULT,
        "MPM_TOL": MPM_TOL,
        "MPM_N_IW": MPM_N_IW,
        "MPM_BETA_FIT": MPM_BETA_FIT,
        "MPM_AAA_INITIAL_TERMS": MPM_AAA_INITIAL_TERMS,
        "MPM_AAA_MAX_TERMS": MPM_AAA_MAX_TERMS,
        "MPM_AAA_GROWTH_FACTOR": MPM_AAA_GROWTH_FACTOR,
        "STATIONARY_MODE": STATIONARY_MODE,
        "PULSE_PROTOCOL": PULSE_PROTOCOL,
        "SQUARE_DURATION": SQUARE_DURATION if PULSE_PROTOCOL == "square" else None,
        "SQUARE_DURATION_UNITS": "hbar/Gamma",
        "STATIONARY_REFERENCE": (
            "biased" if PULSE_PROTOCOL == "downward" else "unbiased"
        ),
        "W_GRID": W_GRID.tolist(),
        "GQ_GRID": GQ_GRID.tolist(),
        "GQ_GRID_UNITS": "meV",
        "GQ_GRID_GAMMA": energy_mev_to_gamma(GQ_GRID, GAMMA).tolist(),
        "PARALLEL": PARALLEL,
        "MAX_WORKERS": MAX_WORKERS,
        "CONTINUE_ON_FAILURE": CONTINUE_ON_FAILURE,
        "USE_TEX": USE_TEX,
        "SAVE_SVG": SAVE_SVG,
        "SHOW_PLOTS": SHOW_PLOTS,
        "SAVE_NPY": SAVE_NPY,
        "MINIPOLE_COMMIT": MINIPOLE_COMMIT,
    }

    with open(run_dir / "run_info.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def save_currents_npy(
    run_dir: Path,
    t_ps: np.ndarray,
    J_grid_uA: np.ndarray,
    W_grid: np.ndarray,
    gq_grid: np.ndarray,
    alpha: str,
) -> None:
    data_dir = run_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Save the shared time axis once
    np.save(data_dir / "t_ps.npy", t_ps)

    total = len(W_grid) * len(gq_grid)
    count = 0
    failed = 0

    for i, W in enumerate(W_grid):
        for j, gq in enumerate(gq_grid):
            if not np.all(np.isfinite(J_grid_uA[i, j, :])):
                failed += 1
                print(
                    f"Skipped failed NPY {failed} → W={float(W):.3f}, "
                    f"g_q={float(gq):.3f}",
                    flush=True,
                )
                continue
            fname = f"J_{alpha}_W{float(W):.3f}_gq{float(gq):.3f}.npy"
            out_path = data_dir / fname
            np.save(out_path, J_grid_uA[i, j, :])
            count += 1
            print(f"Saved NPY {count}/{total} → {out_path}", flush=True)

    print()
    print("#" * 82)
    print(f"Saved t_ps.npy + {count} current NPY files to {data_dir}")
    print(f"Skipped {failed} failed parameter sets")
    print("#" * 82)


# =============================================================================
# System construction
# =============================================================================

def make_sys(W: float, g_q: float) -> System:
    # Linear SCBA is noncontractive in the present model above g/Gamma ~ 1.5.
    # Bound those diagnostic attempts so a failed strong-coupling point does
    # not monopolize an entire sweep.  Successful weak/intermediate points keep
    # the tighter production iteration budget.
    g_q_gamma = float(energy_mev_to_gamma(g_q, GAMMA))
    strong_coupling = abs(g_q_gamma) > 1.0
    if W <= 1.0:
        if PULSE_PROTOCOL in ("upward", "square") and g_q >= 20.0:
            # The unbiased weak-Born reference develops a narrow high-coupling
            # feature that is under-resolved at the downward grid density.
            scba_points = 40001
        else:
            scba_points = 20001 if g_q >= 20.0 else 10001
    elif W <= 2.5 and g_q >= 20.0:
        scba_points = 8001
    elif W <= 5.0 and g_q >= 20.0:
        scba_points = 4001
    else:
        scba_points = N_W_SCBA
    return System(
        pulse_protocol=PULSE_PROTOCOL,
        pulse_duration=SQUARE_DURATION if PULSE_PROTOCOL == "square" else None,
        ETA=1e-3,
        DELTA=5.0,
        leads={
            "L": LeadParams(
                Gamma0=0.5,
                Delta=10.0,
                beta=10,
                mu=0.0,
            ),
            "R": LeadParams(
                Gamma0=0.5,
                Delta=0.0,
                beta=10,
                mu=0.0,
            ),
        },
        W=W,
        g_q=g_q_gamma,
        w_q=0.2,
        e_0=0.0,
        beta_ph=20.0,
        mu_ph=0.0,
        N0=N0_DEFAULT,
        beta_fd=10,
        mu_fd=0.0,
        e_min=-20.0,
        e_max=20.0,
        omega_min=-100.0,
        omega_max=100.0,
        scba_max_iter=200_000,
        scba_mode=STATIONARY_MODE,
        scba_tol_abs=1e-5,
        scba_tol_rel=1e-4,
        scba_mixing=0.05,
        scba_min_iter=10,
        n_w_scba=scba_points,
        mpm_tol=MPM_TOL,
        mpm_n_iw=MPM_N_IW,
        mpm_beta_fit=MPM_BETA_FIT,
        mpm_aaa_initial_terms=MPM_AAA_INITIAL_TERMS,
        mpm_aaa_max_terms=MPM_AAA_MAX_TERMS,
        mpm_aaa_growth_factor=MPM_AAA_GROWTH_FACTOR,
        mpm_green_rel_tol=7e-2,
        verbose=VERBOSE,
    )


# =============================================================================
# Fake backend for quick UI testing
# =============================================================================

def fake_current_alpha(
    sys: System,
    alpha: str,
    t_max: float,
    n_t: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sys.pulse_protocol == "square":
        t = square_time_grid(t_max, n_t, float(sys.pulse_duration))
    else:
        t = np.linspace(0.0, t_max, n_t, dtype=float)

    W = sys.W
    gq = sys.g_q

    omega = 2.0 + 6.0 * gq
    decay = np.exp(-0.15 * W * t)

    current = decay * (
        np.sin(omega * t)
        + 0.35 * np.sin(2.0 * omega * t)
    )

    if alpha == "R":
        current = -0.8 * current

    return t, current


# =============================================================================
# Current computation
# =============================================================================

def _compute_current_with_diagnostics(
    W: float,
    g_q: float,
    alpha: str = ALPHA_DEFAULT,
    t_max: float = T_MAX,
    n_t: int = N_T,
) -> tuple[np.ndarray, np.ndarray, dict]:
    sys = make_sys(W=W, g_q=g_q)
    resolved_g_q_gamma = sys.g_q

    if USE_FAKE_SOLVER:
        if VERBOSE:
            print()
            print("=" * 82)
            print(f"Fake current evaluation | alpha={alpha} | W={W:.3f} | g_q={g_q:.3f}")
            print("=" * 82)

        t_dimless, I_dimless = fake_current_alpha(
            sys=sys,
            alpha=alpha,
            t_max=t_max,
            n_t=n_t,
        )
        diagnostics = {"fake_solver": True, "N0": None}
    else:
        sys.launch()

        if VERBOSE:
            sys.reporter().print_unit_system(GAMMA)

        transient = current_all(
            sys=sys,
            t_max=t_max,
            n_t=n_t,
            omega_int_n_x=OMEGA_INT_N_X,
            omega_int_n_omega=OMEGA_INT_N_OMEGA,
            pulse_duration=(SQUARE_DURATION if PULSE_PROTOCOL == "square" else None),
        )
        t_dimless = transient.t
        I_dimless = transient.currents[alpha]
        diagnostics = dict(transient.diagnostics)
        diagnostics["fake_solver"] = False
        if VERBOSE:
            print("Transient quality diagnostics:")
            for key, value in transient.diagnostics.items():
                print(f"  {key} = {value}")

    current_unit_A = GAMMA * 1.602176634e-19 / (1.054571817e-34 / 1.602176634e-19)
    t_ps = time_to_ps(t_dimless, GAMMA)
    I_uA = current_to_uA(I_dimless, GAMMA)

    if VERBOSE:
        print(f"max|t_dimless| = {np.max(np.abs(t_dimless)):.6e}")
        print(f"max|I_raw|     = {np.max(np.abs(I_dimless)):.6e}")
        print(f"current unit   = {current_unit_A:.6e} A")
        print(f"current unit   = {1e6 * current_unit_A:.6e} uA")
        print(f"max|I_uA|      = {np.max(np.abs(I_uA)):.6e} uA")

    diagnostics["g_q_input_meV"] = float(g_q)
    diagnostics["g_q_resolved_Gamma"] = float(resolved_g_q_gamma)
    diagnostics["g_q_resolved_eV"] = float(g_q) * 1e-3
    diagnostics["pulse_protocol"] = PULSE_PROTOCOL
    diagnostics["pulse_duration"] = (
        SQUARE_DURATION if PULSE_PROTOCOL == "square" else None
    )
    diagnostics["stationary_reference"] = (
        "biased" if PULSE_PROTOCOL == "downward" else "unbiased"
    )
    return t_ps, I_uA, diagnostics


def compute_current(
    W: float,
    g_q: float,
    alpha: str = ALPHA_DEFAULT,
    t_max: float = T_MAX,
    n_t: int = N_T,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper returning only the time grid and selected current."""
    t_ps, current_uA, _ = _compute_current_with_diagnostics(
        W=W, g_q=g_q, alpha=alpha, t_max=t_max, n_t=n_t
    )
    return t_ps, current_uA


# =============================================================================
# Parallel worker
# =============================================================================

def _compute_current_job(
    i: int,
    j: int,
    W: float,
    g_q: float,
    alpha: str,
    t_max: float,
    n_t: int,
    log_dir_str: str,
) -> tuple[int, int, np.ndarray, np.ndarray, str]:
    log_dir = Path(log_dir_str)
    log_path = worker_log_path(log_dir, i, j, W, g_q)

    with open(log_path, "w", encoding="utf-8", buffering=1) as raw_fh:
        writer = TimestampedWriter(raw_fh)

        with redirect_stdout(writer), redirect_stderr(writer):
            pid = os.getpid()

            print("#" * 82)
            print("Worker started")
            print(f"pid={pid}")
            print(f"job indices: i={i}, j={j}")
            print(f"W={W:.6f}")
            print(f"g_q={g_q:.6f}")
            print(f"g_q units=meV; resolved g_q/Gamma={energy_mev_to_gamma(g_q, GAMMA):.6e}")
            print(f"pulse_protocol={PULSE_PROTOCOL}")
            if PULSE_PROTOCOL == "square":
                print(f"pulse_duration={SQUARE_DURATION} hbar/Gamma")
            print(
                "stationary_reference="
                f"{'biased' if PULSE_PROTOCOL == 'downward' else 'unbiased'}"
            )
            print(f"alpha={alpha}")
            print(f"t_max={t_max}")
            print(f"n_t={n_t}")
            print("#" * 82)

            try:
                t, current, diagnostics = _compute_current_with_diagnostics(
                    W=W,
                    g_q=g_q,
                    alpha=alpha,
                    t_max=t_max,
                    n_t=n_t,
                )
                print("Worker finished successfully")
                print(f"max|J| = {np.max(np.abs(current)):.6e} µA")
                print("Quality diagnostics:")
                for key, value in diagnostics.items():
                    print(f"  {key}={value}")
                print(f"log_path = {log_path}")
            except Exception:
                print("Worker failed with exception:")
                import traceback
                failure = {
                    "status": "failed",
                    "W": W,
                    "g_q_input_meV": g_q,
                    "g_q_resolved_Gamma": float(energy_mev_to_gamma(g_q, GAMMA)),
                    "pulse_protocol": PULSE_PROTOCOL,
                    "pulse_duration": (
                        SQUARE_DURATION if PULSE_PROTOCOL == "square" else None
                    ),
                    "exception": traceback.format_exc(),
                }
                traceback.print_exc()
                failure_path = log_path.with_suffix(".failed.json")
                with open(failure_path, "w", encoding="utf-8") as failure_fh:
                    json.dump(failure, failure_fh, indent=2)
                raise

    quality_path = log_path.with_suffix(".quality.json")
    with open(quality_path, "w", encoding="utf-8") as quality_fh:
        json.dump(diagnostics, quality_fh, indent=2)
    return i, j, t, current, str(log_path)


# =============================================================================
# Precomputation over parameter grid
# =============================================================================

def precompute_currents_parallel(
    W_grid: np.ndarray,
    gq_grid: np.ndarray,
    alpha: str = ALPHA_DEFAULT,
    t_max: float = T_MAX,
    n_t: int = N_T,
    max_workers: int | None = None,
    log_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if log_dir is None:
        raise ValueError("log_dir must be provided for parallel precomputation.")

    t_ref: np.ndarray | None = None
    J_grid = np.full((len(W_grid), len(gq_grid), n_t), np.nan, dtype=float)

    jobs: list[tuple[int, int, float, float, str, float, int, str]] = [
        (i, j, float(W), float(gq), alpha, t_max, n_t, str(log_dir))
        for i, W in enumerate(W_grid)
        for j, gq in enumerate(gq_grid)
    ]

    total = len(jobs)
    count = 0

    print()
    print("#" * 82)
    print("Beginning PARALLEL current precomputation over parameter grid")
    print("#" * 82)
    print(f"Gamma = {GAMMA:.6e} eV")
    print(f"alpha = {alpha}")
    print(f"t_max (dimensionless) = {t_max}")
    print(f"n_t = {n_t}")
    print(f"number of W values = {len(W_grid)}")
    print(f"number of g_q values = {len(gq_grid)}")
    print(f"total jobs = {total}")
    print(f"max_workers = {max_workers}")
    print(f"log_dir = {log_dir}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_compute_current_job, *job): job for job in jobs
        }

        for future in as_completed(futures):
            job = futures[future]
            try:
                i, j, t, current, log_path = future.result()
            except Exception as exc:
                if not CONTINUE_ON_FAILURE:
                    raise
                i, j, W, gq = job[:4]
                count += 1
                print(
                    f"Failed current {count}/{total} | (i={i}, j={j}) | "
                    f"W={W:.3f}, g_q={gq:.3f} | {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            if t_ref is None:
                t_ref = t
            elif not np.allclose(t, t_ref):
                raise ValueError("Inconsistent time grids encountered during precomputation.")

            J_grid[i, j, :] = np.real(current)
            count += 1

            print(
                f"Stored current {count}/{total} | "
                f"(i={i}, j={j}) | "
                f"max|J| = {np.max(np.abs(current)):.6e} µA | "
                f"log={log_path}",
                flush=True,
            )

    if t_ref is None:
        if not CONTINUE_ON_FAILURE:
            raise RuntimeError("No currents were computed.")
        fallback = (
            square_time_grid(t_max, n_t, SQUARE_DURATION)
            if PULSE_PROTOCOL == "square"
            else np.linspace(0.0, t_max, n_t)
        )
        t_ref = time_to_ps(fallback, GAMMA)
        print("No parameter set succeeded; preserved failure diagnostics only.")

    print()
    print("#" * 82)
    print("Finished parallel current precomputation")
    print("#" * 82)

    return t_ref, J_grid


def precompute_currents_serial(
    W_grid: np.ndarray,
    gq_grid: np.ndarray,
    alpha: str = ALPHA_DEFAULT,
    t_max: float = T_MAX,
    n_t: int = N_T,
    log_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if log_dir is None:
        raise ValueError("log_dir must be provided for serial precomputation.")

    t_ref: np.ndarray | None = None
    J_grid = np.full((len(W_grid), len(gq_grid), n_t), np.nan, dtype=float)

    total = len(W_grid) * len(gq_grid)
    count = 0

    print()
    print("#" * 82)
    print("Beginning SERIAL current precomputation over parameter grid")
    print("#" * 82)
    print(f"Gamma = {GAMMA:.6e} eV")
    print(f"alpha = {alpha}")
    print(f"t_max (dimensionless) = {t_max}")
    print(f"n_t = {n_t}")
    print(f"number of W values = {len(W_grid)}")
    print(f"number of g_q values = {len(gq_grid)}")
    print(f"total jobs = {total}")
    print(f"log_dir = {log_dir}")

    for i, W in enumerate(W_grid):
        for j, gq in enumerate(gq_grid):
            log_path = worker_log_path(log_dir, i, j, float(W), float(gq))
            failed_exception: Exception | None = None
            diagnostics: dict = {}

            with open(log_path, "w", encoding="utf-8", buffering=1) as raw_fh:
                writer = TimestampedWriter(raw_fh)

                with redirect_stdout(writer), redirect_stderr(writer):
                    print("#" * 82)
                    print("Serial job started")
                    print(f"job indices: i={i}, j={j}")
                    print(f"W={W:.6f}")
                    print(f"g_q={gq:.6f}")
                    print(f"g_q units=meV; resolved g_q/Gamma={energy_mev_to_gamma(gq, GAMMA):.6e}")
                    print(f"pulse_protocol={PULSE_PROTOCOL}")
                    if PULSE_PROTOCOL == "square":
                        print(f"pulse_duration={SQUARE_DURATION} hbar/Gamma")
                    print(
                        "stationary_reference="
                        f"{'biased' if PULSE_PROTOCOL == 'downward' else 'unbiased'}"
                    )
                    print(f"alpha={alpha}")
                    print(f"t_max={t_max}")
                    print(f"n_t={n_t}")
                    print("#" * 82)

                    try:
                        t, current, diagnostics = _compute_current_with_diagnostics(
                            W=float(W),
                            g_q=float(gq),
                            alpha=alpha,
                            t_max=t_max,
                            n_t=n_t,
                        )
                        print("Serial job finished successfully")
                        print(f"max|J| = {np.max(np.abs(current)):.6e} µA")
                        print("Quality diagnostics:")
                        for key, value in diagnostics.items():
                            print(f"  {key}={value}")
                        print(f"log_path = {log_path}")
                    except Exception as exc:
                        import traceback
                        failed_exception = exc
                        print("Serial job failed with exception:")
                        traceback.print_exc()

            if failed_exception is not None:
                failure_path = log_path.with_suffix(".failed.json")
                with open(failure_path, "w", encoding="utf-8") as failure_fh:
                    json.dump(
                        {
                            "status": "failed",
                            "W": float(W),
                            "g_q_input_meV": float(gq),
                            "g_q_resolved_Gamma": float(
                                energy_mev_to_gamma(gq, GAMMA)
                            ),
                            "pulse_protocol": PULSE_PROTOCOL,
                            "pulse_duration": (
                                SQUARE_DURATION if PULSE_PROTOCOL == "square" else None
                            ),
                            "exception_type": type(failed_exception).__name__,
                            "message": str(failed_exception),
                        },
                        failure_fh,
                        indent=2,
                    )
                count += 1
                print(
                    f"Failed current {count}/{total} | (i={i}, j={j}) | "
                    f"W={float(W):.3f}, g_q={float(gq):.3f} | "
                    f"{type(failed_exception).__name__}: {failed_exception}",
                    flush=True,
                )
                if CONTINUE_ON_FAILURE:
                    continue
                raise failed_exception

            quality_path = log_path.with_suffix(".quality.json")
            with open(quality_path, "w", encoding="utf-8") as quality_fh:
                json.dump(diagnostics, quality_fh, indent=2)

            if t_ref is None:
                t_ref = t
            elif not np.allclose(t, t_ref):
                raise ValueError("Inconsistent time grids encountered during precomputation.")

            J_grid[i, j, :] = np.real(current)
            count += 1

            print(
                f"Stored current {count}/{total} | "
                f"(i={i}, j={j}) | "
                f"max|J| = {np.max(np.abs(current)):.6e} µA | "
                f"log={log_path}",
                flush=True,
            )

    if t_ref is None:
        if not CONTINUE_ON_FAILURE:
            raise RuntimeError("No currents were computed.")
        fallback = (
            square_time_grid(t_max, n_t, SQUARE_DURATION)
            if PULSE_PROTOCOL == "square"
            else np.linspace(0.0, t_max, n_t)
        )
        t_ref = time_to_ps(fallback, GAMMA)
        print("No parameter set succeeded; preserved failure diagnostics only.")

    print()
    print("#" * 82)
    print("Finished serial current precomputation")
    print("#" * 82)

    return t_ref, J_grid


def precompute_currents(
    W_grid: np.ndarray,
    gq_grid: np.ndarray,
    alpha: str = ALPHA_DEFAULT,
    t_max: float = T_MAX,
    n_t: int = N_T,
    parallel: bool = PARALLEL,
    max_workers: int | None = MAX_WORKERS,
    log_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if parallel:
        return precompute_currents_parallel(
            W_grid=W_grid,
            gq_grid=gq_grid,
            alpha=alpha,
            t_max=t_max,
            n_t=n_t,
            max_workers=max_workers,
            log_dir=log_dir,
        )

    return precompute_currents_serial(
        W_grid=W_grid,
        gq_grid=gq_grid,
        alpha=alpha,
        t_max=t_max,
        n_t=n_t,
        log_dir=log_dir,
    )


# =============================================================================
# Plot helpers
# =============================================================================

def configure_matplotlib(use_tex: bool = USE_TEX) -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 18
    plt.rcParams["text.usetex"] = use_tex

    if use_tex:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def make_title(alpha: str, W: float, g_q: float) -> str:
    duration = (
        rf",\quad s={SQUARE_DURATION:g}\,\hbar/\Gamma"
        if PULSE_PROTOCOL == "square" else ""
    )
    return (
        rf"{PULSE_PROTOCOL.capitalize()} transient current $J_{{{alpha}}}(t)$"
        "\n"
        rf"$W={W:.3f}\Gamma,\quad g_q={g_q:.3f}\,\mathrm{{meV}}{duration}$"
    )


def make_ylabel(alpha: str) -> str:
    return rf"$J_{{{alpha}}}(t)$ ($\mu$A)"


def save_current_plot_svg(
    t_ps: np.ndarray,
    current_uA: np.ndarray,
    alpha: str,
    W: float,
    g_q: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        t_ps,
        current_uA,
        linewidth=2.0,
        label=rf"$J_{{{alpha}}}(t)$",
    )
    if PULSE_PROTOCOL == "square":
        ax.axvline(
            float(time_to_ps(SQUARE_DURATION, GAMMA)), color="0.25",
            linestyle=":", linewidth=1.5, label=rf"turnoff $s$",
        )

    ax.set_xlabel(r"$t$ (ps)")
    ax.set_ylabel(make_ylabel(alpha))
    ax.set_title(make_title(alpha, W, g_q))
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)

def save_all_current_plots_svg(
    t_ps: np.ndarray,
    J_grid_uA: np.ndarray,
    W_grid: np.ndarray,
    gq_grid: np.ndarray,
    alpha: str,
    fig_dir: Path,
) -> None:
    total = len(W_grid) * len(gq_grid)
    count = 0
    failed = 0

    for i, W in enumerate(W_grid):
        for j, gq in enumerate(gq_grid):
            if not np.all(np.isfinite(J_grid_uA[i, j, :])):
                failed += 1
                print(
                    f"Skipped failed figure {failed} → W={float(W):.3f}, "
                    f"g_q={float(gq):.3f}",
                    flush=True,
                )
                continue
            out_path = figure_path(fig_dir, alpha, float(W), float(gq))
            save_current_plot_svg(
                t_ps=t_ps,
                current_uA=J_grid_uA[i, j, :],
                alpha=alpha,
                W=float(W),
                g_q=float(gq),
                out_path=out_path,
            )
            count += 1
            print(f"Saved figure {count}/{total} → {out_path}", flush=True)

    print()
    print("#" * 82)
    print(f"Saved {count} SVG figures to {fig_dir}")
    print(f"Skipped {failed} failed parameter sets")
    print("#" * 82)


def show_single_reference_plot(
    t_ps: np.ndarray,
    J_grid_uA: np.ndarray,
    W_grid: np.ndarray,
    gq_grid: np.ndarray,
    alpha: str,
    W0: float = 20.0,
    gq0: float = 0.1,
) -> None:
    i0 = int(np.argmin(np.abs(W_grid - W0)))
    j0 = int(np.argmin(np.abs(gq_grid - gq0)))
    current = J_grid_uA[i0, j0, :]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        t_ps,
        np.real(current),
        linewidth=2.0,
        label=rf"$\mathrm{{Re}}\,J_{{{alpha}}}(t)$",
    )
    ax.plot(
        t_ps,
        np.imag(current),
        linewidth=2.0,
        linestyle="--",
        label=rf"$\mathrm{{Im}}\,J_{{{alpha}}}(t)$",
    )

    ax.set_xlabel(r"$t$ (ps)")
    ax.set_ylabel(make_ylabel(alpha))
    ax.set_title(make_title(alpha, float(W_grid[i0]), float(gq_grid[j0])))
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.show()


# =============================================================================
# Main program
# =============================================================================

def main() -> None:
    run_dir = make_run_dir(protocol=PULSE_PROTOCOL)
    log_dir = run_dir / "logs"
    fig_dir = run_dir / "figures"
    write_run_metadata(run_dir)

    with open(master_log_path(run_dir), "w", encoding="utf-8", buffering=1) as raw_fh:
        writer = TimestampedWriter(raw_fh)

        with redirect_stdout(writer), redirect_stderr(writer):
            configure_matplotlib(use_tex=USE_TEX)

            alpha0 = ALPHA_DEFAULT

            print("#" * 82)
            print("Run started")
            print(f"run_dir = {run_dir}")
            print(f"log_dir = {log_dir}")
            print(f"fig_dir = {fig_dir}")
            print(f"PULSE_PROTOCOL = {PULSE_PROTOCOL}")
            print(f"STATIONARY_MODE = {STATIONARY_MODE}")
            if PULSE_PROTOCOL == "square":
                print(f"SQUARE_DURATION = {SQUARE_DURATION} hbar/Gamma")
            print(
                "STATIONARY_REFERENCE = "
                f"{'biased' if PULSE_PROTOCOL == 'downward' else 'unbiased'}"
            )
            print(f"USE_TEX = {USE_TEX}")
            print(f"SAVE_SVG = {SAVE_SVG}")
            print(f"SHOW_PLOTS = {SHOW_PLOTS}")
            print(f"SAVE_NPY = {SAVE_NPY}")
            print(f"N_W_SCBA = {N_W_SCBA}")
            print(f"OMEGA_INT_N_X = {OMEGA_INT_N_X}")
            print(f"OMEGA_INT_N_OMEGA = {OMEGA_INT_N_OMEGA}")
            print("#" * 82)

            t_ps, J_grid_uA = precompute_currents(
                W_grid=W_GRID,
                gq_grid=GQ_GRID,
                alpha=alpha0,
                t_max=T_MAX,
                n_t=N_T,
                parallel=PARALLEL,
                max_workers=MAX_WORKERS,
                log_dir=log_dir,
            )
            successful_jobs = int(
                np.sum(np.all(np.isfinite(J_grid_uA), axis=-1))
            )
            failed_jobs = int(J_grid_uA.shape[0] * J_grid_uA.shape[1] - successful_jobs)
            print(
                f"Grid summary: {successful_jobs} succeeded, "
                f"{failed_jobs} failed"
            )

            if SAVE_NPY:
                save_currents_npy(
                    run_dir=run_dir,
                    t_ps=t_ps,
                    J_grid_uA=J_grid_uA,
                    W_grid=W_GRID,
                    gq_grid=GQ_GRID,
                    alpha=alpha0,
                )

            if SAVE_SVG:
                save_all_current_plots_svg(
                    t_ps=t_ps,
                    J_grid_uA=J_grid_uA,
                    W_grid=W_GRID,
                    gq_grid=GQ_GRID,
                    alpha=alpha0,
                    fig_dir=fig_dir,
                )

            print("#" * 82)
            print("Run finished; inspect the grid summary and failure diagnostics")
            print("#" * 82)

    if SHOW_PLOTS:
        configure_matplotlib(use_tex=USE_TEX)
        show_single_reference_plot(
            t_ps=t_ps,
            J_grid_uA=J_grid_uA,
            W_grid=W_GRID,
            gq_grid=GQ_GRID,
            alpha=ALPHA_DEFAULT,
        )


if __name__ == "__main__":
    main()
