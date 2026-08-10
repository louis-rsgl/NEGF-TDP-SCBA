from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from backend.observables import current_alpha
from backend.system_classes import LeadParams, System
from backend.units import current_to_uA, time_to_ps


# =============================================================================
# Global configuration
# =============================================================================

GAMMA: float = 0.01  # eV
VERBOSE: bool = True
USE_FAKE_SOLVER: bool = False

ALPHA_DEFAULT: str = "L"
T_MAX: float = 2.0
N_T: int = 2001

N_W_SCBA: int = 1001
OMEGA_INT_N_X: int | None = 1001
OMEGA_INT_N_OMEGA: int | None = 1001

W_GRID = np.array([1, 2.5, 5, 10.0, 20.0, 100.0], dtype=float)
GQ_GRID = np.array([0.0], dtype=float)

PARALLEL: bool = True
MAX_WORKERS: int | None = 25

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


def make_run_dir(base: Path = RUN_BASE) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{run_id}"
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
        "W_GRID": W_GRID.tolist(),
        "GQ_GRID": GQ_GRID.tolist(),
        "PARALLEL": PARALLEL,
        "MAX_WORKERS": MAX_WORKERS,
        "USE_TEX": USE_TEX,
        "SAVE_SVG": SAVE_SVG,
        "SHOW_PLOTS": SHOW_PLOTS,
        "SAVE_NPY": SAVE_NPY,
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

    for i, W in enumerate(W_grid):
        for j, gq in enumerate(gq_grid):
            fname = f"J_{alpha}_W{float(W):.3f}_gq{float(gq):.3f}.npy"
            out_path = data_dir / fname
            np.save(out_path, J_grid_uA[i, j, :])
            count += 1
            print(f"Saved NPY {count}/{total} → {out_path}", flush=True)

    print()
    print("#" * 82)
    print(f"Saved t_ps.npy + {total} current NPY files to {data_dir}")
    print("#" * 82)


# =============================================================================
# System construction
# =============================================================================

def make_sys(W: float, g_q: float) -> System:
    return System(
        ETA=1e-3,
        DELTA=5.0,
        leads={
            "L": LeadParams(
                Gamma0=0.5,
                Delta=10.0,
                beta=0.1,
                mu=0.0,
            ),
            "R": LeadParams(
                Gamma0=0.5,
                Delta=0.0,
                beta=0.1,
                mu=0.0,
            ),
        },
        W=W,
        g_q=g_q,
        w_q=0.2,
        e_0=0.0,
        beta_ph=20.0,
        mu_ph=0.0,
        beta_fd=0.1,
        mu_fd=0.0,
        e_min=-100.0,
        e_max=100.0,
        omega_min=-100.0,
        omega_max=100.0,
        scba_max_iter=20_000,
        scba_tol_abs=1e-5,
        scba_tol_rel=1e-4,
        scba_mixing=0.1,
        scba_min_iter=10,
        n_w_scba=N_W_SCBA,
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
    t = np.linspace(0.0, t_max, n_t, dtype=float)

    W = sys.W
    gq = sys.g_q

    omega = 2.0 + 6.0 * gq
    decay = np.exp(-0.15 * W * t)

    current = decay * (
        np.sin(omega * t)
        + 0.35 * np.sin(2.0 * omega * t)
        + 1j * 0.7 * np.cos(omega * t)
    )

    if alpha == "R":
        current = -0.8 * current

    return t, current


# =============================================================================
# Current computation
# =============================================================================

def compute_current(
    W: float,
    g_q: float,
    alpha: str = ALPHA_DEFAULT,
    t_max: float = T_MAX,
    n_t: int = N_T,
) -> tuple[np.ndarray, np.ndarray]:
    sys = make_sys(W=W, g_q=g_q)

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
    else:
        sys.launch()

        if VERBOSE:
            sys.reporter().print_unit_system(GAMMA)

        t_dimless, I_dimless = current_alpha(
            sys=sys,
            alpha=alpha,
            t_max=t_max,
            n_t=n_t,
            omega_int_n_x=OMEGA_INT_N_X,
            omega_int_n_omega=OMEGA_INT_N_OMEGA,
        )

    current_unit_A = GAMMA * 1.602176634e-19 / (1.054571817e-34 / 1.602176634e-19)
    t_ps = time_to_ps(t_dimless, GAMMA)
    I_uA = current_to_uA(I_dimless, GAMMA)

    if VERBOSE:
        print(f"max|t_dimless| = {np.max(np.abs(t_dimless)):.6e}")
        print(f"max|I_raw|     = {np.max(np.abs(I_dimless)):.6e}")
        print(f"current unit   = {current_unit_A:.6e} A")
        print(f"current unit   = {1e6 * current_unit_A:.6e} uA")
        print(f"max|I_uA|      = {np.max(np.abs(I_uA)):.6e} uA")

    return t_ps, I_uA


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
            print(f"alpha={alpha}")
            print(f"t_max={t_max}")
            print(f"n_t={n_t}")
            print("#" * 82)

            try:
                t, current = compute_current(
                    W=W,
                    g_q=g_q,
                    alpha=alpha,
                    t_max=t_max,
                    n_t=n_t,
                )
                print("Worker finished successfully")
                print(f"max|J| = {np.max(np.abs(current)):.6e} µA")
                print(f"log_path = {log_path}")
            except Exception:
                print("Worker failed with exception:")
                import traceback
                traceback.print_exc()
                raise

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
    J_grid = np.empty((len(W_grid), len(gq_grid), n_t), dtype=np.complex128)

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
        futures = [executor.submit(_compute_current_job, *job) for job in jobs]

        for future in as_completed(futures):
            i, j, t, current, log_path = future.result()

            if t_ref is None:
                t_ref = t
            elif not np.allclose(t, t_ref):
                raise ValueError("Inconsistent time grids encountered during precomputation.")

            J_grid[i, j, :] = current
            count += 1

            print(
                f"Stored current {count}/{total} | "
                f"(i={i}, j={j}) | "
                f"max|J| = {np.max(np.abs(current)):.6e} µA | "
                f"log={log_path}",
                flush=True,
            )

    if t_ref is None:
        raise RuntimeError("No currents were computed.")

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
    J_grid = np.empty((len(W_grid), len(gq_grid), n_t), dtype=np.complex128)

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

            with open(log_path, "w", encoding="utf-8", buffering=1) as raw_fh:
                writer = TimestampedWriter(raw_fh)

                with redirect_stdout(writer), redirect_stderr(writer):
                    print("#" * 82)
                    print("Serial job started")
                    print(f"job indices: i={i}, j={j}")
                    print(f"W={W:.6f}")
                    print(f"g_q={gq:.6f}")
                    print(f"alpha={alpha}")
                    print(f"t_max={t_max}")
                    print(f"n_t={n_t}")
                    print("#" * 82)

                    t, current = compute_current(
                        W=float(W),
                        g_q=float(gq),
                        alpha=alpha,
                        t_max=t_max,
                        n_t=n_t,
                    )

                    print("Serial job finished successfully")
                    print(f"max|J| = {np.max(np.abs(current)):.6e} µA")
                    print(f"log_path = {log_path}")

            if t_ref is None:
                t_ref = t
            elif not np.allclose(t, t_ref):
                raise ValueError("Inconsistent time grids encountered during precomputation.")

            J_grid[i, j, :] = current
            count += 1

            print(
                f"Stored current {count}/{total} | "
                f"(i={i}, j={j}) | "
                f"max|J| = {np.max(np.abs(current)):.6e} µA | "
                f"log={log_path}",
                flush=True,
            )

    if t_ref is None:
        raise RuntimeError("No currents were computed.")

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
    return (
        rf"Transient current $J_{{{alpha}}}(t)$"
        "\n"
        rf"$W={W:.3f}\Gamma,\quad g_q={g_q:.3f}\Gamma$"
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

    for i, W in enumerate(W_grid):
        for j, gq in enumerate(gq_grid):
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
    print(f"Saved {total} SVG figures to {fig_dir}")
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
    run_dir = make_run_dir()
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
            print("Run finished successfully")
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