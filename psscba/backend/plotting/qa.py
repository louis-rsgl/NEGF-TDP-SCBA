from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

_mpl_cache = Path(tempfile.gettempdir()) / f"psscba-matplotlib-{os.getuid()}"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from psscba.backend.model.units import current_to_uA, time_to_ps
from psscba.backend.poles.minimal import build_pole_cache, sigma_mpm
from psscba.frontend.configuration import RunConfig, load_config
from psscba.frontend.storage import (
    atomic_json,
    load_kernel,
    load_pole_basis,
    load_saved_current_views,
)
from psscba.backend.stationary.initializer import make_system
from psscba.types import FixedKernelTrajectory, OuterIteration, ProjectionResult, PSSCBAResult


COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


@contextmanager
def publication_style():
    with mpl.rc_context({
        "font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "text.usetex": False, "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 12, "axes.linewidth": 1.0,
        "lines.linewidth": 2.0, "xtick.minor.visible": True, "ytick.minor.visible": True,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.7,
        "legend.frameon": False, "svg.fonttype": "none", "savefig.transparent": False,
        "svg.hashsalt": "negf-tdp-ps-scba-v3",
    }):
        yield


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".svg", dir=path.parent)
    os.close(fd)
    try:
        fig.savefig(
            temporary,
            format="svg",
            bbox_inches="tight",
            metadata={"Date": None, "Creator": "NEGF TDP PS-SCBA"},
        )
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        plt.close(fig)


def save_svg(fig, path: Path) -> None:
    """Atomically save and close a Matplotlib figure as SVG."""
    _save(fig, path)


def _title(config: RunConfig, method: str) -> str:
    turnoff = ""
    if config.protocol.name == "square":
        turnoff = rf", $s\Gamma={float(config.protocol.duration):g}$"
    return (
        rf"{config.protocol.name}{turnoff}; $W/\Gamma={config.model.bandwidth:g}$; "
        rf"$\omega_0/\Gamma={config.model.phonon_energy:g}$; "
        rf"$g={config.model.coupling_meV:g}\,\mathrm{{meV}}$ "
        rf"$(g/\Gamma={config.model.coupling:g})$; {method.replace('_', ' ')}"
    )


def _positive(values: Iterable[float], floor: float = 1e-18) -> np.ndarray:
    data = np.asarray(
        tuple(np.nan if value is None else value for value in values), dtype=float
    )
    return np.where(np.isfinite(data), np.maximum(np.abs(data), floor), np.nan)


def _switch_markers(ax, config: RunConfig, *, physical: bool = False) -> None:
    scale = float(time_to_ps(1.0, config.model.gamma_eV)) if physical else 1.0
    ax.axvline(0.0, color="0.25", linewidth=1.0, zorder=0)
    if config.protocol.name == "square":
        ax.axvline(scale * float(config.protocol.duration), color="0.25", linestyle="--", linewidth=1.2, label="turnoff")


def plot_outer_convergence(config: RunConfig, history, path: Path) -> None:
    with publication_style():
        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
        if history:
            iteration = np.asarray([item.iteration for item in history])
            axes[0].semilogy(iteration, _positive(item.residual_retarded for item in history), marker="o", color=COLORS[0], label=r"$R_\Sigma^R$")
            axes[0].semilogy(iteration, _positive(item.residual_lesser for item in history), marker="s", color=COLORS[1], label=r"$R_\Sigma^<$")
            axes[0].axhline(config.numerics.outer_tolerance, color="0.25", linestyle="--", label="tolerance")
            axes[1].semilogy(iteration, _positive(item.projected_green_retarded_change for item in history), marker="o", color=COLORS[2], label=r"$\Delta \mathcal{P}G^R$")
            axes[1].semilogy(iteration, _positive(item.projected_green_lesser_change for item in history), marker="s", color=COLORS[3], label=r"$\Delta \mathcal{P}G^<$")
            axes[2].plot(iteration, [item.projected_occupation for item in history], marker="o", color=COLORS[0], label=r"$\bar{n}_{\mathcal{P}}$")
            hartree = np.asarray([np.nan if item.sigma_hartree_candidate is None else item.sigma_hartree_candidate for item in history])
            if np.any(np.isfinite(hartree)):
                axes[2].plot(iteration, hartree, marker="s", color=COLORS[1], label=r"$\Sigma_H/\Gamma$")
            resource_axis = axes[2].twinx()
            durations = np.asarray([np.nan if item.step_duration_seconds is None else item.step_duration_seconds for item in history])
            rss = np.asarray([np.nan if item.peak_rss_bytes is None else item.peak_rss_bytes / 1024**3 for item in history])
            if np.any(np.isfinite(durations)):
                resource_axis.plot(iteration, durations, ":", color=COLORS[4], label="step [s]")
            if np.any(np.isfinite(rss)):
                resource_axis.plot(iteration, rss, "--", color=COLORS[5], label="peak RSS [GiB]")
            resource_axis.set_ylabel("step time [s] / peak RSS [GiB]")
            lines, labels = axes[2].get_legend_handles_labels()
            more, more_labels = resource_axis.get_legend_handles_labels()
            axes[2].legend(lines + more, labels + more_labels, ncol=2)
        else:
            axes[0].text(0.5, 0.5, "No completed outer step", transform=axes[0].transAxes, ha="center")
        axes[0].set_ylabel("installed residual")
        axes[1].set_ylabel("projected-$G$ change")
        axes[2].set_ylabel("occupation / Hartree")
        axes[2].set_xlabel("outer SCBA step")
        if axes[0].get_legend_handles_labels()[0]:
            axes[0].legend(ncol=3)
        if axes[1].get_legend_handles_labels()[0]:
            axes[1].legend(ncol=2)
        axes[0].set_title(_title(config, "PS-SCBA"))
        _save(fig, path)


def publication_current(config: RunConfig, result: PSSCBAResult, lead: str, path: Path, *, physical: bool = True) -> None:
    raw_values = result.trajectory.currents.get(lead)
    corrected_values = result.trajectory.corrected_currents.get(lead)
    if raw_values is None:
        return
    time = np.asarray(result.trajectory.observable_time)
    if len(time) != len(raw_values):
        raise ValueError("observable_time and current arrays have different lengths.")
    with publication_style():
        fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
        plot_time = time_to_ps(time, config.model.gamma_eV) if physical else time
        convert = (lambda values: current_to_uA(values, config.model.gamma_eV)) if physical else (lambda values: values)
        color = COLORS[0 if lead == "L" else 1]
        ax.plot(plot_time, convert(raw_values), color=color, label=rf"$J_{lead}$ raw pole/Filon")
        if corrected_values is not None:
            ax.plot(
                plot_time,
                convert(corrected_values),
                color=COLORS[2],
                linestyle="--",
                label=rf"$J_{lead}$ conservation-corrected ($g=0$ diagnostic)",
            )
        _switch_markers(ax, config, physical=physical)
        ax.set_xlabel(r"$t\,[\mathrm{ps}]$" if physical else r"$t\,[\hbar/\Gamma]$")
        ax.set_ylabel(rf"$J_{lead}(t)\,[\mu\mathrm{{A}}]$" if physical else rf"$J_{lead}(t)\,[e\Gamma/\hbar]$")
        ax.set_title(_title(config, result.diagnostics.get("method", result.kernel.method)))
        ax.legend()
        _save(fig, path)


def _surface_subsample(time: np.ndarray, energy: np.ndarray, values: np.ndarray,
                       *, max_time: int = 900, max_energy: int = 4000,
                       energy_window: tuple[float, float] | None = None):
    """Return a bounded rendering view, optionally clipped in energy."""
    time = np.asarray(time, dtype=float)
    energy = np.asarray(energy, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(time), len(energy)):
        raise ValueError("Amplitude surface must be indexed as (time, energy).")
    if energy_window is not None:
        lower, upper = (float(value) for value in energy_window)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("energy_window must be a finite increasing pair.")
        mask = (energy >= lower) & (energy <= upper)
        if not np.any(mask):
            raise ValueError("energy_window does not overlap the amplitude grid.")
        energy = energy[mask]
        values = values[:, mask]
    ti = np.unique(np.linspace(0, len(time) - 1, min(max_time, len(time)), dtype=int))
    ei = np.unique(np.linspace(0, len(energy) - 1, min(max_energy, len(energy)), dtype=int))
    return time[ti], energy[ei], values[np.ix_(ti, ei)]


def publication_amplitude_surface(
    config: RunConfig,
    result: PSSCBAResult,
    values: np.ndarray,
    label: str,
    path: Path,
) -> None:
    """Render a production A²/C² energy--time image in natural units."""
    time, energy, surface = _surface_subsample(
        result.trajectory.time, result.trajectory.energy, values,
        energy_window=(-20.0, 20.0),
    )
    with publication_style():
        fig, ax = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
        image = ax.imshow(
            surface.T, origin="lower", aspect="auto", interpolation="nearest",
            extent=(float(time[0]), float(time[-1]), float(energy[0]), float(energy[-1])),
            cmap="viridis",
        )
        ax.set_xlabel(r"$t\,[\hbar/\Gamma]$")
        energy_label = r"$E'/\Gamma$" if label.startswith(r"C(") else r"$E/\Gamma$"
        ax.set_ylabel(energy_label)
        # Keep the requested comparison window exact when the computational
        # grid extends beyond it; smaller grids remain fully visible.
        ax.set_ylim(max(-20.0, float(energy[0])), min(20.0, float(energy[-1])))
        ax.set_title(rf"$|{label}|^2$; " + _title(config, "amplitude image"))
        fig.colorbar(image, ax=ax, pad=0.02, label=rf"$|{label}|^2$")
        _save(fig, path)


def write_amplitude_surfaces(config: RunConfig, result: PSSCBAResult, run_dir: Path) -> None:
    """Write the lead-resolved A² and interaction C² production figures."""
    output = run_dir / "plots" / "publication"
    output.mkdir(parents=True, exist_ok=True)
    for lead, values in result.trajectory.amplitude_a_squared.items():
        publication_amplitude_surface(
            config, result, values, rf"A_{lead}(t,E)", output / f"amplitude_A2_{lead}.svg"
        )
    branch_products = result.trajectory.amplitude_c_squared_by_frequency
    if branch_products:
        for frequency, values in sorted(branch_products.items()):
            sign = "minus" if frequency < 0.0 else "plus"
            weight = (
                float(result.kernel.phonon_occupation) + 1.0
                if frequency < 0.0
                else float(result.kernel.phonon_occupation)
            )
            publication_amplitude_surface(
                config,
                result,
                values,
                rf"C(t,{frequency:g},E') [w={weight:.3g}]",
                output / f"amplitude_C2_{sign}_omega.svg",
            )
    elif result.trajectory.amplitude_c_squared is not None:
        publication_amplitude_surface(
            config, result, result.trajectory.amplitude_c_squared,
            r"C(t,\omega_0,E')",
            output / "amplitude_C2.svg",
        )


def write_qa_plots(config: RunConfig, result: PSSCBAResult, run_dir: Path, *, progress=None) -> None:
    if not config.plotting.enabled:
        return
    if progress is not None:
        progress("qa_rendering", {"message": "rendering QA and publication SVGs"})
    qa = run_dir / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    plot_outer_convergence(config, result.history, qa / "01_outer_convergence.svg")
    plot_outer_convergence(config, result.history, run_dir / "plots" / "publication" / "outer_convergence.svg")

    with publication_style():
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
        energy = result.kernel.energy
        for index, (kernel, label) in enumerate(((result.initial_kernel, "initial"), (result.kernel, "final"))):
            if kernel is None:
                continue
            axes[0].plot(energy, kernel.sigma_retarded.real, label=label, color=COLORS[index])
            axes[1].plot(energy, -kernel.sigma_retarded.imag, label=label, color=COLORS[index])
            axes[2].plot(energy, kernel.sigma_lesser.imag, label=label, color=COLORS[index])
        axes[0].set_ylabel(r"$\mathrm{Re}\,\Sigma^R/\Gamma$")
        axes[1].set_ylabel(r"$-\mathrm{Im}\,\Sigma^R/\Gamma$")
        axes[2].set_ylabel(r"$\mathrm{Im}\,\Sigma^</\Gamma$")
        axes[2].set_xlabel(r"$\omega/\Gamma$")
        axes[0].legend()
        axes[0].set_title(_title(config, result.kernel.method))
        _save(fig, qa / "02_initial_final_kernel.svg")

    # MPM is a diagnostic, not a production acceptance gate.  In particular,
    # the final installed kernel may be a slightly updated kernel for which a
    # fresh diagnostic fit fails even though the fixed-point trajectory and
    # currents have already passed their numerical gates.  Prefer the latest
    # checkpoint basis (the same fail-closed reuse mechanism used by the
    # solver), but never let a plotting-only fit invalidate a completed run.
    system = make_system(config)
    basis = load_pole_basis(run_dir / "checkpoint")
    cache = None
    mpm_status: dict[str, object] = {
        "status": "unavailable",
        "source": "plot-only diagnostic",
        "pole_basis_available": basis is not None,
    }
    try:
        cache = build_pole_cache(system, result.kernel, previous_basis=basis)
        mpm_status.update({
            "status": "available",
            "fit_method": cache.fit_method,
            "fit_terms": cache.fit_terms,
            "pole_basis_reused": bool(cache.pole_basis_reused),
        })
    except Exception as exc:  # pragma: no cover - exercised through regression test
        mpm_status["reason"] = f"{type(exc).__name__}: {exc}"
    atomic_json(qa / "03_mpm_reconstruction.json", mpm_status)

    with publication_style():
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
        energy = result.kernel.energy
        target = result.kernel.sigma_dynamic_retarded
        axes[0].plot(energy, target.real, label="grid", color=COLORS[0])
        axes[1].plot(energy, -target.imag, color=COLORS[0], label="grid")
        if cache is not None:
            fitted = sigma_mpm(energy + 1j * system.ETA, cache.zeta, cache.weights)
            axes[0].plot(energy, fitted.real, "--", label="MPM", color=COLORS[1])
            axes[1].plot(energy, -fitted.imag, "--", label="MPM", color=COLORS[1])
            axes[2].semilogy(energy, _positive(np.abs(target - fitted)))
            axes[2].set_ylabel("absolute error")
        else:
            axes[0].text(
                0.02, 0.86, "plot-only MPM fit unavailable; run remains valid",
                transform=axes[0].transAxes, fontsize=9, color=COLORS[1],
            )
            axes[2].text(
                0.5, 0.5, "MPM diagnostic not evaluated",
                transform=axes[2].transAxes, ha="center", va="center",
            )
            axes[2].set_ylabel("absolute error (N/A)")
        axes[0].set_ylabel(r"$\mathrm{Re}\,\Sigma_{\rm dyn}^R/\Gamma$")
        axes[1].set_ylabel(r"$-\mathrm{Im}\,\Sigma_{\rm dyn}^R/\Gamma$")
        axes[2].set_xlabel(r"$\omega/\Gamma$")
        axes[0].legend()
        axes[1].legend()
        _save(fig, qa / "03_mpm_reconstruction.svg")

    currents = result.trajectory.currents
    corrected_currents = result.trajectory.corrected_currents
    current_time_natural = np.asarray(result.trajectory.observable_time)
    physical_qa = config.plotting.physical_qa_units
    current_time = time_to_ps(current_time_natural, config.model.gamma_eV) if physical_qa else current_time_natural
    plotted_currents = {
        lead: current_to_uA(values, config.model.gamma_eV) if physical_qa else values
        for lead, values in currents.items()
    }
    with publication_style():
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        for index, (lead, values) in enumerate(plotted_currents.items()):
            ax.plot(current_time, values, label=rf"$J_{lead}$ raw pole/Filon", color=COLORS[index])
            corrected = corrected_currents.get(lead)
            if corrected is not None:
                corrected_plot = current_to_uA(corrected, config.model.gamma_eV) if physical_qa else corrected
                ax.plot(
                    current_time,
                    corrected_plot,
                    label=rf"$J_{lead}$ conservation-corrected ($g=0$ diagnostic)",
                    color=COLORS[2],
                    linestyle="--",
                )
        if "L" in plotted_currents and "R" in plotted_currents:
            ax.plot(current_time, 0.5 * (plotted_currents["L"] - plotted_currents["R"]), ":", label="transport", color=COLORS[2])
        _switch_markers(ax, config, physical=physical_qa)
        ax.set_xlabel(r"$t\,[\mathrm{ps}]$" if physical_qa else r"$t\,[\hbar/\Gamma]$")
        ax.set_ylabel(r"$J_\alpha(t)\,[\mu\mathrm{A}]$" if physical_qa else r"$J_\alpha(t)\,[e\Gamma/\hbar]$")
        ax.legend()
        ax.set_title(_title(config, result.kernel.method))
        _save(fig, qa / "04_current.svg")

    with publication_style():
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
        occupation = result.trajectory.occupation
        if occupation is not None:
            axes[0].plot(current_time, occupation, color=COLORS[0])
            axes[0].axhline(result.trajectory.projected.occupation, linestyle="--", color=COLORS[1])
            axes[1].plot(current_time, occupation - result.trajectory.projected.occupation, color=COLORS[2])
        axes[0].set_ylabel(r"$N_D$")
        axes[1].set_ylabel(r"$N_D-\bar{n}_{\mathcal{P}}$")
        axes[1].set_xlabel(r"$t\,[\mathrm{ps}]$" if physical_qa else r"$t\,[\hbar/\Gamma]$")
        _save(fig, qa / "05_occupation.svg")

    with publication_style():
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
        residual = result.trajectory.continuity_residual
        collision = result.trajectory.collision_source
        if residual is not None:
            plotted_residual = current_to_uA(residual, config.model.gamma_eV) if physical_qa else residual
            axes[0].plot(current_time, plotted_residual, label="continuity", color=COLORS[0])
            if collision is not None:
                plotted_collision = current_to_uA(collision, config.model.gamma_eV) if physical_qa else collision
                axes[0].plot(current_time, plotted_collision, "--", label="discarded collision", color=COLORS[1])
            finite_collision = result.trajectory.collision_source_finite
            if finite_collision is not None:
                plotted_finite = current_to_uA(finite_collision, config.model.gamma_eV) if physical_qa else finite_collision
                axes[0].plot(current_time, plotted_finite, ":", label="finite-window collision", color=COLORS[2])
            prehistory_collision = result.trajectory.collision_source_prehistory
            if prehistory_collision is not None:
                plotted_prehistory = current_to_uA(prehistory_collision, config.model.gamma_eV) if physical_qa else prehistory_collision
                axes[0].plot(current_time, plotted_prehistory, "-.", label="prehistory tail", color=COLORS[3])
            if collision is not None:
                axes[1].plot(
                    current_time,
                    plotted_residual - plotted_collision,
                    label="continuity - discarded collision",
                    color=COLORS[2],
                )
        axes[0].legend()
        axes[1].legend()
        axes[0].set_ylabel(r"source $[\mu\mathrm{A}]$" if physical_qa else r"source $[e\Gamma/\hbar]$")
        axes[1].set_ylabel("identity mismatch")
        axes[1].set_xlabel(r"$t\,[\mathrm{ps}]$" if physical_qa else r"$t\,[\hbar/\Gamma]$")
        _save(fig, qa / "06_collision_continuity.svg")

    with publication_style():
        fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
        values = [result.trajectory.projected.r_green_retarded, result.trajectory.projected.r_green_lesser, result.trajectory.diagnostics.get("r_sigma_retarded", np.nan), result.trajectory.diagnostics.get("r_sigma_lesser", np.nan)]
        ax.bar([r"$R_G^R$", r"$R_G^<$", r"$R_\Sigma^R$", r"$R_\Sigma^<$"], _positive(values), color=COLORS[:4])
        ax.set_yscale("log")
        ax.set_ylabel("weighted Frobenius diagnostic")
        _save(fig, qa / "07_discarded_sector.svg")

    with publication_style():
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        q_r = result.trajectory.projected.q_green_retarded_time
        q_l = result.trajectory.projected.q_green_lesser_time
        if np.any(np.isfinite(q_r)) or np.any(np.isfinite(q_l)):
            q_time = time_to_ps(result.trajectory.time, config.model.gamma_eV) if physical_qa else result.trajectory.time
            ax.plot(q_time, q_r, label=r"$q_G^R$", color=COLORS[0])
            ax.plot(q_time, q_l, label=r"$q_G^<$", color=COLORS[1])
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Time-resolved QG is unavailable in this legacy run", transform=ax.transAxes, ha="center", va="center")
        ax.set_xlabel(r"$T\,[\mathrm{ps}]$" if physical_qa else r"$T\,[\hbar/\Gamma]$")
        ax.set_ylabel("nonstationary norm")
        _save(fig, qa / "08_time_nonstationarity.svg")

    with publication_style():
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        for index, (lead, values) in enumerate(plotted_currents.items()):
            ax.plot(current_time, values - values[-1], label=rf"$J_{lead}$ raw pole/Filon", color=COLORS[index])
            corrected = corrected_currents.get(lead)
            if corrected is not None:
                corrected_plot = current_to_uA(corrected, config.model.gamma_eV) if physical_qa else corrected
                ax.plot(
                    current_time,
                    corrected_plot - corrected_plot[-1],
                    label=rf"$J_{lead}$ conservation-corrected ($g=0$ diagnostic)",
                    color=COLORS[2],
                    linestyle="--",
                )
        ax.set_xlabel(r"$t\,[\mathrm{ps}]$" if physical_qa else r"$t\,[\hbar/\Gamma]$")
        ax.set_ylabel(r"$J(t)-J(t_{\max})\,[\mu\mathrm{A}]$" if physical_qa else r"$J(t)-J(t_{\max})\,[e\Gamma/\hbar]$")
        ax.legend()
        _save(fig, qa / "09_endpoint_asymptotic.svg")

    for lead in ("L", "R"):
        if config.plotting.natural_units:
            publication_current(
                config, result, lead,
                run_dir / "plots" / "publication" / f"current_{lead}_natural.svg",
                physical=False,
            )
        if config.plotting.physical_publication_figures:
            publication_current(config, result, lead, run_dir / "plots" / "publication" / f"current_{lead}_physical.svg", physical=True)
    write_amplitude_surfaces(config, result, run_dir)


def plot_run(run_directory: str | Path) -> list[Path]:
    """Regenerate every plot supported by saved products of complete or partial runs."""
    run_dir = Path(run_directory)
    config = load_config(run_dir / "resolved.yaml", allow_read_only_legacy=True)
    history_path = run_dir / "outer_history.json"
    history = tuple(OuterIteration(**item) for item in json.loads(history_path.read_text(encoding="utf-8"))) if history_path.exists() else ()
    live = run_dir / "plots" / "live" / "outer_convergence.svg"
    publication = run_dir / "plots" / "publication" / "outer_convergence.svg"
    plot_outer_convergence(config, history, live)
    plot_outer_convergence(config, history, publication)
    generated = [live, publication]
    data = run_dir / "data"
    observable = data / "observable_time.npy"
    loaded_time: np.ndarray | None = None
    loaded_currents: dict[str, np.ndarray] = {}
    if observable.exists():
        # Serialized observables follow the original runner convention
        # (ps/µA).  The plotting and numerical result objects remain natural
        # units, so convert at this boundary using the run manifest.
        loaded_time, loaded_currents, loaded_corrected_currents = load_saved_current_views(
            run_dir, config
        )
        time = loaded_time
        method = "preparation_fixed" if "preparation_fixed" in run_dir.name else "psscba"
        for index, lead in enumerate(("L", "R")):
            values = loaded_currents.get(lead)
            if values is None:
                continue
            if config.plotting.natural_units:
                with publication_style():
                    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
                    ax.plot(
                        time,
                        values,
                        color=COLORS[index],
                        label=rf"$J_{lead}$ raw pole/Filon",
                    )
                    corrected = loaded_corrected_currents.get(lead)
                    if corrected is not None:
                        ax.plot(
                            time,
                            corrected,
                            color=COLORS[2],
                            linestyle="--",
                            label=rf"$J_{lead}$ conservation-corrected ($g=0$ diagnostic)",
                        )
                    _switch_markers(ax, config)
                    ax.set_xlabel(r"$t\,[\hbar/\Gamma]$")
                    ax.set_ylabel(rf"$J_{lead}(t)\,[e\Gamma/\hbar]$")
                    ax.set_title(_title(config, method))
                    ax.legend()
                    path = run_dir / "plots" / "publication" / f"current_{lead}_natural.svg"
                    _save(fig, path)
                    generated.append(path)
            if config.plotting.physical_publication_figures:
                with publication_style():
                    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
                    ax.plot(
                        time_to_ps(time, config.model.gamma_eV),
                        current_to_uA(values, config.model.gamma_eV),
                        color=COLORS[index], label=rf"$J_{lead}$ raw pole/Filon",
                    )
                    corrected = loaded_corrected_currents.get(lead)
                    if corrected is not None:
                        ax.plot(
                            time_to_ps(time, config.model.gamma_eV),
                            current_to_uA(corrected, config.model.gamma_eV),
                            color=COLORS[2],
                            linestyle="--",
                            label=rf"$J_{lead}$ conservation-corrected ($g=0$ diagnostic)",
                        )
                    _switch_markers(ax, config, physical=True)
                    ax.set_xlabel(r"$t\,[\mathrm{ps}]$")
                    ax.set_ylabel(rf"$J_{lead}(t)\,[\mu\mathrm{{A}}]$")
                    ax.set_title(_title(config, method))
                    ax.legend()
                    physical_path = run_dir / "plots" / "publication" / f"current_{lead}_physical.svg"
                    _save(fig, physical_path)
                    generated.append(physical_path)
    required = (
        "projection_lag.npy", "projected_green_retarded_time.npy",
        "projected_green_lesser_time.npy", "projected_green_retarded_energy.npy",
        "projected_green_lesser_energy.npy", "reconstruction_energy.npy", "projection_time.npy",
    )
    if (
        loaded_time is not None
        and (run_dir / "final_kernel" / "kernel.json").exists()
        and all((data / name).exists() for name in required)
    ):
        diagnostics_path = run_dir / "diagnostics.json"
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8")) if diagnostics_path.exists() else {}
        lag = np.load(data / "projection_lag.npy", allow_pickle=False)
        gr = np.load(data / "projected_green_retarded_time.npy", allow_pickle=False)
        gl = np.load(data / "projected_green_lesser_time.npy", allow_pickle=False)
        projection_time = np.load(data / "projection_time.npy", allow_pickle=False)
        q_r_path = data / "q_green_retarded_time.npy"
        q_l_path = data / "q_green_lesser_time.npy"
        projection = ProjectionResult(
            lag=lag, green_retarded=gr, green_lesser=gl,
            green_greater=gl + gr - np.conjugate(gr),
            energy=np.load(data / "reconstruction_energy.npy", allow_pickle=False),
            green_retarded_energy=np.load(data / "projected_green_retarded_energy.npy", allow_pickle=False),
            green_lesser_energy=np.load(data / "projected_green_lesser_energy.npy", allow_pickle=False),
            occupation=float(diagnostics.get("projected_occupation", 0.0)),
            r_green_retarded=float(diagnostics.get("r_green_retarded", 0.0)),
            r_green_lesser=float(diagnostics.get("r_green_lesser", 0.0)),
            q_green_retarded_time=np.load(q_r_path, allow_pickle=False) if q_r_path.exists() else np.full(len(projection_time), np.nan),
            q_green_lesser_time=np.load(q_l_path, allow_pickle=False) if q_l_path.exists() else np.full(len(projection_time), np.nan),
            diagnostics=diagnostics,
        )
        currents = loaded_currents
        optional = lambda name: np.load(data / f"{name}.npy", allow_pickle=False) if (data / f"{name}.npy").exists() else None
        trajectory = FixedKernelTrajectory(
            time=projection_time,
            energy=np.load(data / "reconstruction_energy.npy", allow_pickle=False),
            projected=projection,
            observable_time=loaded_time if loaded_time is not None else np.asarray([], dtype=float),
            currents=currents,
            corrected_currents=loaded_corrected_currents,
            occupation=optional("occupation"),
            continuity_residual=optional("continuity_residual"),
            collision_source=optional("collision_source"),
            collision_source_finite=optional("collision_source_finite"),
            collision_source_prehistory=optional("collision_source_prehistory"),
            amplitude_a_squared={
                lead: optional(f"amplitude_A2_{lead}")
                for lead in ("L", "R")
                if optional(f"amplitude_A2_{lead}") is not None
            },
            amplitude_c_squared=optional("amplitude_C2"),
            amplitude_c_squared_by_frequency={
                frequency: values
                for frequency, values in (
                    (-abs(float(diagnostics.get("amplitude_c_frequency", 0.0))), optional("amplitude_C2_minus_omega")),
                    (abs(float(diagnostics.get("amplitude_c_frequency", 0.0))), optional("amplitude_C2_plus_omega")),
                )
                if values is not None and abs(frequency) > 0.0
            },
            diagnostics=diagnostics,
        )
        kernel = load_kernel(run_dir / "final_kernel")
        initial = load_kernel(run_dir / "initial_kernel") if (run_dir / "initial_kernel" / "kernel.json").exists() else None
        result = PSSCBAResult(True, kernel, trajectory, tuple(history), initial_kernel=initial, diagnostics=diagnostics)
        write_qa_plots(config, result, run_dir)
        generated.extend(sorted((run_dir / "qa").glob("*.svg")))
    return generated
