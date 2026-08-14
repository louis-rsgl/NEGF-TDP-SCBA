from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from backend.units import current_to_uA, time_to_ps


def _load_run(run_dir: Path) -> tuple[dict, np.ndarray, dict[tuple[float, float], np.ndarray]]:
    """Load and validate the complete current grid saved by runner.py."""
    info_path = run_dir / "run_info.json"
    with info_path.open(encoding="utf-8") as stream:
        info = json.load(stream)

    times_ps = np.load(run_dir / "data" / "t_ps.npy")
    widths = np.asarray(info["W_GRID"], dtype=float)
    couplings = np.asarray(info["GQ_GRID"], dtype=float)[:-2]
    currents: dict[tuple[float, float], np.ndarray] = {}
    missing: list[Path] = []

    for coupling in couplings:
        for width in widths:
            path = run_dir / "data" / f"J_L_W{width:.3f}_gq{coupling:.3f}.npy"
            if not path.exists():
                missing.append(path)
                continue
            current = np.asarray(np.load(path), dtype=float)
            if current.shape != times_ps.shape:
                raise ValueError(
                    f"{path} has shape {current.shape}, expected {times_ps.shape}."
                )
            if not np.all(np.isfinite(current)):
                raise ValueError(f"{path} contains non-finite current values.")
            currents[(float(width), float(coupling))] = current

    if missing:
        names = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"The current sweep is incomplete; missing:\n{names}")
    return info, times_ps, currents


def _coupling_title(coupling: float, info: dict) -> str:
    units = info.get("GQ_GRID_UNITS", "Gamma")
    if units == "meV":
        gamma_values = np.asarray(info.get("GQ_GRID_GAMMA", []), dtype=float)
        input_values = np.asarray(info["GQ_GRID"], dtype=float)[:-1]
        matches = np.flatnonzero(np.isclose(input_values, coupling, rtol=0.0, atol=1e-12))
        if matches.size and gamma_values.size == input_values.size:
            resolved = gamma_values[matches[0]]
            return rf"$g={coupling:g}\,\mathrm{{meV}}$  ($g/\Gamma={resolved:g}$)"
        return rf"$g={coupling:g}\,\mathrm{{meV}}$"
    return rf"$g/\Gamma={coupling:g}$"


def _coupling_label(coupling: float, info: dict) -> str:
    """Compact coupling label for the shared bandwidth-panel legend."""
    units = info.get("GQ_GRID_UNITS", "Gamma")
    if units == "meV":
        return rf"$g={coupling:g}\,\mathrm{{meV}}$"
    return rf"$g/\Gamma={coupling:g}$"


def _add_turnoff_marker(
    ax: plt.Axes,
    *,
    info: dict,
    physical_units: bool,
    include_label: bool,
) -> None:
    if str(info.get("PULSE_PROTOCOL", "downward")) != "square":
        return
    duration = float(info["SQUARE_DURATION"])
    switch_time = (
        float(time_to_ps(duration, float(info["GAMMA"])))
        if physical_units else duration
    )
    label = (
        rf"turnoff $s={duration:g}\,\hbar/\Gamma$"
        if include_label else "_nolegend_"
    )
    ax.axvline(
        switch_time,
        color="0.2",
        linestyle=":",
        linewidth=1.6,
        label=label,
    )


def _decorate_axis(
    ax: plt.Axes,
    *,
    panel: int,
    ncols: int,
    nrows: int,
    physical_units: bool,
) -> None:
    ax.axhline(0.0, color="0.25", linewidth=0.8, alpha=0.55)
    ax.grid(alpha=0.25)
    ax.margins(x=0.0, y=0.08)
    ax.tick_params(labelsize=13)
    if panel % ncols == 0:
        if physical_units:
            ax.set_ylabel(r"Current $J_L(t)$ [$\mu$A]", fontsize=15)
        else:
            ax.set_ylabel(r"Current $J_L(t)$ [$e\Gamma/\hbar$]", fontsize=15)
    if panel // ncols == nrows - 1:
        if physical_units:
            ax.set_xlabel(r"Time $t$ [ps]", fontsize=15)
        else:
            ax.set_xlabel(r"Time $t$ [$\hbar/\Gamma$]", fontsize=15)


def _figure_title(info: dict, *, physical_units: bool, panels: str) -> str:
    protocol = str(info.get("PULSE_PROTOCOL", "downward")).capitalize()
    units = "physical units" if physical_units else "original Maciejko units"
    title = rf"{protocol} current: {units}; panels by {panels}"
    if str(info.get("PULSE_PROTOCOL", "downward")) == "square":
        title += rf", $s={float(info['SQUARE_DURATION']):g}\,\hbar/\Gamma$"
    return title


def _make_figure(
    *,
    info: dict,
    times: np.ndarray,
    currents: dict[tuple[float, float], np.ndarray],
    physical_units: bool,
) -> plt.Figure:
    widths = np.asarray(info["W_GRID"], dtype=float)
    couplings = np.asarray(info["GQ_GRID"], dtype=float)
    ncols = 2
    nrows = math.ceil(couplings.size / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(25.0, 6.0 * nrows),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )

    for panel, coupling in enumerate(couplings):
        ax = axes.flat[panel]
        for width in widths:
            current = currents[(float(width), float(coupling))]
            is_wbl = bool(np.isclose(width, np.max(widths)))
            style = "--" if is_wbl else "-"
            label = (
                rf"WBL ($W={width:g}\Gamma$)"
                if is_wbl
                else rf"$W={width:g}\Gamma$"
            )
            ax.plot(times, current, style, linewidth=2.2, label=label)

        _add_turnoff_marker(
            ax,
            info=info,
            physical_units=physical_units,
            include_label=panel == 0,
        )
        ax.set_title(_coupling_title(float(coupling), info), fontsize=17)
        _decorate_axis(
            ax,
            panel=panel,
            ncols=ncols,
            nrows=nrows,
            physical_units=physical_units,
        )

    for panel in range(couplings.size, axes.size):
        axes.flat[panel].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(
        _figure_title(info, physical_units=physical_units, panels="coupling"),
        fontsize=22,
    )
    fig.legend(
        handles,
        labels,
        loc="outside right center",
        ncols=1,
        frameon=True,
        fontsize=14,
    )
    return fig


def _make_width_figure(
    *,
    info: dict,
    times: np.ndarray,
    currents: dict[tuple[float, float], np.ndarray],
    physical_units: bool,
) -> plt.Figure:
    """Plot one panel per bandwidth and one current trace per coupling."""
    widths = np.asarray(info["W_GRID"], dtype=float)
    couplings = np.asarray(info["GQ_GRID"], dtype=float)
    ncols = 2
    nrows = math.ceil(widths.size / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(25.0, 6.0 * nrows),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )

    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, couplings.size))
    for panel, width in enumerate(widths):
        ax = axes.flat[panel]
        for color, coupling in zip(colors, couplings, strict=True):
            ax.plot(
                times,
                currents[(float(width), float(coupling))],
                color=color,
                linewidth=2.2,
                label=_coupling_label(float(coupling), info),
            )

        _add_turnoff_marker(
            ax,
            info=info,
            physical_units=physical_units,
            include_label=panel == 0,
        )
        is_wbl = bool(np.isclose(width, np.max(widths)))
        width_title = (
            rf"WBL ($W={width:g}\Gamma$)"
            if is_wbl else rf"$W={width:g}\Gamma$"
        )
        ax.set_title(width_title, fontsize=17)
        _decorate_axis(
            ax,
            panel=panel,
            ncols=ncols,
            nrows=nrows,
            physical_units=physical_units,
        )

    for panel in range(widths.size, axes.size):
        axes.flat[panel].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(
        _figure_title(info, physical_units=physical_units, panels="bandwidth"),
        fontsize=22,
    )
    fig.legend(
        handles,
        labels,
        loc="outside right center",
        ncols=1,
        frameon=True,
        fontsize=14,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one Maciejko-style bandwidth-comparison panel per phonon coupling "
            "and one coupling-comparison panel per bandwidth, in both physical and "
            "original dimensionless units."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="completed protocol-specific runner directory",
    )
    args = parser.parse_args()

    info, times_ps, currents_uA = _load_run(args.run_dir)
    gamma = float(info["GAMMA"])
    output_dir = args.run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    physical = _make_figure(
        info=info,
        times=times_ps,
        currents=currents_uA,
        physical_units=True,
    )
    physical_path = output_dir / "current_subplots_uA_ps.svg"
    physical.savefig(physical_path, format="svg", bbox_inches="tight")
    plt.close(physical)

    time_unit_ps = float(time_to_ps(1.0, gamma))
    current_unit_uA = float(current_to_uA(1.0, gamma))
    dimensionless = _make_figure(
        info=info,
        times=times_ps / time_unit_ps,
        currents={key: value / current_unit_uA for key, value in currents_uA.items()},
        physical_units=False,
    )
    dimensionless_path = output_dir / "current_subplots_original_units.svg"
    dimensionless.savefig(dimensionless_path, format="svg", bbox_inches="tight")
    plt.close(dimensionless)

    physical_by_width = _make_width_figure(
        info=info,
        times=times_ps,
        currents=currents_uA,
        physical_units=True,
    )
    physical_by_width_path = output_dir / "current_subplots_by_W_uA_ps.svg"
    physical_by_width.savefig(
        physical_by_width_path, format="svg", bbox_inches="tight"
    )
    plt.close(physical_by_width)

    dimensionless_by_width = _make_width_figure(
        info=info,
        times=times_ps / time_unit_ps,
        currents={key: value / current_unit_uA for key, value in currents_uA.items()},
        physical_units=False,
    )
    dimensionless_by_width_path = (
        output_dir / "current_subplots_by_W_original_units.svg"
    )
    dimensionless_by_width.savefig(
        dimensionless_by_width_path, format="svg", bbox_inches="tight"
    )
    plt.close(dimensionless_by_width)

    print(f"Saved {physical_path}")
    print(f"Saved {dimensionless_path}")
    print(f"Saved {physical_by_width_path}")
    print(f"Saved {dimensionless_by_width_path}")


if __name__ == "__main__":
    main()
