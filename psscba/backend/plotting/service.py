"""Deterministic plotting service used by runs and campaign aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from psscba.backend.plotting.qa import publication_style, save_svg, write_qa_plots
from psscba.frontend.configuration import CaseConfig, RunConfig
from psscba.types import PSSCBAResult


@dataclass(frozen=True)
class RunPlotter:
    config: CaseConfig | RunConfig

    def write(self, result: PSSCBAResult, run_dir: str | Path, *, progress=None) -> None:
        write_qa_plots(self.config, result, Path(run_dir), progress=progress)


def write_aggregate_heatmaps(
    records: Sequence[Mapping[str, object]],
    output_directory: str | Path,
) -> list[Path]:
    metrics = (
        "r_green_retarded",
        "r_green_lesser",
        "r_sigma_retarded",
        "r_sigma_lesser",
        "max_continuity_residual",
        "collision_identity_scaled_mismatch",
        "collision_source_max",
        "current_scale",
    )
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    ps_records = [item for item in records if item.get("method") == "psscba"]
    for protocol in sorted({str(item["protocol"]) for item in ps_records}):
        frequencies = sorted({
            float(item["phonon_energy"])
            for item in ps_records
            if item["protocol"] == protocol
        })
        for frequency in frequencies:
            selected = [
                item for item in ps_records
                if item["protocol"] == protocol
                and float(item["phonon_energy"]) == frequency
            ]
            bandwidths = sorted({float(item["bandwidth"]) for item in selected})
            couplings = sorted({float(item["coupling_meV"]) for item in selected})
            for metric in metrics:
                values = np.full((len(bandwidths), len(couplings)), np.nan)
                for item in selected:
                    if metric in item:
                        row = bandwidths.index(float(item["bandwidth"]))
                        column = couplings.index(float(item["coupling_meV"]))
                        values[row, column] = float(item[metric])
                with publication_style():
                    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
                    image = ax.imshow(values, origin="lower", aspect="auto")
                    ax.set_xticks(range(len(couplings)), labels=[f"{value:g}" for value in couplings])
                    ax.set_yticks(range(len(bandwidths)), labels=[f"{value:g}" for value in bandwidths])
                    ax.set_xlabel(r"$g\,[\mathrm{meV}]$")
                    ax.set_ylabel(r"$W/\Gamma$")
                    ax.set_title(rf"{protocol}; $\omega_0/\Gamma={frequency:g}$; {metric}")
                    fig.colorbar(image, ax=ax)
                    path = output_directory / f"{protocol}_w{frequency:g}_{metric}.svg"
                    save_svg(fig, path)
                    paths.append(path)
    return paths
