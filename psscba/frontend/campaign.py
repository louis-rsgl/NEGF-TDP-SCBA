from __future__ import annotations

from dataclasses import replace
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from psscba.frontend.configuration import (
    ModelConfig,
    ProtocolConfig,
    ProjectionConfig,
    RunConfig,
    load_config,
    require_shared_production_grid,
)
from psscba.acceptance import enforce_acceptance
from psscba.frontend.storage import (
    atomic_json,
    checkpoint_writer,
    create_run_directory,
    load_checkpoint,
    load_pole_basis,
    save_result,
)
from psscba.backend.solver import solve_preparation_fixed, solve_psscba


class CampaignExecutionError(RuntimeError):
    """Raised after all jobs finish when one or more isolated jobs failed."""


def case_config(
    base: RunConfig,
    protocol: str,
    bandwidth: float,
    coupling: float,
    phonon_energy: float | None = None,
) -> RunConfig:
    duration = base.campaign.square_duration if protocol == "square" else None
    projection = base.projection
    model = replace(
        base.model,
        bandwidth=float(bandwidth),
        coupling_meV=float(coupling),
        phonon_energy=(
            base.model.phonon_energy if phonon_energy is None else float(phonon_energy)
        ),
    )
    # The campaign configuration owns the reconstruction grid. Do not
    # silently switch grids based on bandwidth: a sweep must use one declared
    # time/energy discretization so bandwidth comparisons are numerical
    # comparisons rather than comparisons of discretization error.
    numerics = base.numerics
    requested_time_max = base.campaign.time_max_by_protocol.get(protocol)
    if requested_time_max is not None:
        intervals = int(np.ceil(
            (float(requested_time_max) - projection.time_min) / projection.time_step
        ))
        projection = replace(
            projection,
            time_max=projection.time_min + intervals * projection.time_step,
        )
    elif protocol == "square":
        requested_max = float(duration) + 96.0
        intervals = int(np.ceil((requested_max - projection.time_min) / projection.time_step))
        projection = replace(
            projection,
            time_max=projection.time_min + intervals * projection.time_step,
        )
    config = replace(
        base,
        model=model,
        protocol=ProtocolConfig(name=protocol, duration=duration),
        projection=projection,
        numerics=numerics,
    )
    config.validate()
    return config


def campaign_cases(config: RunConfig) -> Iterable[RunConfig]:
    for protocol in config.campaign.protocols:
        for bandwidth in config.campaign.bandwidths:
            for coupling in config.campaign.couplings_meV:
                for phonon_energy in config.campaign.phonon_energies:
                    yield case_config(
                        config, protocol, bandwidth, coupling, phonon_energy
                    )


def run_case(
    config: RunConfig,
    *,
    preparation_fixed: bool = False,
    run_dir: str | Path | None = None,
    blas_threads: int | None = None,
) -> Path:
    from psscba.backend.plotting import RunPlotter
    from threadpoolctl import threadpool_limits

    blas_limit = threadpool_limits(
        limits=config.campaign.blas_threads if blas_threads is None else blas_threads,
        user_api="blas",
    )

    method = "preparation_fixed" if preparation_fixed else "psscba"
    label = (
        f"{method}_{config.protocol.name}_W{config.model.bandwidth:g}_"
        f"w{config.model.phonon_energy:g}_g{config.model.coupling_meV:g}meV"
    )
    run_dir = (
        create_run_directory(config, label=label)
        if run_dir is None
        else Path(run_dir)
    )
    # Existing/restarted directories may predate the storage-unit manifest.
    # Write it before any result or plotting product is produced.
    from psscba.frontend.storage import write_units_metadata
    write_units_metadata(run_dir, config)
    from psscba.frontend.tracking import RunTracker

    tracker = RunTracker(run_dir, config)
    try:
        atomic_json(run_dir / "status.json", {"status": "running", "iteration": 0})
        with tracker:
            if preparation_fixed:
                result = solve_preparation_fixed(config, progress=tracker.callback)
            else:
                result = solve_psscba(
                    config,
                    checkpoint=checkpoint_writer(run_dir),
                    progress=tracker.callback,
                    step_completed=tracker.completed_step,
                )
            save_result(run_dir, result)
            RunPlotter(config).write(result, run_dir, progress=tracker.callback)
            if not preparation_fixed and config.numerics.enforce_hard_gates:
                enforce_acceptance(config, result)
            tracker.state = "complete"
    except (KeyboardInterrupt, SystemExit) as exc:
        atomic_json(run_dir / "status.json", {
            "status": "interrupted", "phase": tracker.phase,
            "iteration": tracker.outer_step, "resumable": (run_dir / "checkpoint" / "kernel.json").exists(),
        })
        raise
    except Exception as exc:
        atomic_json(
            run_dir / "failure.json",
            {"type": type(exc).__name__, "message": str(exc), "phase": tracker.phase},
        )
        atomic_json(run_dir / "status.json", {"status": "failed", "phase": tracker.phase})
        raise
    del blas_limit
    return run_dir


def resume_case(
    run_directory: str | Path,
    *,
    validation_session: str | Path | None = None,
    blas_threads: int | None = None,
) -> Path:
    """Resume one compatible job from its latest installed-kernel checkpoint."""
    from psscba.backend.plotting import RunPlotter
    from psscba.frontend.tracking import RunTracker
    from threadpoolctl import threadpool_limits

    run_dir = Path(run_directory)
    config, kernel, history = load_checkpoint(run_dir)
    basis = load_pole_basis(run_dir / "checkpoint")
    from psscba.frontend.storage import write_units_metadata
    write_units_metadata(run_dir, config)
    limits = threadpool_limits(
        limits=config.campaign.blas_threads if blas_threads is None else blas_threads,
        user_api="blas",
    )
    tracker = RunTracker(run_dir, config, history)
    if validation_session is not None:
        from psscba.frontend.validation import update_validation_run_status

        update_validation_run_status(validation_session, run_dir, "running")
    try:
        atomic_json(run_dir / "status.json", {
            "status": "running",
            "iteration": kernel.iteration,
            "resumed": True,
        })
        with tracker:
            result = solve_psscba(
                config.case,
                initial_kernel=kernel,
                initial_history=history,
                initial_basis=basis,
                checkpoint=checkpoint_writer(run_dir),
                progress=tracker.callback,
                step_completed=tracker.completed_step,
            )
            save_result(run_dir, result)
            RunPlotter(config).write(result, run_dir, progress=tracker.callback)
            if config.numerics.enforce_hard_gates:
                enforce_acceptance(config, result)
            tracker.state = "complete"
            if validation_session is not None:
                update_validation_run_status(validation_session, run_dir, "accepted")
    except (KeyboardInterrupt, SystemExit):
        atomic_json(run_dir / "status.json", {
            "status": "interrupted",
            "phase": tracker.phase,
            "iteration": tracker.outer_step,
            "resumable": True,
        })
        if validation_session is not None:
            update_validation_run_status(validation_session, run_dir, "interrupted")
        raise
    except Exception as exc:
        atomic_json(run_dir / "failure.json", {
            "type": type(exc).__name__,
            "message": str(exc),
            "phase": tracker.phase,
        })
        atomic_json(run_dir / "status.json", {"status": "failed", "phase": tracker.phase})
        if validation_session is not None:
            update_validation_run_status(validation_session, run_dir, "failed")
        raise
    finally:
        del limits
    return run_dir


def _resume_campaign_job(run_dir: Path, method: str, blas_threads: int) -> Path:
    checkpoint = run_dir / "checkpoint" / "kernel.json"
    if method == "psscba" and checkpoint.exists():
        return resume_case(run_dir, blas_threads=blas_threads)
    config = load_config(run_dir / "resolved.yaml")
    return run_case(
        config,
        preparation_fixed=method == "preparation_fixed",
        run_dir=run_dir,
        blas_threads=blas_threads,
    )


def resume_campaign(campaign_directory: str | Path) -> list[Path]:
    """Resume incomplete compatible jobs while preserving accepted results."""
    from psscba.frontend.dashboard import CampaignDashboard
    campaign_dir = Path(campaign_directory)
    manifest_path = campaign_dir / "campaign.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from psscba.frontend.configuration import CONFIG_FORMAT

    if str(manifest.get("config_format")) != CONFIG_FORMAT:
        raise ValueError("This campaign uses a retired configuration format and cannot be resumed.")
    dashboard = CampaignDashboard(campaign_dir, manifest)
    incomplete = []
    outputs = []
    for job in dashboard.jobs:
        run_dir = Path(job["run_directory"])
        outputs.append(run_dir)
        snapshot_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8")).get("status")
        if snapshot_status != "complete":
            incomplete.append((run_dir, str(job.get("method", "psscba")), str(job["case_id"])))
    if not incomplete:
        dashboard.refresh(reason="resume requested; all jobs already complete")
        dashboard.finish()
        return outputs
    sample_config = load_config(incomplete[0][0] / "resolved.yaml")
    workers = int(manifest.get("max_workers", sample_config.campaign.max_workers))
    threads = int(manifest.get("blas_threads", sample_config.campaign.blas_threads))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_resume_campaign_job, run_dir, method, threads): (case_id, run_dir)
            for run_dir, method, case_id in incomplete
        }
        pending = set(futures)
        while pending:
            completed, pending = wait(
                pending,
                timeout=min(
                    sample_config.tracking.heartbeat_seconds,
                    sample_config.tracking.live_refresh_seconds,
                ),
                return_when=FIRST_COMPLETED,
            )
            for future in completed:
                case_id, _ = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    dashboard.update_status(case_id, "failed", error=str(exc))
                else:
                    dashboard.update_status(case_id, "complete")
            dashboard.refresh(reason="campaign resume heartbeat")
    dashboard.finish()
    failed = [job for job in dashboard.jobs if job.get("status") == "failed"]
    if failed:
        raise CampaignExecutionError(
            f"{len(failed)} resumed campaign jobs failed; inspect {campaign_dir / 'main.log'}."
        )
    return outputs


def run_campaign(
    config: RunConfig,
    *,
    campaign_directory: str | Path | None = None,
) -> list[Path]:
    from psscba.frontend.dashboard import CampaignDashboard
    from psscba.frontend.provenance import software_provenance

    require_shared_production_grid(config)
    jobs: list[tuple[RunConfig, bool, str, Path]] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign_dir = (
        Path(config.output_root) / f"sweep_{stamp}"
        if campaign_directory is None
        else Path(campaign_directory)
    )
    job_root = campaign_dir / "jobs"
    job_root.mkdir(parents=True, exist_ok=False)
    for case in campaign_cases(config):
        baselines = (False, True) if config.campaign.compare_preparation_fixed else (False,)
        for baseline in baselines:
            method = "preparation_fixed" if baseline else "psscba"
            case_id = (
                f"{method}_{case.protocol.name}_W{case.model.bandwidth:g}_"
                f"w{case.model.phonon_energy:g}_g{case.model.coupling_meV:g}meV"
            )
            run_dir = create_run_directory(
                case,
                label=case_id,
                root=job_root,
                exact_name=True,
            )
            jobs.append((case, baseline, case_id, run_dir))
    manifest = {
        "config_format": config.config_format,
        "kind": "sweep",
        "software": software_provenance()["software"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "max_workers": config.campaign.max_workers,
        "blas_threads": config.campaign.blas_threads,
        "jobs": [
            {
                "case_id": case_id,
                "run_directory": str(run_dir.resolve()),
                "status": "created",
                "protocol": case.protocol.name,
                "bandwidth": case.model.bandwidth,
                "phonon_energy": case.model.phonon_energy,
                "coupling_meV": case.model.coupling_meV,
                "method": "preparation_fixed" if baseline else "psscba",
            }
            for case, baseline, case_id, run_dir in jobs
        ],
    }
    atomic_json(campaign_dir / "campaign.json", manifest)
    dashboard = CampaignDashboard(campaign_dir, manifest)
    dashboard.start()
    outputs = [run_dir for _, _, _, run_dir in jobs]
    with ProcessPoolExecutor(max_workers=config.campaign.max_workers) as executor:
        futures = {}
        for case, baseline, case_id, run_dir in jobs:
            dashboard.update_status(case_id, "queued")
            future = executor.submit(
                run_case,
                case,
                preparation_fixed=baseline,
                run_dir=run_dir,
                blas_threads=config.campaign.blas_threads,
            )
            futures[future] = (case_id, run_dir)
        pending = set(futures)
        while pending:
            completed, pending = wait(
                pending,
                timeout=min(
                    config.tracking.heartbeat_seconds,
                    config.tracking.live_refresh_seconds,
                ),
                return_when=FIRST_COMPLETED,
            )
            for future in completed:
                case_id, _ = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    dashboard.update_status(case_id, "failed", error=str(exc))
                else:
                    dashboard.update_status(case_id, "complete")
            dashboard.refresh(reason="campaign heartbeat")
    dashboard.finish()
    failed = [job for job in dashboard.jobs if job.get("status") == "failed"]
    if failed:
        raise CampaignExecutionError(
            f"{len(failed)} of {len(jobs)} campaign jobs failed; "
            f"inspect {campaign_dir / 'main.log'} and jobs/<case-id>/solver.log."
        )
    return outputs


def aggregate(root: str | Path) -> Path:
    root = Path(root)
    records = []
    patterns = ("*/diagnostics.json", "jobs/*/diagnostics.json")
    diagnostics_paths = [path for pattern in patterns for path in root.glob(pattern)]
    for diagnostics_path in diagnostics_paths:
        run_dir = diagnostics_path.parent
        status_path = run_dir / "status.json"
        if not status_path.exists():
            continue
        with open(status_path, encoding="utf-8") as handle:
            status = json.load(handle)
        if status.get("status") != "complete":
            continue
        with open(diagnostics_path, encoding="utf-8") as handle:
            diagnostics = json.load(handle)
        with open(run_dir / "resolved.yaml", encoding="utf-8") as handle:
            import yaml

            resolved = yaml.safe_load(handle)
        records.append(
            {
                "run": run_dir.name,
                "protocol": resolved["protocol"]["name"],
                "bandwidth": resolved["model"]["bandwidth"],
                "coupling_meV": resolved["model"]["coupling_meV"],
                "phonon_energy": resolved["model"]["phonon_energy"],
                **diagnostics,
            }
        )
    output = root / "aggregate.json"
    atomic_json(output, records)
    qa = root / "aggregate_qa"
    from psscba.backend.plotting import write_aggregate_heatmaps

    write_aggregate_heatmaps(records, qa)
    return output
