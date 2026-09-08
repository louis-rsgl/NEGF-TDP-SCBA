from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

from psscba.backend.projection.core import direct_toeplitz_projection, projector_errors
from psscba.frontend.campaign import aggregate, run_campaign, run_case
from psscba.frontend.configuration import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="psscba")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "validate", "profile", "sweep", "compare-preparation"):
        command = sub.add_parser(name)
        command.add_argument("config")
        if name == "validate":
            command.add_argument("--full", action="store_true")
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
    aggregate_command = sub.add_parser("aggregate")
    aggregate_command.add_argument("root")
    return parser


def synthetic_validation() -> dict[str, float]:
    time = np.linspace(-1.0, 1.0, 17)
    difference = time[:, None] - time[None, :]
    matrix = np.where(difference >= 0.0, np.exp(-0.3 * difference), 0.0)
    projected = direct_toeplitz_projection(matrix, lesser=False)
    stationary_error = float(np.linalg.norm(projected - matrix) / np.linalg.norm(matrix))
    idempotency, orthogonality = projector_errors(matrix, lesser=False)
    if max(stationary_error, idempotency, orthogonality) >= 1e-12:
        raise RuntimeError("Synthetic projector validation failed.")
    return {
        "stationary_error": stationary_error,
        "idempotency_error": idempotency,
        "orthogonality_error": orthogonality,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plot":
        from psscba.backend.plotting import plot_run

        for path in plot_run(args.run_directory):
            print(path)
        return 0
    if args.command == "status":
        path = Path(args.path)
        if path.is_dir() and (path / "campaign.json").exists():
            from psscba.frontend.dashboard import CampaignDashboard

            payload = json.loads((path / "campaign.json").read_text(encoding="utf-8"))
            CampaignDashboard(path, payload).refresh(reason="manual status refresh")
            print(path / "main.log")
            print(path / "dashboard.html")
            return 0
        progress = path / "progress.json" if path.is_dir() else path
        print(json.dumps(json.loads(progress.read_text(encoding="utf-8")), indent=2))
        return 0
    if args.command == "aggregate":
        print(aggregate(args.root))
        return 0
    if args.command == "resume":
        run_dir = Path(args.run_directory)
        from psscba.frontend.campaign import resume_campaign, resume_case

        if (run_dir / "campaign.json").exists():
            if args.validation_session:
                raise ValueError("--validation-session applies only to an individual run.")
            payload = json.loads((run_dir / "campaign.json").read_text(encoding="utf-8"))
            if payload.get("kind") == "validation":
                from psscba.frontend.validation import run_validation_program

                config = load_config(run_dir / "resolved.yaml")
                for output in run_validation_program(config, session_path=run_dir):
                    print(output)
            else:
                for output in resume_campaign(run_dir):
                    print(output)
        else:
            print(resume_case(run_dir, validation_session=args.validation_session))
        return 0

    config = load_config(args.config)
    if args.command in ("run", "sweep", "compare-preparation"):
        from psscba.frontend.dashboard import BANNER, SOFTWARE_NAME

        print(BANNER)
        print(SOFTWARE_NAME)
    if args.command == "validate":
        config.validate()
        print(synthetic_validation())
        if args.full:
            from psscba.frontend.dashboard import BANNER, SOFTWARE_NAME
            from psscba.frontend.validation import run_validation_program

            session = Path(args.session) if args.session else (
                Path(config.output_root)
                / f"validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            )
            print(BANNER, flush=True)
            print(f"{SOFTWARE_NAME} validation", flush=True)
            print(f"validation_directory={session.resolve()}", flush=True)
            for run_dir in run_validation_program(config, session_path=session):
                print(run_dir)
        return 0
    if args.command == "profile":
        synthetic_validation()
        from psscba.frontend.profiling import profile_blas

        print(profile_blas(config))
        return 0
    if args.command == "run":
        print(run_case(config))
        return 0
    if args.command == "compare-preparation":
        print(run_case(config))
        print(run_case(config, preparation_fixed=True))
        return 0
    if args.command == "sweep":
        for output in run_campaign(config, campaign_directory=args.directory):
            print(output)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
