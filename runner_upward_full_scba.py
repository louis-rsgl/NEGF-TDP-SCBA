"""Production upward-step sweep with fully self-consistent SCBA."""

from __future__ import annotations

import runner


def configure() -> None:
    runner.PULSE_PROTOCOL = "upward"
    runner.STATIONARY_MODE = "self_consistent"
    runner.T_MAX = 3.0
    runner.N_T = 201

    # Run this protocol as one serial process.  The three protocol entry
    # points may be launched concurrently without nested worker pools.
    runner.PARALLEL = False
    runner.MAX_WORKERS = 1
    runner.CONTINUE_ON_FAILURE = True


def main() -> None:
    configure()
    runner.main()


if __name__ == "__main__":
    main()
