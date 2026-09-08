from __future__ import annotations

from datetime import datetime
import time

from psscba.backend.model.units import gamma_to_current_unit_A, gamma_to_time_unit_s


def format_seconds(seconds: float) -> str:
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:03d}:{minutes:02d}:{secs:02d}"


class Reporter:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.run_name = "NEGF-TDPSCBA"
        self.author = "Louis Rossignol and Hong Guo"

    def banner(self):
        if not self.verbose:
            return

        now = datetime.now().strftime("%d-%b-%Y %H:%M:%S")

        width = 82
        side = "##########"
        border = "#" * width
        inner_width = width - 2 * len(side)

        def line(text: str) -> str:
            return f"{side}{text:^{inner_width}}{side}"

        print()
        print(border)
        print(line(f"Welcome to work with {self.run_name}"))
        print(line(self.author))
        print(line(now))
        print(border)
        print()

    def section(self, title: str) -> None:
        if not self.verbose:
            return
        print()
        print("+" * 82)
        print(title)
        print("+" * 82)

    def info(self, message: str) -> None:
        if not self.verbose:
            return
        print(message, flush=True)

    def timed(self, title: str):
        return TimedBlock(self, title)

    def print_unit_system(self, gamma_eV: float) -> None:
        if not self.verbose:
            return

        t_unit_ps = 1e12 * gamma_to_time_unit_s(gamma_eV)
        I_unit_uA = 1e6 * gamma_to_current_unit_A(gamma_eV)

        self.section("Physical unit system")
        self.info(f"Gamma = {gamma_eV:.6e} eV")
        self.info(f"Time unit   = ħ/Gamma = {t_unit_ps:.6e} ps")
        self.info(f"Current unit = e Gamma / ħ = {I_unit_uA:.6e} µA")


class TimedBlock:
    def __init__(self, reporter: Reporter, title: str):
        self.reporter = reporter
        self.title = title
        self.t0: float | None = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        if self.reporter.verbose:
            print(f"{self.title} ......", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = 0.0 if self.t0 is None else (time.perf_counter() - self.t0)

        if self.reporter.verbose:
            if exc_type is None:
                print(f"------ finished. time used: {format_seconds(dt)}", flush=True)
            else:
                print(f"------ failed. time used: {format_seconds(dt)}", flush=True)

        return False
