from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, TYPE_CHECKING

import numpy as np

Lead = str
PulseProtocol = Literal["downward", "upward", "square"]

if TYPE_CHECKING:
    from backend.SCBA import FrozenSCBA, Solver, SolverResult
    from backend.minimal_poles import PoleCache
    from backend.reporting import Reporter


@dataclass
class LeadParams:
    Gamma0: float
    Delta: float
    beta: float
    mu: float


@dataclass
class System:
    pulse_protocol: PulseProtocol = "downward"
    pulse_duration: float | None = None
    ETA: float = 1e-5
    DELTA: float = 0.0

    leads: Dict[Lead, LeadParams] = field(default_factory=dict)

    W: float = 1.0
    g_q: float = 0.0
    w_q: float = 0.0
    e_0: float = 0.0

    beta_ph: float = 1.0
    mu_ph: float = 0.0
    N0: float | None = None

    beta_fd: float = 1.0
    mu_fd: float = 0.0

    e_min: float = -10.0
    e_max: float = 10.0
    omega_min: float = -10.0
    omega_max: float = 10.0

    scba_max_iter: int = 200
    scba_mode: str = "self_consistent"
    scba_tol_abs: float = 1e-8
    scba_tol_rel: float = 1e-6
    scba_mixing: float = 0.05
    scba_min_iter: int = 5
    n_w_scba: int = 2001
    scba_diis_size: int = 6
    scba_diis_start: int = 10
    scba_diis_damping: float = 0.5
    scba_diis_regularization: float = 1e-10
    scba_spectral_weight_tol: float = 1e-1
    scba_edge_fraction_tol: float = 5e-2

    mpm_tol: float = 1e-8
    mpm_n_iw: int = 256
    mpm_beta_fit: float = 20.0
    mpm_reconstruction_floor: float = 1e-10
    mpm_fit_abs_tol: float = 1e-5
    mpm_fit_rel_tol: float = 1.5e-1
    mpm_green_rel_tol: float = 2e-2
    mpm_aaa_rtol: float = 1e-6
    mpm_aaa_max_terms: int = 120
    pole_residue_tol: float = 1e-10
    pole_causality_tol: float = 1e-10
    pole_merge_tol: float = 1e-7
    current_energy_batch: int = 32
    square_residue_n_theta: int = 64
    square_residue_abs_tol: float = 1e-8
    square_residue_rel_tol: float = 1e-6
    square_residue_cancellation_rel_tol: float = 1e-12

    verbose: bool = True

    _cached_eq_poles: list | None = None
    _noneq_result: Optional["SolverResult"] = None
    _noneq_solver: Optional["Solver"] = None
    _frozen_scba: Optional["FrozenSCBA"] = field(default=None, init=False, repr=False)
    _pole_cache: Optional["PoleCache"] = field(default=None, init=False, repr=False)
    _square_kernel_cache: object | None = field(default=None, init=False, repr=False)
    _reporter: Optional["Reporter"] = field(default=None, init=False, repr=False)

    @property
    def lead_names(self) -> List[Lead]:
        return list(self.leads.keys())

    def Gamma0(self, lead: Lead) -> float:
        return self.leads[lead].Gamma0

    def Delta(self, lead: Lead) -> float:
        return self.leads[lead].Delta

    def beta_fc(self, lead: Lead) -> float:
        return self.leads[lead].beta

    def mu_fc(self, lead: Lead) -> float:
        return self.leads[lead].mu

    @property
    def reference_is_biased(self) -> bool:
        self.validate_pulse_protocol()
        return self.pulse_protocol == "downward"

    @property
    def reference_device_shift(self) -> float:
        return self.DELTA if self.reference_is_biased else 0.0

    def reference_lead_shift(self, lead: Lead) -> float:
        return self.Delta(lead) if self.reference_is_biased else 0.0

    def validate_pulse_protocol(self) -> None:
        if self.pulse_protocol not in ("downward", "upward", "square"):
            raise ValueError(
                "pulse_protocol must be 'downward', 'upward', or 'square'; got "
                f"{self.pulse_protocol!r}."
            )
        if self.pulse_protocol == "square":
            if self.pulse_duration is None:
                raise ValueError("pulse_duration is required for the square protocol.")
            if not np.isfinite(self.pulse_duration) or self.pulse_duration < 0.0:
                raise ValueError("pulse_duration must be finite and nonnegative.")

    def reporter(self) -> "Reporter":
        if self._reporter is None:
            from backend.reporting import Reporter
            self._reporter = Reporter(verbose=self.verbose)
        return self._reporter

    def launch(self) -> None:
        rep = self.reporter()

        rep.banner()
        rep.section("System configuration")

        self.validate_pulse_protocol()
        rep.info(f"pulse_protocol = {self.pulse_protocol}")
        if self.pulse_protocol == "square":
            rep.info(f"pulse_duration = {self.pulse_duration:.6e} hbar/Gamma")
        rep.info(
            "stationary_reference = "
            f"{'biased' if self.reference_is_biased else 'unbiased'}"
        )
        rep.info(f"ETA = {self.ETA:.6e}")
        rep.info(f"DELTA = {self.DELTA:.6e}")
        rep.info(f"W = {self.W:.6e}")
        rep.info(f"g_q = {self.g_q:.6e}")
        rep.info(f"w_q = {self.w_q:.6e}")
        rep.info(f"e_0 = {self.e_0:.6e}")
        rep.info(f"beta_ph = {self.beta_ph:.6e}")
        rep.info(f"mu_ph = {self.mu_ph:.6e}")
        rep.info(f"N0 = {'thermal' if self.N0 is None else f'{self.N0:.6e}'}")
        rep.info(f"beta_fd = {self.beta_fd:.6e}")
        rep.info(f"mu_fd = {self.mu_fd:.6e}")
        rep.info(f"e window = [{self.e_min:.6e}, {self.e_max:.6e}]")
        rep.info(f"omega window = [{self.omega_min:.6e}, {self.omega_max:.6e}]")

        rep.info("SCBA solver parameters:")
        rep.info(f"  mode     = {self.scba_mode}")
        rep.info(f"  max_iter = {self.scba_max_iter}")
        rep.info(f"  tol_abs  = {self.scba_tol_abs:.3e}")
        rep.info(f"  tol_rel  = {self.scba_tol_rel:.3e}")
        rep.info(f"  mixing   = {self.scba_mixing:.3e}")
        rep.info(f"  min_iter = {self.scba_min_iter}")
        rep.info(f"  n_w      = {self.n_w_scba}")
        rep.info(f"  DIIS      = size {self.scba_diis_size}, start {self.scba_diis_start}")
        rep.info("MPM parameters:")
        rep.info(f"  tolerance = {self.mpm_tol:.3e}")
        rep.info(f"  n_iw      = {self.mpm_n_iw}")
        rep.info(f"  beta_fit  = {self.mpm_beta_fit:.6e}")
        rep.info(f"  causal AAA fallback rtol = {self.mpm_aaa_rtol:.3e}")

        rep.info("Leads:")
        for lead in self.lead_names:
            rep.info(
                f"  {lead}: "
                f"Gamma0={self.Gamma0(lead):.6e}, "
                f"Delta={self.Delta(lead):.6e}, "
                f"beta={self.beta_fc(lead):.6e}, "
                f"mu={self.mu_fc(lead):.6e}"
            )

    def invalidate_noneq_cache(self) -> None:
        self._noneq_result = None
        self._noneq_solver = None
        self._frozen_scba = None
        self._pole_cache = None
        self._square_kernel_cache = None

    def solve_noneq(
        self,
        w_min: Optional[float] = None,
        w_max: Optional[float] = None,
        n_w: Optional[int] = None,
        force: bool = False,
    ):
        """Solve the protocol's stationary reference (legacy public name)."""
        self.validate_pulse_protocol()
        if self._noneq_solver is not None and self._noneq_result is not None and not force:
            return self._noneq_result

        from backend.SCBA import Solver

        solver = Solver(
            self,
            w_min=self.omega_min if w_min is None else w_min,
            w_max=self.omega_max if w_max is None else w_max,
            n_w=self.n_w_scba if n_w is None else n_w,
        )

        result = solver.solve()
        self._noneq_solver = solver
        self._noneq_result = result
        self._frozen_scba = solver.frozen
        self._pole_cache = None
        self._square_kernel_cache = None

        return result

    def frozen_scba(self) -> "FrozenSCBA":
        if self._frozen_scba is None:
            self.solve_noneq()
        if self._frozen_scba is None:
            raise RuntimeError("SCBA completed without producing a frozen state.")
        return self._frozen_scba

    def solve_stationary(self, *args, **kwargs):
        """Protocol-neutral alias for :meth:`solve_noneq`."""
        return self.solve_noneq(*args, **kwargs)

    def prepare_poles(self, force: bool = False) -> "PoleCache":
        if self._pole_cache is not None and not force:
            return self._pole_cache
        from backend.minimal_poles import build_pole_cache

        self._pole_cache = build_pole_cache(self, self.frozen_scba())
        self._square_kernel_cache = None
        return self._pole_cache

    def require_noneq(self) -> None:
        if self._noneq_solver is None or self._noneq_result is None:
            raise RuntimeError("Nonequilibrium solution not available. Call solve_noneq() first.")

    def GR_noneq(self, e):
        self.require_noneq()
        return self._noneq_solver.GR(e)

    def Gless_noneq(self, e):
        self.require_noneq()
        return self._noneq_solver.Gless(e)

    def GA_noneq(self, e):
        self.require_noneq()
        return self._noneq_solver.GA(e)

    def G_reference_R(self, e):
        self.require_noneq()
        return self._noneq_solver.GR(e)

    def G_reference_less(self, e):
        self.require_noneq()
        return self._noneq_solver.Gless(e)

    def G_reference_A(self, e):
        self.require_noneq()
        return self._noneq_solver.GA(e)
