from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
import yaml


CONFIG_FORMAT = "negf-tdp-ps-scba"
READ_ONLY_LEGACY_FORMATS = frozenset({"2.0", "3.0"})
ProtocolName = Literal["upward", "downward", "square", "zero"]


@dataclass(frozen=True)
class ModelConfig:
    gamma_eV: float = 0.01
    gamma_left: float = 0.5
    gamma_right: float = 0.5
    epsilon_0: float = 0.0
    device_shift: float = 5.0
    left_shift: float = 10.0
    right_shift: float = 0.0
    beta: float = 10.0
    beta_ph: float = 20.0
    phonon_energy: float = 0.2
    phonon_occupation: float | None = None
    bandwidth: float = 20.0
    coupling_meV: float = 1.0

    @property
    def coupling(self) -> float:
        return 1e-3 * self.coupling_meV / self.gamma_eV

    def validate(self) -> None:
        positive = {
            "gamma_eV": self.gamma_eV,
            "bandwidth": self.bandwidth,
            "beta": self.beta,
            "beta_ph": self.beta_ph,
            "phonon_energy": self.phonon_energy,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive; got {value!r}.")
        if self.gamma_left < 0.0 or self.gamma_right < 0.0:
            raise ValueError("Lead linewidths must be non-negative.")
        if not np.isclose(self.gamma_left + self.gamma_right, 1.0):
            raise ValueError("Dimensionless lead linewidths must sum to one Gamma.")
        if self.coupling_meV < 0.0:
            raise ValueError("coupling_meV must be non-negative.")
        if self.phonon_occupation is not None and self.phonon_occupation < 0.0:
            raise ValueError("phonon_occupation must be non-negative or null.")


@dataclass(frozen=True)
class ProtocolConfig:
    name: ProtocolName = "upward"
    duration: float | None = None

    def validate(self) -> None:
        if self.name not in ("upward", "downward", "square", "zero"):
            raise ValueError(f"Unknown protocol {self.name!r}.")
        if self.name == "square":
            if self.duration is None or not np.isfinite(self.duration) or self.duration < 0:
                raise ValueError("A square protocol requires a finite non-negative duration.")


@dataclass(frozen=True)
class ProjectionConfig:
    time_min: float = -32.0
    time_max: float = 96.0
    time_step: float = 0.025
    weight: Literal["uniform"] = "uniform"
    dense_validation: bool = False
    reconstruction_lead: str = "L"
    lag_batch: int = 64
    energy_batch: int = 64
    # Number of dynamic observation times evaluated in one BLAS pole-sum tile.
    amplitude_time_batch: int = 16

    def time_grid(self) -> np.ndarray:
        span = self.time_max - self.time_min
        intervals = int(round(span / self.time_step))
        if intervals < 1 or not np.isclose(intervals * self.time_step, span):
            raise ValueError("Projection interval must contain an integer number of time steps.")
        return np.linspace(self.time_min, self.time_max, intervals + 1)

    def validate(self, protocol: ProtocolConfig) -> None:
        if self.weight != "uniform":
            raise ValueError("PS-SCBA production currently requires the uniform projector.")
        if not np.isfinite(self.time_step) or self.time_step <= 0.0:
            raise ValueError("time_step must be finite and positive.")
        if self.time_min >= 0.0 or self.time_max <= 0.0:
            raise ValueError("Projection window must straddle the first switching surface t=0.")
        if protocol.name == "square" and float(protocol.duration) > self.time_max:
            raise ValueError("Square turnoff must lie inside the projection window.")
        if self.reconstruction_lead not in ("L", "R"):
            raise ValueError("reconstruction_lead must be L or R.")
        if self.lag_batch < 1 or self.energy_batch < 1 or self.amplitude_time_batch < 1:
            raise ValueError("Projection batch sizes must be positive integers.")
        self.time_grid()


@dataclass(frozen=True)
class NumericsConfig:
    energy_min: float = -100.0
    energy_max: float = 100.0
    energy_step: float = 0.05
    # Stationary-grid and pole-fit validation regulator.  Causal transient
    # pole functions are evaluated at E+i0+, never at E+i*eta.
    eta: float = 1e-5
    stationary_max_iter: int = 200_000
    stationary_tolerance_abs: float = 1e-7
    stationary_tolerance_rel: float = 1e-6
    stationary_mixing: float = 0.05
    outer_max_iter: int = 2_000
    outer_tolerance: float = 1e-5
    outer_mixing: float = 0.05
    sigma_floor: float = 1e-12
    green_floor: float = 1e-12
    mpm_search_tolerance: float = 1e-6
    mpm_final_tolerance: float = 1e-8
    mpm_n_iw: int = 512
    mpm_beta_fit: float = 80.0
    mpm_aaa_initial_terms: int = 120
    mpm_aaa_max_terms: int = 480
    mpm_aaa_growth_factor: float = 1.5
    # Production uses the reconstruction grid for current quadrature as well.
    # Explicit bounds remain available only for isolated convergence or
    # archive-reference studies; all production configurations leave these
    # fields null (or set them equal to the reconstruction grid).
    current_energy_min: float | None = None
    current_energy_max: float | None = None
    current_energy_points: int | None = None
    current_time_points: int | None = None
    current_method: Literal["finite_window", "direct_oracle"] = "finite_window"
    # Compatibility field for old configurations.  Production currents are
    # always evaluated from the pole-energy A/C/Psi representation; the
    # ``time_domain`` value is deprecated and is mapped to that same evaluator.
    current_representation: Literal["auto", "time_domain", "psi"] = "auto"
    # Semi-infinite pre-pulse collision-history quadrature controls.  These
    # are numerical parameters and are varied explicitly by validation.
    collision_prehistory_quadrature_order: int = 64
    collision_prehistory_scale: float = 1.0
    square_residue_method: Literal["batched_contour", "analytic"] = "batched_contour"
    checkpoint_every: int = 1
    enforce_hard_gates: bool = True
    verbose: bool = False

    def energy_grid(self) -> np.ndarray:
        span = self.energy_max - self.energy_min
        intervals = int(round(span / self.energy_step))
        if intervals < 2 or not np.isclose(intervals * self.energy_step, span):
            raise ValueError("Energy interval must contain an integer number of energy steps.")
        return np.linspace(self.energy_min, self.energy_max, intervals + 1)

    def current_grid_matches_reconstruction(self) -> bool:
        """Whether an explicit current grid is identical to the energy grid.

        A null current-grid specification means "use the reconstruction grid"
        and is therefore considered shared.  Explicitly different grids remain
        available to isolated convergence/reference studies, but production
        validation and sweeps reject them.
        """
        if (
            self.current_energy_min is None
            and self.current_energy_max is None
            and self.current_energy_points is None
        ):
            return True
        if (
            self.current_energy_min is None
            or self.current_energy_max is None
            or self.current_energy_points is None
        ):
            return False
        reconstruction = self.energy_grid()
        current = np.linspace(
            self.current_energy_min,
            self.current_energy_max,
            int(self.current_energy_points),
        )
        return len(current) == len(reconstruction) and np.allclose(
            current, reconstruction, rtol=0.0, atol=1e-12
        )

    def validate(self, projection: ProjectionConfig) -> None:
        if self.energy_min >= self.energy_max or self.energy_step <= 0.0:
            raise ValueError("Invalid reconstruction-energy grid.")
        self.energy_grid()
        if (self.current_energy_min is None) != (self.current_energy_max is None):
            raise ValueError(
                "current_energy_min and current_energy_max must be both set or both null."
            )
        if self.current_energy_min is not None:
            if not np.isfinite(self.current_energy_min) or not np.isfinite(self.current_energy_max):
                raise ValueError("Current-energy bounds must be finite.")
            if self.current_energy_min >= self.current_energy_max:
                raise ValueError("Current-energy bounds must be strictly increasing.")
            if self.current_energy_points is None:
                raise ValueError(
                    "current_energy_points is required with explicit current-energy bounds."
                )
        if self.current_energy_points is not None and self.current_energy_points < 2:
            raise ValueError("current_energy_points must be at least two when set.")
        if projection.time_step >= np.pi / max(abs(self.energy_min), abs(self.energy_max)):
            raise ValueError("time_step violates the reconstruction-grid Nyquist bound.")
        if not (0.0 < self.outer_mixing <= 1.0):
            raise ValueError("outer_mixing must lie in (0, 1].")
        if self.outer_tolerance <= 0.0 or self.outer_max_iter < 1:
            raise ValueError("Outer tolerance and iteration limit must be positive.")
        if self.current_method not in ("finite_window", "direct_oracle"):
            raise ValueError("current_method must be finite_window or direct_oracle.")
        if self.current_representation not in ("auto", "time_domain", "psi"):
            raise ValueError(
                "current_representation must be auto, time_domain, or psi."
            )
        if self.collision_prehistory_quadrature_order < 2:
            raise ValueError("collision_prehistory_quadrature_order must be at least two.")
        if (
            not np.isfinite(self.collision_prehistory_scale)
            or self.collision_prehistory_scale <= 0.0
        ):
            raise ValueError("collision_prehistory_scale must be finite and positive.")
        if self.square_residue_method not in ("batched_contour", "analytic"):
            raise ValueError("square_residue_method must be batched_contour or analytic.")


@dataclass(frozen=True)
class SweepConfig:
    bandwidths: tuple[float, ...] = (1.0, 2.5, 5.0, 10.0, 20.0, 100.0)
    couplings_meV: tuple[float, ...] = (0.0, 0.01, 0.5, 1.0, 2.5, 5.0)
    phonon_energies: tuple[float, ...] = (0.2,)
    protocols: tuple[str, ...] = ("upward", "downward", "square")
    # Keep the default protocol window independent of any physical frequency.
    # Production values are declared in YAML; this is only a generic fallback.
    square_duration: float = 3.0
    time_max_by_protocol: dict[str, float] = field(default_factory=dict)
    compare_preparation_fixed: bool = True
    max_workers: int = 1
    blas_threads: int = 1

    def validate(self) -> None:
        if self.max_workers < 1 or self.blas_threads < 1:
            raise ValueError("Campaign worker and BLAS thread counts must be positive.")
        if not self.protocols or not self.bandwidths or not self.couplings_meV:
            raise ValueError("Campaign axes must not be empty.")
        if not self.phonon_energies or any(
            not np.isfinite(value) or value <= 0.0 for value in self.phonon_energies
        ):
            raise ValueError("Campaign phonon energies must be finite and positive.")
        if any(name not in ("upward", "downward", "square") for name in self.protocols):
            raise ValueError("Campaign protocols must be upward, downward, or square.")
        unknown = set(self.time_max_by_protocol) - {"upward", "downward", "square"}
        if unknown:
            raise ValueError(f"Unknown protocol time maxima: {', '.join(sorted(unknown))}")
        if any(not np.isfinite(value) or value <= 0.0 for value in self.time_max_by_protocol.values()):
            raise ValueError("Protocol time maxima must be finite and positive.")
        square_max = self.time_max_by_protocol.get("square")
        if square_max is not None and square_max < self.square_duration:
            raise ValueError("The square campaign time maximum must include its turnoff.")


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = True
    heartbeat_seconds: float = 60.0
    live_refresh_seconds: float = 10.0
    convergence_refresh_steps: int = 1
    resource_sampling: bool = True
    memory_policy: Literal["warn"] = "warn"
    memory_warning_factor: float = 1.25

    def validate(self) -> None:
        if self.heartbeat_seconds <= 0.0 or not np.isfinite(self.heartbeat_seconds):
            raise ValueError("tracking.heartbeat_seconds must be finite and positive.")
        if self.live_refresh_seconds <= 0.0 or not np.isfinite(self.live_refresh_seconds):
            raise ValueError("tracking.live_refresh_seconds must be finite and positive.")
        if self.convergence_refresh_steps < 1:
            raise ValueError("tracking.convergence_refresh_steps must be positive.")
        if self.memory_policy != "warn":
            raise ValueError("Only tracking.memory_policy: warn is supported.")
        if self.memory_warning_factor < 1.0:
            raise ValueError("tracking.memory_warning_factor must be at least one.")


@dataclass(frozen=True)
class PlottingConfig:
    enabled: bool = True
    formats: tuple[Literal["svg"], ...] = ("svg",)
    style: Literal["serif_mathtext"] = "serif_mathtext"
    # Optional derived figures; physical µA/ps figures are the default.
    natural_units: bool = True
    physical_qa_units: bool = True
    physical_publication_figures: bool = True
    separate_lead_currents: bool = True
    automatic_qa_refresh: bool = True
    use_external_latex: bool = False

    def validate(self) -> None:
        if self.formats != ("svg",):
            raise ValueError("Plotting output is SVG-only.")
        if self.style != "serif_mathtext":
            raise ValueError("The serif_mathtext plotting style is required.")
        if not self.separate_lead_currents:
            raise ValueError("Separate lead-current plots are required.")
        if self.use_external_latex:
            raise ValueError("External LaTeX is intentionally unsupported; use MathText.")


CampaignConfig = SweepConfig


@dataclass(frozen=True)
class ValidationConfig:
    """Declarative axes used by the validation ladder.

    The production model remains the source of the baseline physics.  This
    block describes the additional cases and numerical resolutions so that
    validation choices live in YAML rather than hidden Python constants.
    """

    frequencies: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0)
    couplings_meV: tuple[float, ...] = (0.0, 0.01, 0.1, 0.5, 1.0, 2.5, 5.0)
    trusted_couplings_meV: tuple[float, ...] = (0.0, 0.01, 0.1, 0.5, 1.0)
    diagnostic_couplings_meV: tuple[float, ...] = (2.5, 5.0)
    bandwidths: tuple[float, ...] = (1.0, 2.5, 5.0, 10.0, 20.0, 100.0)
    protocols: tuple[str, ...] = ("upward", "downward", "square")
    hot_occupation: float = 1.0
    square_durations: tuple[float, ...] = (0.0, 0.5, 2.0, 3.0, 5.5)
    dense_time_max: float = 3.0
    compact_energy_bound: float = 20.0
    time_steps: tuple[float, ...] = (0.04, 0.02, 0.01)
    reconstruction_energy_bounds: tuple[float, ...] = (40.0, 80.0, 160.0, 200.0)
    reconstruction_energy_steps: tuple[float, ...] = (0.1, 0.05, 0.025, 0.02)
    current_energy_bounds: tuple[float, ...] = (20.0, 40.0, 80.0, 160.0, 200.0)
    current_energy_steps: tuple[float, ...] = (0.1, 0.05, 0.025, 0.02)
    prehistory_time_mins: tuple[float, ...] = (-2.0, -4.0, -8.0, -16.0)
    prehistory_quadrature_orders: tuple[int, ...] = (32, 64, 128)
    prehistory_scales: tuple[float, ...] = (0.5, 1.0, 2.0)
    stationary_etas: tuple[float, ...] = (1.0e-3, 1.0e-4, 1.0e-5)
    mpm_tolerances: tuple[float, ...] = (1.0e-6, 1.0e-8, 1.0e-10)
    streamed_window_mins: tuple[float, ...] = (-2.0, -4.0, -8.0, -16.0)
    streamed_window_maxs: tuple[float, ...] = (3.0, 6.0, 12.0, 24.0)
    approximation_pairwise: bool = True
    hot_stress_pairwise: bool = True
    compare_preparation_fixed: bool = True

    def validate(self) -> None:
        positive_axes = {
            "frequencies": self.frequencies,
            "bandwidths": self.bandwidths,
            "time_steps": self.time_steps,
            "reconstruction_energy_bounds": self.reconstruction_energy_bounds,
            "reconstruction_energy_steps": self.reconstruction_energy_steps,
            "current_energy_bounds": self.current_energy_bounds,
            "current_energy_steps": self.current_energy_steps,
            "prehistory_quadrature_orders": self.prehistory_quadrature_orders,
            "prehistory_scales": self.prehistory_scales,
            "stationary_etas": self.stationary_etas,
            "mpm_tolerances": self.mpm_tolerances,
        }
        for name, values in positive_axes.items():
            if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
                raise ValueError(f"validation.{name} must contain finite positive values.")
        if not self.couplings_meV or any(value < 0.0 for value in self.couplings_meV):
            raise ValueError("validation.couplings_meV must be non-empty and non-negative.")
        if not self.trusted_couplings_meV or any(value < 0.0 for value in self.trusted_couplings_meV):
            raise ValueError("validation.trusted_couplings_meV must be non-empty and non-negative.")
        if not self.diagnostic_couplings_meV or any(value < 0.0 for value in self.diagnostic_couplings_meV):
            raise ValueError("validation.diagnostic_couplings_meV must be non-empty and non-negative.")
        configured_couplings = set(float(value) for value in self.couplings_meV)
        if not set(float(value) for value in self.trusted_couplings_meV).issubset(configured_couplings):
            raise ValueError("validation.trusted_couplings_meV must be drawn from couplings_meV.")
        if not set(float(value) for value in self.diagnostic_couplings_meV).issubset(configured_couplings):
            raise ValueError("validation.diagnostic_couplings_meV must be drawn from couplings_meV.")
        if any(value < 0.0 for value in self.square_durations):
            raise ValueError("validation.square_durations must be non-negative.")
        if self.hot_occupation <= 0.0 or not np.isfinite(self.hot_occupation):
            raise ValueError("validation.hot_occupation must be finite and positive.")
        if self.dense_time_max <= 0.0 or not np.isfinite(self.dense_time_max):
            raise ValueError("validation.dense_time_max must be finite and positive.")
        if self.compact_energy_bound <= 0.0 or not np.isfinite(self.compact_energy_bound):
            raise ValueError("validation.compact_energy_bound must be finite and positive.")
        if len(self.streamed_window_mins) != len(self.streamed_window_maxs):
            raise ValueError("validation streamed window minima and maxima must have equal length.")
        if any(left >= right for left, right in zip(self.streamed_window_mins, self.streamed_window_maxs)):
            raise ValueError("validation streamed windows must have increasing bounds.")
        if not self.protocols or any(name not in ("upward", "downward", "square") for name in self.protocols):
            raise ValueError("validation.protocols must be upward, downward, or square.")


@dataclass(frozen=True)
class CaseConfig:
    """Fully resolved configuration consumed by one backend solver."""

    config_format: str = CONFIG_FORMAT
    model: ModelConfig = field(default_factory=ModelConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)

    def validate(self) -> None:
        if self.config_format != CONFIG_FORMAT:
            raise ValueError(
                f"Unsupported config_format {self.config_format!r}; expected {CONFIG_FORMAT!r}."
            )
        self.model.validate()
        self.protocol.validate()
        self.projection.validate(self.protocol)
        self.numerics.validate(self.projection)
        self.tracking.validate()
        self.plotting.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunConfig:
    config_format: str = CONFIG_FORMAT
    model: ModelConfig = field(default_factory=ModelConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    campaign: SweepConfig = field(default_factory=SweepConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    output_root: str = "runs"

    def validate(self) -> None:
        if self.config_format != CONFIG_FORMAT:
            raise ValueError(
                f"Unsupported config_format {self.config_format!r}; expected {CONFIG_FORMAT!r}."
            )
        self.model.validate()
        self.protocol.validate()
        self.projection.validate(self.protocol)
        self.numerics.validate(self.projection)
        self.campaign.validate()
        self.validation.validate()
        self.tracking.validate()
        self.plotting.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def case(self) -> CaseConfig:
        return CaseConfig(
            config_format=self.config_format,
            model=self.model,
            protocol=self.protocol,
            projection=self.projection,
            numerics=self.numerics,
            tracking=self.tracking,
            plotting=self.plotting,
        )


def require_shared_production_grid(config: RunConfig | CaseConfig) -> None:
    """Reject an explicitly split current grid for production workflows."""
    if not config.numerics.current_grid_matches_reconstruction():
        raise ValueError(
            "Production validation and sweeps require the current quadrature "
            "to use the reconstruction energy grid; use an isolated case for "
            "independent current-grid studies."
        )


T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any] | None) -> T:
    values = {} if values is None else dict(values)
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {', '.join(unknown)}")
    tuple_fields = {
        "bandwidths", "couplings_meV", "phonon_energies", "protocols", "formats",
        "frequencies", "trusted_couplings_meV", "diagnostic_couplings_meV",
        "square_durations", "time_steps", "reconstruction_energy_bounds",
        "reconstruction_energy_steps", "current_energy_bounds", "current_energy_steps",
        "prehistory_time_mins", "prehistory_quadrature_orders", "prehistory_scales",
        "stationary_etas", "mpm_tolerances", "streamed_window_mins", "streamed_window_maxs",
    }
    for name in tuple_fields:
        if name in values:
            values[name] = tuple(values[name])
    return cls(**values)


def load_config(path: str | Path, *, allow_read_only_legacy: bool = False) -> RunConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML configuration must be a mapping.")
    legacy_version = raw.pop("schema_version", None)
    if legacy_version is not None:
        legacy_version = str(legacy_version)
        if not (
            allow_read_only_legacy
            and legacy_version in READ_ONLY_LEGACY_FORMATS
        ):
            raise ValueError(
                "This run uses a retired numbered configuration format. "
                "Historical runs are read/plot only and cannot be resumed."
            )
    allowed = {item.name for item in fields(RunConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown RunConfig keys: {', '.join(unknown)}")
    config_format = str(raw.get("config_format", CONFIG_FORMAT))
    if config_format != CONFIG_FORMAT:
        raise ValueError(
            f"Unsupported config_format {config_format!r}; expected {CONFIG_FORMAT!r}."
        )
    config = RunConfig(
        config_format=CONFIG_FORMAT,
        model=_construct(ModelConfig, raw.get("model")),
        protocol=_construct(ProtocolConfig, raw.get("protocol")),
        projection=_construct(ProjectionConfig, raw.get("projection")),
        numerics=_construct(NumericsConfig, raw.get("numerics")),
        campaign=_construct(SweepConfig, raw.get("campaign")),
        validation=_construct(ValidationConfig, raw.get("validation")),
        tracking=_construct(TrackingConfig, raw.get("tracking")),
        plotting=_construct(PlottingConfig, raw.get("plotting")),
        output_root=raw.get("output_root", "runs"),
    )
    config.validate()
    return config
