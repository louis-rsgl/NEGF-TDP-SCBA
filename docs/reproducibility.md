# Reproducibility and coding-agent guide

## Installation and execution model

Python 3.13 is the repository's stated development version. From the repository
root:

```bash
python -m pip install -r requirements.txt
pytest -q
python runner.py
```

There is no `pyproject.toml` or installable package definition. Imports assume
the repository root is on `sys.path`, so run scripts from that directory. The
calculation uses NumPy/SciPy on CPU. It has no GPU or MPI implementation;
`runner.py` can distribute independent parameter points with Python process
workers.

## Minimal library example

This small phonon-free example exercises the public result interface without a
MiniPole fit. Increase grids and enable nonzero `g_q` only after it succeeds.

```python
from backend.observables import current_all
from backend.system_classes import LeadParams, System

system = System(
    pulse_protocol="upward",
    ETA=1e-4,
    DELTA=0.4,
    leads={
        "L": LeadParams(Gamma0=0.4, Delta=0.7, beta=2.0, mu=0.0),
        "R": LeadParams(Gamma0=0.6, Delta=0.0, beta=2.0, mu=0.0),
    },
    W=3.0,
    g_q=0.0,
    w_q=0.3,
    e_0=0.1,
    omega_min=-12.0,
    omega_max=12.0,
    e_min=-12.0,
    e_max=12.0,
    n_w_scba=241,
    scba_mode="weak_born",
    verbose=False,
)

result = current_all(
    system,
    t_max=0.1,
    n_t=11,
    omega_int_n_omega=121,
)

print(result.t.shape)                 # (11,)
print(result.currents["L"].shape)    # (11,)
print(result.occupation.shape)        # (11,)
print(result.diagnostics)
print(system.provenance())
print(system.citations())
```

For a square pulse, set `pulse_protocol="square"` and a nonnegative
`pulse_duration <= t_max`. The generated time grid contains the turnoff time.

## Inputs, outputs, shapes, and conventions

- All `System` energies, inverse temperatures, and times are dimensionless in
  powers of the total linewidth scale \(\Gamma\). Time is in \(\hbar/\Gamma\)
  and current in \(e\Gamma/\hbar\).
- The production runner alone accepts `GAMMA` in eV and `GQ_GRID` in meV and
  converts once at the backend boundary.
- Energy and time inputs may be scalars or one-dimensional NumPy-compatible
  arrays for the `A` and `C` helpers. Public current routines return a
  `TransientResult` with one-dimensional time, occupation, continuity, and
  per-lead current arrays of length `n_t`.
- `omega_int_n_x` and `omega_int_n_omega`, if both supplied, must be equal.
- `current_alpha` returns `(time, current)`; `current_all` retains every lead,
  occupation, diagnostics, and continuity residual.

## Solver selection

- `scba_mode="weak_born"`: noniterative strict \(O(g^2)\) kernel from the
  no-phonon reference. This is the default production-runner mode.
- `scba_mode="self_consistent"`: dressed fixed-point iteration using joint
  retarded/lesser mixing and optional Pulay/DIIS. The three `*_full_scba.py`
  scripts select this mode.
- `pulse_protocol`: `downward`, `upward`, or `square`. Changing it after a
  stationary solution invalidates the physical meaning of cached objects;
  construct a new `System` or call `invalidate_noneq_cache()`.

## Failure conditions

Successful output is withheld for explicit SCBA nonconvergence, spectral
sum-rule or edge failure, invalid MPM/AAA reconstruction, upper-half-plane or
nearly degenerate Green poles, upward boundary failure, square internal-residue
failure, or turnoff discontinuity. The continuity residual is reported but is
not a pass/fail gate because numerical derivatives at switching surfaces can
dominate it.

## Performance and convergence

The stationary grid, two-dimensional lesser current integrations, and square
internal residues dominate memory and runtime. `current_energy_batch` limits
outer-energy working memory. Process parallelism duplicates solver state, pole
data, and integration buffers. Converge the stationary window and spacing, MPM
sampling/tolerance, current energy grid, time grid, and square contour controls
for the observable of interest; a unit test is not a production convergence
study.

## Output provenance

New `run_info.json` files include software/repository state, Python and direct
dependency versions, and structured citation records. `System.provenance()`
adds the selected protocol, stationary approximation, units, and all public
numerical settings. `System.citations()` returns the method/software citation
records. Null release version and DOI fields are intentional: those records do
not yet exist in the repository.
