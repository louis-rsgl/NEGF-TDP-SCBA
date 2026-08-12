# NEGF-TDPSCBA

Time-dependent nonequilibrium Green-function transport for a single electronic
level coupled to finite-bandwidth Lorentzian leads and one phonon mode. The
stationary electron--phonon problem is solved in the self-consistent Born
approximation (SCBA); that stationary kernel is frozen during an upward step,
downward step, or finite square pulse and the transient retarded problem is
evaluated with a Minimal Pole Method (MPM) representation. The production
runner supports all three protocols through one configuration block.

The pulse construction extends Maciejko, Wang, and Guo, *Phys. Rev. B* **74**,
085324 (2006), DOI [10.1103/PhysRevB.74.085324](https://doi.org/10.1103/PhysRevB.74.085324).

## Model and conventions

The lead linewidth and retarded lead self-energy use the same Lorentzian
normalization:

\[
\Gamma_\alpha(\omega)=\Gamma_\alpha^0\frac{W^2}{\omega^2+W^2},
\qquad
\widetilde\Sigma_{\alpha,\mathrm{lead}}^R(z)
=\frac{\Gamma_\alpha^0W}{2(z+iW)}.
\]

The central region has level energy \(\epsilon_0\), coupling \(g\), and one
phonon mode \(\omega_0\). `System.N0` is the prescribed stationary phonon
occupation. When it is `None`, the code uses the thermal value determined by
`beta_ph` and `mu_ph`; an explicit non-negative value selects a nonthermal
stationary phonon population.

All backend energies stored in `System` are dimensionless ratios to the total
linewidth scale \(\Gamma\). The runner is the physical-unit boundary:
`GAMMA` is specified in eV, `GQ_GRID` is specified in meV, and each requested
coupling is converted once using

\[
g/\Gamma=10^{-3}g_{\rm meV}/\Gamma_{\rm eV}.
\]

The requested meV value is retained in filenames, plots, logs, and metadata;
quality files also store the resolved `g_q_resolved_Gamma`. `W_GRID`, the
voltage shifts, dot level, and phonon frequency remain expressed in units of
\(\Gamma\). This convention keeps the SCBA study in its intended weak-coupling
parameterization and prevents physical meV values from being interpreted as
multiples of \(\Gamma\).

## Numerical pipeline

For every parameter set the program performs these steps exactly once:

1. Build the protocol's stationary kernel: unbiased before an upward step or
   square pulse and biased before a downward step. The runner uses strict weak-coupling
   `weak_born` mode: construct the complete \(O(g^2)\) electron--phonon
   self-energy from the no-phonon stationary reference, then solve the fixed
   Dyson/Keldysh equations once. `self_consistent` mode remains available for
   dressed SCBA studies and uses Pulay/DIIS mixing of both components.
2. Freeze \(\Sigma_{\rm ep}^{R,<}\), separating the static Hartree term from
   the causal dynamic retarded self-energy. The immutable record also retains
   the exact lesser Green function used to construct \(\Sigma_{\rm ep}^<\);
   in strict weak-Born mode this is the no-phonon kernel input, not the
   subsequently dressed stationary lesser function.
3. Evaluate the dynamic self-energy on an auxiliary upper-half-plane frequency
   grid and fit it with MiniPole. If that continuation fails in a narrow band,
   a causal AAA fallback discovers real-axis candidate poles, discards every
   upper-half-plane candidate, refits the lower-plane residues, and must pass
   the same self-energy, Green-function, pole, and causality gates.
4. Reconstruct both the unbiased \(G_{\rm fr}^R\) and biased
   \(G_{\rm ss}^R\) from the same fitted self-energy. A generalized arrowhead
   eigenproblem extracts both Green-pole sets; MiniPole self-energy poles are
   never inserted directly into a transient residue sum.
5. Evaluate the selected protocol's \(A\), \(B\), \(C\), and \(D\) functions
   and the full lead plus frozen-electron--phonon lesser current. Downward uses
   the unbiased pole set and upward uses the biased pole set. Square uses the
   upward solution until \(s\), then combines the stored biased-pole history
   with unbiased-pole post-turnoff propagation.

SCBA, MPM, causality, pole-conditioning, reconstruction, and retarded-boundary
failures abort the affected parameter job. Invalid scientific output
is not saved as successful. The raw upward residue sums must reproduce
\(A_\alpha(0)=C(0)=G_{\rm fr}^R\) before the exact boundary is imposed in the
time loop. Square jobs additionally validate the internal contour residues and
the raw \(A/C\) continuity identities at turnoff. The Lorentzian finite
difference is evaluated with the Dyson-consistent sign fixed by these identities.

## Installation

Python 3.13 is the primary development version. Install the pinned dependencies
with:

```bash
python -m pip install -r requirements.txt
```

The requirements include NumPy, SciPy, Matplotlib, tqdm, MiniPole's `kneed`
runtime dependency, and MiniPole pinned to the reviewed Git commit used by this
implementation. SymPy is no longer needed by the runtime pole path.

## Running calculations

`runner.py` is configured through the constants near the top of the file. It
does not currently use command-line options. Make one intentional configuration
change at a time and keep the generated `run_info.json`; it is the exact record
of the values used by a run.

### Choose the pulse protocol

The three protocols use the same voltage shifts (`DELTA` for the dot and each
lead's `Delta`) but apply them at different times:

| `PULSE_PROTOCOL` | State for \(t<0\) | Voltage history for \(t>0\) | Long-time state |
|---|---|---|---|
| `"downward"` | stationary biased state | shifts are removed at \(t=0\) | unbiased |
| `"upward"` | stationary unbiased state | shifts are applied at \(t=0\) | biased |
| `"square"` | stationary unbiased state | shifts are on for \(0<t<s\), then removed | unbiased |

For a downward step, set:

```python
PULSE_PROTOCOL = "downward"
T_MAX = 3.0
N_T = 201
```

For an upward step, set:

```python
PULSE_PROTOCOL = "upward"
T_MAX = 3.0
N_T = 201
```

For a square pulse, also select its duration in units of
\(\hbar/\Gamma\):

```python
PULSE_PROTOCOL = "square"
SQUARE_DURATION = 3.0
T_MAX = 6.0
N_T = 201
```

Square pulses require

```text
0 <= SQUARE_DURATION <= T_MAX
```

The square time grid is constructed piecewise so that it contains
`SQUARE_DURATION` exactly. `N_T` remains the total number of saved time points.
`SQUARE_DURATION` is ignored by the two step protocols.

Do not change `PULSE_PROTOCOL` on a `System` after its stationary state has
been built. Upward and square runs freeze an unbiased stationary kernel;
downward runs freeze a biased kernel. The backend rejects a mismatched cache.

### Select the parameter sweep

The production grids are ordinary NumPy arrays:

```python
W_GRID = np.array([1.0, 2.5, 5.0, 10.0, 20.0, 100.0])
GQ_GRID = np.array([0.0, 0.01, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0])
```

`W_GRID` is in units of \(\Gamma\). `GQ_GRID` is deliberately in meV and is
converted once when the runner constructs `System`:

\[
g/\Gamma=10^{-3}g_{\rm meV}/\Gamma_{\rm eV}.
\]

For example, with `GAMMA = 0.01` eV, `g_q = 20` meV means
\(g/\Gamma=2\), not 20. The input-meV and resolved dimensionless values are
both stored in the quality JSON.

Other important physical controls are defined in `make_sys(...)`:

- `DELTA`: dot voltage shift in units of \(\Gamma\);
- each `LeadParams.Delta`: lead voltage shift in units of \(\Gamma\);
- `w_q`: phonon energy in units of \(\Gamma\);
- `N0=None`: thermal phonon occupation;
- explicit `N0 >= 0`: prescribed nonthermal phonon occupation;
- `ALPHA_DEFAULT`: lead whose current is saved by the runner.

### Choose numerical grids

The main accuracy/cost controls are:

```python
N_W_SCBA = 2001
OMEGA_INT_N_X = 1001
OMEGA_INT_N_OMEGA = 1001
MPM_N_IW = 512
MPM_BETA_FIT = 80.0
```

`OMEGA_INT_N_X` and `OMEGA_INT_N_OMEGA` must match in the current
implementation. Current evaluation scales approximately as
\(N_tN_\epsilon^2\), so a 1001-point, 201-time calculation is a production
job rather than a quick check. Square pulses additionally construct and cache
the internal turnoff residues once per parameter set.

`make_sys(...)` increases the stationary grid for some narrow-band/high-coupling
points; consequently the effective stationary point count in a worker quality
file can be larger than `N_W_SCBA`.

Before launching all 48 production jobs, use a smoke grid such as:

```python
W_GRID = np.array([1.0])
GQ_GRID = np.array([0.0, 0.01])
N_T = 21
OMEGA_INT_N_X = 101
OMEGA_INT_N_OMEGA = 101
```

Restore the production grids only after this run passes and its current shape
looks sensible. Grid convergence should be checked by increasing the
stationary, current-energy, and time grids separately.

### Serial and parallel execution

The conservative default is serial:

```python
PARALLEL = False
MAX_WORKERS = 1
CONTINUE_ON_FAILURE = True
```

Parallel mode is enabled with:

```python
PARALLEL = True
MAX_WORKERS = 4
```

Each worker holds its own stationary arrays and transient integration buffers.
Choose `MAX_WORKERS` from available memory, not only CPU count. Serial mode is
recommended while diagnosing new parameters. With
`CONTINUE_ON_FAILURE=True`, a failed parameter set writes `.failed.json` and
the sweep continues. Set it to `False` to stop at the first failure.

### Launch and monitor a run

Foreground execution is simplest while testing:

```bash
MPLCONFIGDIR=/tmp/negf-mpl-cache python runner.py
```

For a long run, put the environment assignment after `env`; otherwise `nohup`
tries to execute the assignment as a program:

```bash
nohup env MPLCONFIGDIR=/tmp/negf-mpl-cache \
  python runner.py > nohup_square.out 2>&1 &
runner_pid=$!
```

The output filename can be renamed for upward or downward runs. Monitor the
shell process and newest protocol-specific run with:

```bash
ps -p "$runner_pid" -o pid,etime,cmd
tail -f nohup_square.out
run_dir=$(ls -dt run_square_* | head -n 1)
tail -f "$run_dir/master.log"
```

`$!` is only useful in the same shell immediately after launching the
background process. At a later time, use:

```bash
pgrep -af 'python.*runner.py'
```

Useful progress counts are:

```bash
find "$run_dir/logs" -name '*.quality.json' | wc -l
find "$run_dir/logs" -name '*.failed.json' | wc -l
```

A successful worker ends with `finished successfully` and has a corresponding
`.quality.json`. A complete sweep ends with `Run finished` in `master.log`.

### Output lifecycle

Every invocation immediately creates one of:

```text
run_downward_YYYYMMDD_HHMMSS/
run_upward_YYYYMMDD_HHMMSS/
run_square_YYYYMMDD_HHMMSS/
```

The directory contains:

- `run_info.json`: exact physical and numerical configuration;
- `master.log`: sweep-level progress;
- `logs/worker_W..._gq....log`: detailed worker output;
- `logs/worker_W..._gq....quality.json`: diagnostics for a successful job;
- `logs/worker_W..._gq....failed.json`: traceback for a failed job;
- `data/t_ps.npy`: shared time grid after a completed sweep;
- `data/J_<lead>_W..._gq....npy`: one successful selected-lead current;
- `figures/J_<lead>_W..._gq....svg`: individual current figures.

The runner currently accumulates the current grid in memory and writes NPY/SVG
outputs after the sweep finishes. If a serial run is interrupted, completed
worker logs and quality JSON remain useful, but the `data/` directory may not
exist yet. Such a directory is an incomplete diagnostic run and cannot be used
by the subplot plotter. Start a new runner invocation after fixing the cause;
do not combine files from runs with different `run_info.json` settings.

### Diagnose scientific validation failures

Worker quality files record:

- SCBA convergence and retarded/lesser residuals;
- MiniPole method, self-energy poles, and reconstruction errors;
- unbiased and biased Green-pole counts and causality;
- upward initial-boundary errors;
- square turnoff, internal-residue, and contour-convergence errors;
- energy-grid limits and point counts;
- maximum charge-continuity residual.

The `scaled_error` values are measured against the configured absolute plus
relative acceptance band. Values at or below 1 pass. Do not simply loosen a
gate when it fails: first inspect the absolute error, reconstruction error,
pole causality, energy window, and grid convergence.

The continuity residual is a diagnostic rather than an automatic pass/fail
gate. Its reported maximum can be dominated by a finite-difference derivative
at a switching surface. Inspect its behavior away from \(t=0\) and, for a
square pulse, away from \(t=s\), then repeat with finer time and energy grids.

### Plot a completed run

All plot commands require an explicit completed run directory. To create the
large phonon panels in both unit systems:

```bash
python plot_phonon_subplots.py run_square_YYYYMMDD_HHMMSS
```

The same command works for either step protocol:

```bash
python plot_phonon_subplots.py run_upward_YYYYMMDD_HHMMSS
python plot_phonon_subplots.py run_downward_YYYYMMDD_HHMMSS
```

It writes:

```text
figures/current_subplots_uA_ps.svg
figures/current_subplots_original_units.svg
```

Each panel fixes one phonon coupling and overlays all bandwidths. Square plots
mark \(t=s\), and the shared bandwidth legend is placed to the right of the
panels. The plotter reads units, grids, protocol, and duration from
`run_info.json`.

For the phonon-free Maciejko bandwidth comparison:

```bash
python plot_maciejko.py run_square_YYYYMMDD_HHMMSS
```

For a coupling overlay at one bandwidth:

```bash
python plot_phonons.py run_square_YYYYMMDD_HHMMSS --W 20
```

To plot the newest completed run after verifying it is complete:

```bash
run_dir=$(ls -dt run_square_* | head -n 1)
python plot_phonon_subplots.py "$run_dir"
```

### Library interface

The runner saves only the selected lead, but the backend can return every lead
current, occupation, continuity array, and diagnostics:

```python
from backend.observables import current_all, current_alpha

result = current_all(system, t_max=6.0, n_t=201)
t = result.t
left_current = result.currents["L"]
right_current = result.currents["R"]
occupation = result.occupation
continuity = result.continuity_residual
diagnostics = result.diagnostics

t, left_current = current_alpha(system, "L", t_max=6.0, n_t=201)
```

Dispatch uses `system.pulse_protocol`. Explicit
`current_all_down/up/square` and `current_alpha_down/up/square` functions are
also available. For square calculations, set `system.pulse_duration` before
building the frozen state.

Important backend controls include:

```text
N0                         explicit phonon occupation, or None for thermal
pulse_protocol             downward (library default), upward, or square
pulse_duration             required nonnegative duration for square
scba_*                     stationary convergence and mixing controls
mpm_tol                    requested MiniPole tolerance (default 1e-8)
mpm_n_iw                   auxiliary upper-half-plane samples
mpm_beta_fit               auxiliary sampling scale
mpm_fit_abs_tol            self-energy/Green absolute acceptance threshold
mpm_fit_rel_tol            relative self-energy acceptance threshold
mpm_green_rel_tol          relative Green-function acceptance threshold
current_energy_batch       outer-energy integration batch size
square_residue_n_theta     internal square contour points (default 64)
square_residue_abs_tol     square contour absolute tolerance (default 1e-8)
square_residue_rel_tol     square contour relative tolerance (default 1e-6)
```

## Validation

Run the automated tests with:

```bash
pytest -q
```

The suite covers protocol-specific stationary references, Lorentzian
normalization and gauge shifts, Hilbert-transform sign, SCBA
convergence/failure, controlled interpolation tails, both Green-pole sets,
upward, downward, and square switching identities, direct real-axis checks,
stable `expc`, and small full-current calculations. Production studies should
additionally tighten the SCBA/MPM grids until the current and continuity
residual are converged.

## References

- J. Maciejko, J. Wang, and H. Guo, *Phys. Rev. B* **74**, 085324 (2006).
- L. Zhang, Y. Yu, and E. Gull, *Phys. Rev. B* **110**, 235131 (2024), DOI
  [10.1103/PhysRevB.110.235131](https://doi.org/10.1103/PhysRevB.110.235131).

## Authors and citation

Louis Rossignol and Hong Guo, McGill University.

```bibtex
@software{rossignol_negf_tdpscba,
  author = {Rossignol, Louis and Guo, Hong},
  title = {NEGF-TDPSCBA: Time-Dependent Quantum Transport with Frozen-SCBA Phonons},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/louis-rsgl/NEGF-TDPSCBA}
}
```
