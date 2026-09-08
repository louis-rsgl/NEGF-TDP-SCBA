# NEGF TDP PS-SCBA

Finite-bandwidth transient transport with a protocol-conditioned projected stationary self-consistent Born approximation (PS-SCBA).

## Purpose

The solver treats a spinless resonant level coupled to two Lorentzian leads and one thermal phonon mode. Each case follows

```text
installed stationary kernel -> causal rational poles and fixed-kernel amplitudes
-> A, B, C, D, Psi -> G(t,t') -> uniform projection P G
-> stationary Hartree/Fock candidate -> residual, linear mixing, checkpoint
```

The mathematical specification is [`docs/scientific-method.md`](docs/scientific-method.md). Production cases use streamed two-time reconstruction, reduced projection data, and pole-energy A/C/Psi currents. Dense two-time arrays are enabled only for explicitly selected validation cases. Retarded pole extraction is hybrid: stationary kernels use the upstream Matsubara `MiniPole` class, projected kernels use causal AAA plus residue refitting, and physical Green poles are solved algebraically.

## Quick start

Run from the repository root with Python 3.13 or newer. This is a source checkout, not an installed wheel.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install "numpy>=2.0" "scipy>=1.14" "matplotlib>=3.9" "PyYAML>=6.0" "threadpoolctl>=3.6" kneed "mini_pole @ git+https://github.com/Green-Phys/MiniPole.git@15e4a541d652d2584fd4680413d11b5571f78d9b"
python -m pip install "pytest>=8"
cp examples/example.yaml my_case.yaml
python runner.py check my_case.yaml
python runner.py run my_case.yaml
```

The example is intentionally small and educational. Choose finer grids and longer windows for production studies.

## Commands and monitoring

`runner.py` owns branded logging, process locking, Git provenance, and validation authorization. It is safe with `nohup`:

```bash
nohup python -u runner.py check my_case.yaml >/dev/null 2>&1 &
nohup python -u runner.py validate my_case.yaml >/dev/null 2>&1 &
nohup python -u runner.py sweep my_case.yaml >/dev/null 2>&1 &
nohup python -u runner.py resume runs/<job> >/dev/null 2>&1 &
```

```text
check CONFIG                 schema and synthetic checks
test [PYTEST_ARGS...]        complete unit suite
profile CONFIG               BLAS and memory profile
validate CONFIG              tests, profile, oracle preflight, full validation
run CONFIG                   one PS-SCBA case
sweep CONFIG                 validated parameter sweep
compare-preparation CONFIG   explicit preparation-fixed comparison
resume RUN                   continue a job or campaign
status RUN_OR_CAMPAIGN       refresh campaign dashboard or print job progress
plot RUN                     regenerate deterministic SVG products
aggregate DIRECTORY          aggregate completed jobs
```

Every command writes its own record under `runs/`; no root-level `logs/` directory is required. Long commands create `runs/negf-tdp-ps-scba_<mode>_<UTC>/launcher.log`. `check` creates the same type of small run record. Campaigns additionally contain `main.log`, `dashboard.html`, `campaign.json`, `summary.json`, and `jobs/<case-id>/`. Each job has `solver.log`, `progress.json`, `tracking_history.jsonl`, `convergence.csv`, a latest checkpoint, data, and plots.

Validation is intentionally staged before its first case. The launcher log and
the session `main.log` print each preflight protocol (`upward`, `downward`,
`square`) and its current substage: stationary initialization, fixed-kernel
pulse, fast pole/Filon current, dense oracle, oracle comparison, and
production-grid conservation. Batch counts and elapsed seconds are included,
so a dense oracle calculation does not appear idle. The live HTML dashboard
shows the same preflight stage.

```bash
python runner.py status runs/<campaign-or-job>
tail -f runs/<campaign>/main.log
tail -f runs/<campaign>/jobs/*/solver.log
```

Validation writes `runs/.validation-ready.json` when all production gates pass; it may include a diagnostic-failure count for non-gating stress cases. A sweep refuses a missing marker, dirty tree, different Git commit, or changed configuration hash. Interrupted jobs retain their checkpoints.

The validation ladder is configured by the `validation:` block in
`configs/production.yaml`; no physical frequency is substituted by the
runner. The production baseline is the manuscript value
`phonon_energy: 0.2` with thermal `phonon_occupation: null`. A separate
`hot_N1` family sets `N0=1` and derives `beta_ph = log1p(1/N0)/phonon_energy`
for each declared frequency. The frequency axis is `[0.1, 0.2, 0.5, 1.0, 2.0]`.

Validation has foundation, numerical/collision, approximation, and stress
tiers. Dense two-time storage is restricted to compact references and
collision cases; long-window and parameter-map cases are streamed using
`campaign.max_workers`. The
collision identity error (continuity minus the discarded-sector source) is
reported separately from the magnitude of that source, which measures the
effect of the PS-SCBA projection. Finite-window and semi-infinite prehistory
collision contributions are saved independently. Numerical gates retain the
existing thresholds; strong-coupling and hot-stress failures are accumulated
as diagnostics and do not authorize claims about full two-time SCBA.

A failed gating case or preflight blocks the production marker, but the
validation ladder continues through every planned case. Diagnostic-only
failures produce `complete_with_diagnostic_failures` and are recorded in
`campaign.json` and `.validation-ready.json`.

## Outputs and acceptance

Physical current products are the default (`t` in ps and `J` in µA); natural-unit figures carry the `_natural.svg` suffix. Current figures label both the raw pole/Filon current and, when available, the separate noninteracting conservation-corrected diagnostic view. Dense/streaming, causality, Hermiticity, and collision gates apply to cases with `projection.dense_validation: true`; streamed production cases retain compact residual and current diagnostics. The outer loop uses linear mixing; Pulay/Broyden acceleration is deferred.

Completed production plots also include lead-resolved amplitude images:
`plots/publication/amplitude_A2_L.svg`, `amplitude_A2_R.svg`,
`amplitude_C2_minus_omega.svg`, and `amplitude_C2_plus_omega.svg`. The A
panels show \(|A_\alpha(t,E)|^2\); the C panels show the explicit emission
and absorption branches \(|C(t,-\omega_0,E')|^2\) and
\(|C(t,+\omega_0,E')|^2\), with their SCBA weights recorded in
`diagnostics.json`. Their vertical coordinate is dimensionless, with
natural-unit axes \(t[\hbar/\Gamma]\) and \(E/\Gamma\) for A, or
\(E'/\Gamma\) for C. Publication images use the resonance-focused window
\(-20\le E/\Gamma\le20\) (or \(-20\le E'/\Gamma\le20\)); the full
products remain in `data/amplitude_A2_*.npy` and
`data/amplitude_C2_*_omega.npy`. Titles report both \(W/\Gamma\) and
\(\omega_0/\Gamma\), since the former sets the contact energy scale and
the latter sets the phonon sideband shifts. Rendering is decimated and clipped
only for visualization; it does not alter the calculation or saved arrays.

## Development

Run `python runner.py test`. Protocol amplitudes use blocked pole-energy tiles controlled by `amplitude_time_batch` and `energy_batch`, tested against scalar references for upward, downward, and square protocols. The direct A/Psi implementation is an independent validation oracle. The repository does not currently call upstream `MiniPoleRf` directly because projected self-energies are stored as finite real-axis arrays rather than callable analytic functions; causal AAA is used for that case and validated on the complete grid.

## Citation

Use [`CITATION.cff`](CITATION.cff). The finite-bandwidth pulse model follows Maciejko, Wang, and Guo, *Phys. Rev. B* **74**, 085324 (2006). The cited Minimal Pole Method is used through the upstream Matsubara `MiniPole` implementation for stationary kernels; projected kernels use the separately documented causal AAA rational fit, and physical Green poles are obtained algebraically.

## Configuration

The top level is `config_format: negf-tdp-ps-scba`. Sections are:

| Section | Main fields | Units |
|---|---|---|
| `model` | lead fractions, level/lead shifts, temperatures, `bandwidth`, `phonon_energy`, `coupling_meV` | energies in Gamma unless marked eV/meV |
| `protocol` | `upward`, `downward`, `square`, or `zero`; square `duration` | hbar/Gamma |
| `projection` | finite time window, `time_step`, uniform weight, `energy_batch`, `amplitude_time_batch`, `lag_batch` | natural time |
| `numerics` | reconstruction/current grids, MPM tolerances, stationary/outer limits, `outer_mixing` | natural units |
| `campaign` | bandwidth, coupling, phonon, protocol axes, square duration, worker/BLAS counts | coupling axis is meV |
| `tracking` | refresh/heartbeat intervals, resource sampling, memory warning | seconds |
| `plotting` | SVG, MathText/serif style, physical/natural products | — |

The time grid must contain an integer number of intervals and satisfy

\[
\Delta t < \frac{\pi}{\max(|E_{\min}|,|E_{\max}|)}.
\]

`numerics.eta` is a regulator for stationary-grid quadrature and pole-fit
validation only. The causal pole representation is evaluated on the physical
real-axis boundary (E+i0^+) for transient amplitudes and observables; `eta`
is not added as an untracked lifetime. Production calculations must still
demonstrate convergence as `eta` is reduced.

For production, the physical-current integral uses the same energy grid as the
stationary reconstruction and SCBA (`[-200,200]`, `Delta E=0.02` in the
production configuration). This shared discretization keeps the current,
occupation, projected Green functions, and collision functional on one
well-defined quadrature. Independent current bounds are allowed only in
explicit convergence/reference studies and are never used for production
authorization. The raw pole current is retained, and the exact-`g=0` conserving
view is stored separately and plotted with an explicit diagnostic label.
Run diagnostics record the current-energy range, point count, and step, along
with the observable-time point count and step, for reproducible
quadrature-convergence checks.

The lesser-source factorization follows that same production grid. The code's
pole-centred lesser-grid refinement is disabled by default and is available
only through an explicit isolated diagnostic call; it is not part of a
validation or sweep case.

The production phonon energy is the manuscript value \(\omega_0=0.2\Gamma\),
so \(\beta_{ph}=20\) gives a thermal occupation
\(N_0=[\exp(4)-1]^{-1}\approx1.87\times10^{-2}\). The checked-in
`configs/visual_phonon.yaml` uses the same frequency but explicit `N0=1` and
`coupling_meV: 1.0` (\(g_q/\Gamma=0.1\)); it is a labelled visibility/stress
benchmark, not a replacement for the thermal production state.

`current_representation` is retained only for compatibility with older YAML
files. The production `finite_window` method always uses the pole/Filon
evaluator; `time_domain` is deprecated and does not select a dense time-domain
current integral. Use `current_method: direct_oracle` for the slower literal
reference implementation.

Internally, energy is in \(\Gamma\), time in \(\hbar/\Gamma\), and current in \(e\Gamma/\hbar\). Saved observable arrays use ps and µA and include `units.json` with exact conversion factors.
