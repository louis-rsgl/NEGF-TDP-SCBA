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

## Equations implemented

This section summarizes the scalar Lorentzian-dot equations implemented in the
backend, following the v26 frozen-SCBA derivation. The stationary reference is
biased for a downward step and unbiased for upward and square pulses.

### Frozen stationary SCBA

For stationary reference shifts \(d_0\) on the dot and \(d_\alpha\) in lead
\(\alpha\),

\[
\Sigma_{\alpha,\mathrm{lead}}^R(\omega)
=\frac{\Gamma_\alpha^0W}{2(\omega-d_\alpha+iW)},
\qquad
\Sigma_{\mathrm{lead}}^<(\omega)
=i\sum_\alpha f_\alpha(\omega-d_\alpha)
\Gamma_\alpha^0\frac{W^2}{(\omega-d_\alpha)^2+W^2}.
\]

Here \((d_0,d_\alpha)=(\Delta,\Delta_\alpha)\) for downward and
\((d_0,d_\alpha)=(0,0)\) for upward or square. The stationary equations are

\[
G^R=\left[
\omega-\epsilon_0-d_0-\Sigma_{\mathrm{lead}}^R
-\Sigma_{\mathrm{ep}}^R+i\eta
\right]^{-1},
\qquad
G^<=G^R(\Sigma_{\mathrm{lead}}^<+\Sigma_{\mathrm{ep}}^<)G^A.
\]

For one phonon mode,

\[
G^>=G^<+G^R-G^A,
\]

\[
\Sigma_{\mathrm{ep}}^<(\omega)
=g^2\left[(N_0+1)G^<(\omega+\omega_0)
+N_0G^<(\omega-\omega_0)\right],
\]

\[
\Sigma_{\mathrm{ep}}^>(\omega)
=g^2\left[(N_0+1)G^>(\omega-\omega_0)
+N_0G^>(\omega+\omega_0)\right].
\]

The spectral and retarded constructions are

\[
\Gamma_{\mathrm{ep}}(\omega)
=i[\Sigma_{\mathrm{ep}}^>(\omega)-\Sigma_{\mathrm{ep}}^<(\omega)],
\]

\[
\Sigma_{\mathrm{ep,dyn}}^R(\omega)
=\mathcal P\!\int\frac{d\omega'}{2\pi}
\frac{\Gamma_{\mathrm{ep}}(\omega')}{\omega-\omega'}
-\frac{i}{2}\Gamma_{\mathrm{ep}}(\omega).
\]

The code evaluates the principal value with a zero-padded, non-wrapping
Hilbert transform. The stationary occupation and Hartree contribution are

\[
\bar n=-i\int\frac{d\omega}{2\pi}G^<(\omega),
\qquad
\Sigma_H=-\frac{2g^2\bar n}{\omega_0},
\qquad
\Sigma_{\mathrm{ep}}^R=\Sigma_H+\Sigma_{\mathrm{ep,dyn}}^R.
\]

`N0=None` selects the thermal Bose value; otherwise the supplied nonnegative
`N0` is used. Shifted interpolation never wraps around the frequency grid:
outside it, \(G^R(z)\to1/(z+i\eta)\) and \(G^<(z)\to0\).

### Linear mixing and SCBA convergence

In `scba_mode="self_consistent"`, each trial pair is recomputed from the
current pair:

\[
G_{k,\mathrm{trial}}^R
=\left[
\omega-\epsilon_0-d_0-\Sigma_{\mathrm{lead}}^R
-\Sigma_{\mathrm{ep}}^R[G_k]+i\eta
\right]^{-1},
\]

\[
G_{k,\mathrm{trial}}^<
=G_{k,\mathrm{trial}}^R
[\Sigma_{\mathrm{lead}}^<+\Sigma_{\mathrm{ep}}^<[G_k]]
G_{k,\mathrm{trial}}^A.
\]

Both components are mixed together. With
\(X_k=(G_k^R,G_k^<)^T\), plain linear mixing is

\[
\boxed{X_{k+1}=(1-\alpha)X_k+\alpha X_{k,\mathrm{trial}}},
\qquad \alpha=\texttt{scba\_mixing}.
\]

Small \(\alpha\) is generally more stable but slower. After
`scba_diis_start`, the solver may replace this candidate with a damped
Pulay/DIIS combination of recent trial and residual vectors. If that system is
singular or produces non-finite data, the code falls back to linear mixing.

Using

\[
\|F\|_2=\left(\sum_i|F(\omega_i)|^2\Delta\omega\right)^{1/2},
\]

convergence requires absolute and relative residuals of both \(G^R\) and
\(G^<\) to satisfy `scba_tol_abs` and `scba_tol_rel`. Non-convergence raises
`SCBAConvergenceError` and prevents successful scientific output.

The production runner currently uses `scba_mode="weak_born"`. This is a
strict \(O(g^2)\) frozen-kernel calculation, not an iterative SCBA loop:
\(\Sigma_{\mathrm{ep}}[G_0]\) is evaluated once from the no-phonon stationary
reference, followed by one dressed Dyson/Keldysh update. Therefore
`scba_mixing` and DIIS do not affect a `weak_born` run. Select
`"self_consistent"` explicitly to use iterative mixing.

### MiniPole and physical Green poles

Only the causal dynamic retarded self-energy is fitted. The lesser Green
function and lesser self-energy remain real-grid SCBA data. The auxiliary
samples are

\[
\nu_n=\frac{(2n+1)\pi}{\beta_{\mathrm{fit}}},
\qquad
\Sigma_{\mathrm{ep,dyn}}^R(i\nu_n)
=\int\frac{d\omega}{2\pi}
\frac{\Gamma_{\mathrm{ep}}(\omega)}{i\nu_n-\omega}.
\]

MiniPole constructs

\[
\boxed{
\Sigma_{\mathrm{ep,dyn}}^R(z)
\simeq\sum_{j=1}^{M}\frac{s_j}{z-\zeta_j},
\qquad \operatorname{Im}\zeta_j<0.
}
\]

The same fitted \(\{\zeta_j,s_j\}\) is used in the unbiased and biased
propagators:

\[
G_d^R(z)=\left[
z-\epsilon_0-d_0-\Sigma_H
-\sum_\alpha\frac{\Gamma_\alpha^0W}{2(z-d_\alpha+iW)}
-\sum_j\frac{s_j}{z-\zeta_j}
\right]^{-1}.
\]

The self-energy poles \(\zeta_j\) are not the physical Green poles. The Green
poles \(\xi_r\) are the zeros of this Dyson denominator and are extracted from
an equivalent non-Hermitian arrowhead eigenproblem. If
\(a_m=\sum_{\alpha\in m}\Gamma_\alpha^0W/2\) groups leads with the same shift,
the arrowhead matrix has first row

\[
(\epsilon_0+d_0+\Sigma_H,\ a_1,\ldots,a_L,\ s_1,\ldots,s_M)
\]

and auxiliary diagonal entries

\[
(d_1-iW,\ldots,d_L-iW,\zeta_1,\ldots,\zeta_M),
\]

with ones in the first column below the dot entry. Its eigenvalues are the
candidate \(\xi_r\). Their analytic residues are

\[
\boxed{
R_r=\left[
1+\sum_m\frac{a_m}{(\xi_r-d_m+iW)^2}
+\sum_j\frac{s_j}{(\xi_r-\zeta_j)^2}
\right]^{-1}.
}
\]

Lorentzian embedding poles and MiniPole self-energy poles become zeros of the
Dyson-reconstructed Green function; they are not inserted into Green-pole
sums. Negligible pole--zero pairs are filtered by residue magnitude.
Upper-half-plane or ill-conditioned nearly degenerate Green poles abort the
job. At \(g=0\), the standard two-lead setup has two unbiased and three biased
Maciejko Green poles.

### Pulse transforms and current

The stable history cardinal function is

\[
\operatorname{expc}(x|t)=\frac{e^{ixt}-1}{ix},
\qquad \operatorname{expc}(0|t)=t.
\]

For the upward step, v26 gives the finite-pole structure

\[
A_\alpha^\uparrow(\epsilon,t)
=G_{\mathrm{ss}}^{\uparrow,R}(\epsilon+\Delta_\alpha)
+\sum_\ell
\frac{R_\ell^\uparrow
e^{-i(\xi_\ell^\uparrow-\epsilon-\Delta_\alpha)t}}
{\xi_\ell^\uparrow-\epsilon}
\mathcal M_\alpha^\uparrow(\xi_\ell^\uparrow,\epsilon),
\]

and, with \(E=\omega+\epsilon'\),

\[
C^\uparrow(t,\omega,\epsilon')
=G_{\mathrm{ss}}^{\uparrow,R}(E)
+\sum_\ell
\frac{R_\ell^\uparrow e^{-i(\xi_\ell^\uparrow-E)t}}
{\xi_\ell^\uparrow-E}
\mathcal N^\uparrow(\xi_\ell^\uparrow,E).
\]

The voltage-difference kernels \(\mathcal M^\uparrow\) and
\(\mathcal N^\uparrow\) contain no explicit phonon insertion: all retarded
phonon dressing is already in the frozen propagators. The code evaluates them
in its shifted-lead gauge and validates

\[
A_\alpha^\uparrow(\epsilon,0)=\widetilde G_{\mathrm{fr}}^R(\epsilon),
\qquad
C^\uparrow(0,\omega,\epsilon')
=\widetilde G_{\mathrm{fr}}^R(\omega+\epsilon').
\]

The downward transform begins at the biased stationary boundary and uses the
unbiased pole set for its decay:

\[
A_\alpha^\downarrow(\epsilon,0)
=\overline G_{\mathrm{bias}}^R(\epsilon+\Delta_\alpha),
\qquad
A_\alpha^\downarrow(\epsilon,\infty)
=\widetilde G_{\mathrm{fr}}^R(\epsilon).
\]

For a square pulse, causality first gives

\[
\{A,B,C,D,J\}_\square(t)=\{A,B,C,D,J\}_\uparrow(t),
\qquad 0\le t\le s.
\]

For \(t>s\), the v26 retarded residue formulas are

\[
\boxed{
A_\alpha^\square(\epsilon,t)
=\widetilde G_{\mathrm{fr}}^R(\epsilon)
-e^{i\Delta_\alpha s}
\sum_rR_r^0e^{-i(\xi_r^0-\epsilon)t}
\mathcal S_\alpha^\square(\xi_r^0,\epsilon;s),
}
\]

\[
\boxed{
C^\square(t,\omega,\epsilon')
=\widetilde G_{\mathrm{fr}}^R(E)
-\sum_rR_r^0e^{-i(\xi_r^0-E)t}
\mathcal S_C^\square(\xi_r^0,E;s).
}
\]

The outer poles \(\xi_r^0,R_r^0\) are unbiased. The square amplitudes retain
the biased-pole history through the internal lower-plane operator

\[
\boxed{
\mathfrak R_-[F]
=-\sum_{p\in\mathcal P_-(F)}\operatorname*{Res}_{z=p}F(z).
}
\]

The full assembled integrand is tested before cancellations are accepted; its
candidates include biased Green poles, Lorentzian lead poles, and kinematic
poles. MiniPole self-energy poles are never inserted as independent Green
poles.

The post-turnoff lead history is

\[
\begin{aligned}
B_\beta^>(\epsilon,\epsilon',t;s)
={}&e^{i(\epsilon-\epsilon')s}
\operatorname{expc}(\epsilon-\epsilon'|t-s)
\widetilde G_{\mathrm{fr}}^R(\epsilon')\\
&-e^{i\Delta_\beta s}\sum_rR_r^0
e^{i(\epsilon-\xi_r^0)s}
\operatorname{expc}(\epsilon-\xi_r^0|t-s)
\mathcal S_\beta^\square(\xi_r^0,\epsilon';s),
\end{aligned}
\]

with an analogous \(D_\alpha^>\) for the frozen phonon lesser history. The
complete square lesser sector combines the unbiased advanced boundary, stored
upward \(B^\uparrow(s),D^\uparrow(s)\), and the new post-turnoff histories. No
SCBA or MiniPole update occurs at \(t=s\).

For all protocols the scalar lead current is

\[
\boxed{
J_\alpha(t)=-2e\int\frac{d\epsilon}{2\pi}
\Gamma_\alpha(\epsilon)
\operatorname{Im}[\Psi_\alpha(\epsilon,t)
+f_\alpha(\epsilon)A_\alpha(\epsilon,t)].
}
\]

The code also returns occupation and the continuity diagnostic

\[
\mathcal R_{\mathrm{cont}}(t)
=\sum_\alpha J_\alpha(t)-\frac{dn(t)}{dt}.
\]

Square turnoff validation enforces

\[
A_\alpha^\square(\epsilon,s^+)=A_\alpha^\uparrow(\epsilon,s^-),
\qquad
C^\square(s^+,\omega,\epsilon')=C^\uparrow(s^-,\omega,\epsilon').
\]

At \(s=0\) there is no transient; setting \(s=t\) recovers the upward
solution; at long times the square pulse returns to the unbiased frozen state.

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
exist yet. Such a directory is an incomplete diagnostic run. Start a new
runner invocation after fixing the cause; do not combine files from runs with
different `run_info.json` settings.

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
