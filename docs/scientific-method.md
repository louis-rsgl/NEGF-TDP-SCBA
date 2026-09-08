# Scientific method

This document specifies the equations implemented by NEGF TDP PS-SCBA. All backend variables are dimensionless: \(E\) is measured in \(\Gamma\), time in \(\hbar/\Gamma\), and current in \(e\Gamma/\hbar\). The storage layer converts observable traces to ps and \(\mu\mathrm{A}\).

## Model and parameters

The device is a spinless level with energy \(\epsilon_0\), coupled to leads \(\alpha\in\{L,R\}\) and a thermal phonon of frequency \(\omega_0\):

\[
H=\epsilon_0 d^\dagger d+\sum_{\alpha k}\epsilon_{\alpha k}c^\dagger_{\alpha k}c_{\alpha k}
 +\sum_{\alpha k}(V_{\alpha k}c^\dagger_{\alpha k}d+\mathrm{h.c.})
 +g_q d^\dagger d(b+b^\dagger)+\omega_0 b^\dagger b .
\]

Each contact has Lorentzian linewidth

\[
\Gamma_\alpha(E)=\Gamma_\alpha\frac{W^2}{E^2+W^2},
\qquad \Gamma_L+\Gamma_R=\Gamma .
\]

The lead distributions are \(f_\alpha(E)=[1+\exp(\beta_\alpha(E-\mu_\alpha))]^{-1}\). The phonon occupation is \(N_0=[\exp(\beta_{ph}\omega_0)-1]^{-1}\) unless explicitly supplied. The input coupling is in meV and is converted to \(g_q=g_{\mathrm{meV}}10^{-3}/\Gamma_{\mathrm{eV}}\). The manuscript production baseline is \(\omega_0=0.2\) and \(\beta_{ph}=20\), giving \(N_0=[\exp(4)-1]^{-1}\approx1.87\times10^{-2}\). The validation frequency axis is declared in `configs/production.yaml`; the runner does not substitute a frequency in Python. The separate visibility family supplies \(N_0=1\) explicitly and derives \(\beta_{ph}=\log(1+1/N_0)/\omega_0\) per case. It uses `coupling_meV=1` (\(g_q/\Gamma=0.1\)) and is labelled as a hot stress benchmark.

Choose a finite reconstruction window \([t_{\min},t_{\max}]\), uniform step \(\Delta t\), and energy grid \([E_{\min},E_{\max}]\). The grid contains an integer number of intervals and obeys

\[
\Delta t<\pi/\max(|E_{\min}|,|E_{\max}|).
\]

For production, the current quadrature uses the same energy grid as the
stationary reconstruction and SCBA. Independent current bounds are permitted
only for explicitly labelled convergence/reference studies. The projection
weight is uniform.

The lesser-source factorization also uses this uniform grid in production.
Adaptive pole-centred refinement is retained only as an explicit diagnostic
option; it is never silently mixed into a validation or production SCBA
functional.

The finite parameter `numerics.eta` regularizes the stationary real-frequency
quadrature and the comparison grid used to validate a pole fit. It is not a
physical electronic reservoir. After a causal rational representation is
constructed, every transient amplitude and observable uses its retarded
boundary value

\[
G^R(E)=\lim_{\eta\to0^+}G^R(E+i\eta).
\]

Using a finite regulator in the transient denominator without a matching
lesser self-energy would introduce an artificial absorbing bath and violate
both the spectral identity and charge continuity. Production results are
also checked under reduction of the stationary-grid regulator.

## Stationary initialization

For zero pulse, solve the stationary Hartree--Fock SCBA equations on the reconstruction grid. With

\[
G^R(E)=\left[E-\epsilon_0-\Sigma^R_L(E)-\Sigma^R_R(E)-\Sigma_H-\Sigma^R_{ep}(E)\right]^{-1},
\]

the Keldysh equation is

\[
G^<(E)=G^R(E)\left[\Sigma^<_L(E)+\Sigma^<_R(E)+\Sigma^<_{ep}(E)\right]G^A(E).
\]

The converged zero-pulse kernel is (̄\Sigma_0=(\Sigma^R_{ep},\Sigma^<_{ep},\Sigma_H)) and is used to initialize PS-SCBA. The optional historical preparation-fixed comparison instead constructs the stationary kernel for the selected protocol preparation (biased for the downward reference); it keeps that kernel fixed and never feeds the projected result back.

## Retarded pole extraction used by this implementation

The code uses a hybrid pole pipeline; it does not call the real-frequency
`MiniPoleRf` class directly.

| Input available in a run | Extractor used | Purpose |
|---|---|---|
| Stationary SCBA spectral grid | upstream `MiniPole` on Matsubara samples | fit the dynamic retarded electron--phonon self-energy |
| Projected SCBA real-frequency grid | causal SciPy AAA candidate fit followed by weighted least-squares residue refitting | obtain a causal rational self-energy when the projected kernel is tabulated |
| Rational self-energy plus Lorentzian leads | algebraic arrowhead eigenproblem and analytic denominator derivative | obtain physical unbiased and biased Green-function poles and residues |

For the stationary path, the real-axis interaction kernel is transformed to
fermionic Matsubara samples and fitted as

\[
\Sigma^R_{\mathrm{ep,dyn}}(z)
\simeq \sum_j \frac{a_j}{z-\zeta_j},
\qquad \operatorname{Im}\zeta_j<0.
\]

For a projected kernel, the stored real-frequency samples are fitted directly
by the causal AAA fallback. Candidate poles above the real axis are rejected,
the remaining residues are refit on the full grid, and the fit is accepted only
after self-energy and Green-function reconstruction tests pass. This fallback
is a rational approximation strategy, not the upstream `MiniPoleRf` algorithm.

The physical Green poles are then *not* fitted a second time. They are roots of
the rational Green-function denominator and their residues are computed from
the derivative of that denominator. The `PoleCache` containing these poles and
residues is reused for every time tile, source sector, and current evaluation
within one installed-kernel trajectory. The coefficient-plan and phase reuse
around this cache is an ongoing performance optimization; the cache itself is
rebuilt whenever the SCBA kernel changes.

The upstream MiniPole project provides both Matsubara and real-frequency
variants. The real-frequency `MiniPoleRf` variant requires callable analytic
functions and conformal-map contour moments. Our projected kernels are finite
real-axis arrays, so applying `MiniPoleRf` would first require an interpolated
analytic evaluator. The implementation therefore uses the direct causal AAA
route for those arrays and records the method in the pole diagnostics.

## Installed-kernel fixed-point algorithm

For iteration (n), install (̄\Sigma_n) and create fresh iteration-local state.

1. Extract a causal rational representation of the dynamic retarded component
   \[
   \Sigma^R_{dyn}(z)\simeq\sum_p\frac{r_p}{z-\zeta_p},
   \qquad \operatorname{Im}\zeta_p<0.
   \]
   The extractor is stationary `MiniPole` or projected-kernel causal AAA as
   described above; physical Green poles are subsequently obtained
   algebraically from the installed rational kernel.
2. Construct unbiased and protocol-biased physical Green-function poles from the installed kernel.
3. Evaluate protocol amplitudes (A) and (C). Every residue term is evaluated as a blocked contraction
   \[
   \sum_pK_{pm}e^{-i(\zeta_p-E_m)t_b}
   =e^{iE_mt_b}\left[K^\mathsf{T}P\right]_{mb},
   \qquad P_{pb}=e^{-i\zeta_pt_b}.
   \]
   Energy/time tiles preserve the residue equations while avoiding repeated scalar-time work.
4. Accumulate the lead and interaction source histories \(B\), \(D\), and \(\Psi\), the analytic Keldysh source terms for the selected switching protocol.
5. Reconstruct \(G^R(t,t')\) and \(G^<(t,t')\). Negative observation times use stationary endpoint data from the same installed kernel. A square pulse uses its pre-turnoff branch for \(t\le s\) and analytic post-turnoff history for \(t>s\).
6. Apply the uniform stationary projector. With \(T=(t+t')/2\) and \(\tau=t-t'\),
   \[
   (\mathcal PG)(\tau)=\frac{1}{L_T}\int_{T\in\mathcal W}G(T+\tau/2,T-\tau/2)\,dT,
   \]
   discretized as a weighted average over equal-time diagonals. For a lesser source factorization
   \[
   \Sigma^<(E)=\sum_s w_s(E)|v_s(E)\rangle\langle v_s(E)|,
   \quad G^<(t,t')=\sum_s\int dE\,w_s(E)a_s(t,E)a_s^*(t',E),
   \]
   the diagonal average is an autocorrelation of \(a_s\). Zero-padded FFT correlations evaluate it without storing a dense \(N_t\times N_t\) matrix during ordinary iterations.
7. Construct projected self-energy candidates. Writing \(G^>=G^<+G^R-G^A\),
   \[
   \Sigma^<(E)=g_q^2[(N_0+1)G^<(E+\omega_0)+N_0G^<(E-\omega_0)],
   \]
   \[
   \Sigma^>(E)=g_q^2[(N_0+1)G^>(E-\omega_0)+N_0G^>(E+\omega_0)],
   \]
   \[
   \Gamma_{ep}=i(\Sigma^>-\Sigma^<),\qquad
   \Sigma^R_{dyn}=\mathcal H[\Gamma_{ep}]-\frac{i}{2}\Gamma_{ep},
   \qquad \Sigma_H=-\frac{2g_q^2}{\omega_0}n.
   \]
8. Compute weighted installed residuals \(R_\Sigma^R,R_\Sigma^<\), mix retarded, lesser, and Hartree components with the configured linear rate \(\alpha\), and atomically checkpoint the new kernel. Pulay, Anderson, and Broyden acceleration are not used.
9. Re-run the pulse with the mixed kernel, evaluate the installed residual, and accept only when the configured tolerance and observable gates pass.

### Explicit and generic lesser representations

The explicit phonon-branch form is the SCBA construction. For a mixed
installed lesser kernel, the inner solve instead uses

\[
G_{\rm ep}^<(t,t')=\int\frac{dE}{2\pi}
e^{-iE(t-t')}C(t,0,E)\bar\Sigma_{\rm ep}^<(E)C^\dagger(t',0,E).
\]

Substituting the two shifted terms of \(\bar\Sigma_{\rm ep}^<\) and changing
variables \(E=\epsilon'+\eta\omega_0\) gives the explicit \(\eta=\pm1\)
expression. The code uses the generic form for mixed iterates; the explicit
branch contraction is exposed as a validation diagnostic and is exercised by
targeted finite-coupling scientific-consistency tests.

## Observables and validation

Lead currents are evaluated from the pole-energy \(A/C/\Psi\) expression with analytic stationary tails and piecewise-linear Filon oscillatory transforms. Production cases use streamed two-time reconstruction and reduced projection data. Dense two-time arrays are enabled only for explicitly selected validation cases; those cases additionally support collision/discarded-sector diagnostics and Green-function checks. Dense arrays are never integrated to produce production currents. For plotting, the raw pole/Filon current is shown together with the separately stored noninteracting conservation-corrected diagnostic view when available. For production diagnostics, the converged kernel is also used to evaluate both phonon branches \(C(t,-\omega_0,E')\) and \(C(t,+\omega_0,E')\), with weights \(N_0+1\) and \(N_0\), respectively. Their squared moduli are written as separate two-dimensional energy--time images. `current_time_points`, when set, selects a nested subset of existing nonnegative trajectory nodes including both endpoints; it is observable sampling, not an independent time quadrature.

The current-energy integration is converged independently in both range and
spacing. In particular, fast/direct agreement on one finite grid proves that
the two implementations evaluate the same discretized equation; it does not
prove that the grid represents the infinite energy integral. The raw
noninteracting continuity residual is therefore retained as a quadrature
diagnostic even when a separately stored conserving (g=0) view is produced.
Observable diagnostics record `current_energy_min/max`,
`current_energy_points`, `current_energy_step`, and the corresponding
`observable_time_points/step`, so range, spacing, and time-step studies can be
compared without inferring units from array lengths.
That view applies only the common-mode correction
\[
J_\alpha^{\rm corr}(t)=J_\alpha^{\rm raw}(t)-
\frac{\Gamma_\alpha}{\Gamma_L+\Gamma_R}
\left[J_L^{\rm raw}(t)+J_R^{\rm raw}(t)-\dot n(t)\right],
\]
so the transport (lead-antisymmetric) current is unchanged while the sum obeys
continuity to numerical precision. It is stored and plotted as a diagnostic,
not as a replacement for the raw pole/Filon archive product.

The primary occupation is the lag-zero projected lesser value,
\(n=-i(\mathcal PG)^<(0)\). An independent finite-grid diagnostic compares it
with \(-i\int dE\, (\mathcal PG)^<(E)/(2\pi)\); the difference is reported as
`projected_occupation_independent_error` and is used in energy-window and
energy-step convergence studies. The collision functional is evaluated on the same finite window and verifies

\[
J_L+J_R-e\dot N=-e\,\mathcal C[G,\delta\Sigma_{ep}].
\]

Because the switch is applied to a steady nonequilibrium state, the collision
integral also contains `t' < t_min`. For dense validation trajectories the
implementation evaluates this omitted part from stationary pre-pulse source
factors and the same auxiliary-pole propagators used for post-pulse rows,
using a mapped semi-infinite quadrature. The projected Green functions in this
tail are evaluated from their stationary representation at every lag; they are
not interpolated from the finite stored lag window. Changing `t_min` therefore changes
only the stored representation and provides a measurable window-sensitivity
test; it does not assume that the system was empty before the first sample. The
finite-window and prehistory-tail contributions are stored separately so that
quadrature error can be distinguished from a missing-history error.

Hard acceptance gates for explicitly dense validation cases require fast/direct agreement below \(10^{-5}\), dense/streaming projection agreement below \(10^{-10}\), retarded causality below \(10^{-12}\), lesser Hermiticity below \(10^{-10}\), installed residual below tolerance, collision identity mismatch below \(10^{-4}J_{\rm scale}\), and observable/kernel convergence limits. The identity mismatch is distinct from the collision-source magnitude and \(R_G,R_\Sigma\), which quantify the omitted PS-SCBA sector and are reported rather than assigned a universal pass/fail threshold. Streamed production cases apply the installed-kernel, pole, and observable gates while omitting dense-only discarded-sector checks.

## State ownership and outputs

`CaseWorkspace` owns grids, distributions, quadrature, transforms, and memory estimates. One installed-kernel solver owns its MPM and physical-pole cache. One `SCBAIteration` owns amplitudes, histories, Green functions, projection, candidate kernel, and residuals. Disk output retains scalar history and the latest resumable kernel rather than every dense iteration.

Each run writes resolved configuration, Git/runtime provenance, units, progress and convergence history, raw physical currents, optional conservation-corrected current views, occupation, kernels, diagnostics, and deterministic SVG figures. Natural-unit arrays are the numerical source of truth; storage readers normalize physical files back to natural units. The compatibility field `current_representation` does not select an evaluator: `finite_window` always means the pole/Filon implementation, while `direct_oracle` selects the literal reference path.
