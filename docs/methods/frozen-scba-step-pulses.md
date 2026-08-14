# Frozen-SCBA finite-bandwidth step-pulse transport

## Problem addressed

This method computes nonlinear transient quantum transport after a sudden
upward or downward one-body voltage shift when lead memory is retained through
a Lorentzian finite bandwidth and electron--phonon effects are approximated by
a stationary self-energy frozen at the switching boundary.

## Physical/mathematical formulation

The implementation is a partition-free, connected lead--device--lead problem.
A downward step begins in a biased stationary state and removes the shifts at
\(t=0\). An upward step begins in an unbiased stationary state and applies the
shifts at \(t=0\). The electronic Green function is time dependent; the same
\(\overline\Sigma_{\rm ep}^{R,<}(t-t')\) is retained throughout the pulse.

## Core method

The v26 derivation reduces the retarded Volterra problem to stationary
propagators plus finite Green-pole sums for the transforms \(A_\alpha\) and
\(C\). Time-integrated transforms \(B\) and \(D\) assemble lead and frozen
electron--phonon lesser histories. The current is

\[
J_\alpha(t)=-2e\int\frac{d\epsilon}{2\pi}\Gamma_\alpha(\epsilon)
\operatorname{Im}[\Psi_\alpha(\epsilon,t)+f_\alpha(\epsilon)A_\alpha(\epsilon,t)].
\]

## Key equations

The defining equations, the upward and downward residue formulas, and boundary
identities are summarized in the root README. The authoritative repository
derivation is v26, sections “Downward Step Pulse,” “Upward Pulse,” and their
Lorentzian-dot specializations.

## Algorithm

1. Choose the protocol-specific stationary reference.
2. Construct a weak-Born or self-consistent stationary electron--phonon kernel.
3. Freeze its retarded and lesser components.
4. Build unbiased and biased stationary retarded propagators using the same
   minimal-pole self-energy representation.
5. Evaluate the protocol's \(A,B,C,D\) functions and integrate the lead plus
   frozen-electron--phonon lesser current.
6. Report occupation, charge-continuity residual, and fit/boundary diagnostics.

## Assumptions

The code assumptions in the stationary-kernel page apply. In addition, voltage
histories are sudden piecewise-constant shifts, the leads and device are
initially connected, and the one-body energies follow the prescribed shifts.
The electron--phonon self-energy remains stationary during the transient.

## Applicability

Use this approach when finite-bandwidth lead memory is important, a step
voltage is the relevant drive, and freezing a well-defined stationary
electron--phonon kernel is an acceptable approximation. A downward calculation
is appropriate for switch-off from a biased stationary boundary; an upward
calculation is appropriate for switch-on from an unbiased boundary.

## Limitations

Do not use it as a substitute for fully time-dependent SCBA when the applied
field substantially changes the electronic distribution entering the
self-energy, when phonon occupation evolves, or when electron--phonon feedback
during the pulse is essential. The code is restricted to the scalar
Lorentzian-dot model and does not implement arbitrary waveforms, multiorbital
devices, arbitrary lead bandstructures, or an exact interacting solution.
Long-time dephasing arguments in v26 assume integrable frequency amplitudes
and no undamped real-axis bound-state poles.

## Relationship to existing methods

Maciejko, Wang, and Guo developed the finite-bandwidth partition-free exact
nonlinear electronic pulse theory. The repository manuscript explicitly
describes the present construction as an extension that adds a stationary
electron--phonon SCBA self-energy and then freezes it. MPM is a separate
published method used for numerical retarded reduction.

## What this work contributes

The repository evidence supports the contribution statement “frozen-SCBA
extension of the Maciejko finite-bandwidth pulse construction” for upward and
downward steps. Its publication status, priority, scope relative to other
frozen-self-energy approaches, and canonical paper identifier require author
verification.

## Implementation in this repository

- [`backend/observables.py`](../../backend/observables.py): `A_down`, `B_down`,
  `C_down`, `D_down`, upward counterparts, currents, occupation, diagnostics.
- [`backend/SCBA.py`](../../backend/SCBA.py): stationary kernel.
- [`backend/minimal_poles.py`](../../backend/minimal_poles.py): retarded pole
  representation.
- [`runner_downward_full_scba.py`](../../runner_downward_full_scba.py) and
  [`runner_upward_full_scba.py`](../../runner_upward_full_scba.py): dedicated
  self-consistent configurations.

## Validation and benchmarks

The test suite verifies initial/asymptotic boundaries, \(B(0)=D(0)=0\), direct
real-axis comparisons for selected \(A,C\) points, current finiteness, protocol
dispatch, and nonzero-coupling MPM use. The phonon-free pole limits recover the
two unbiased and three biased Maciejko poles. Archived sweep outputs are
reproducibility artifacts, not a documented external benchmark or experimental
validation set.

## References

J. Maciejko, J. Wang, and H. Guo, “Time-dependent quantum transport far from
equilibrium: An exact nonlinear response theory,” *Physical Review B* 74,
085324 (2006), DOI: [10.1103/PhysRevB.74.085324](https://doi.org/10.1103/PhysRevB.74.085324).

The associated frozen-SCBA manuscript is
[`E-phonon_Keldysh_frozen_SCBA_refined_proofs_v26.tex`](../../E-phonon_Keldysh_frozen_SCBA_refined_proofs_v26.tex);
no DOI or arXiv identifier is present in the repository.

## How to cite this method

If using or independently reimplementing the finite-bandwidth frozen-SCBA step
construction, cite both the Maciejko--Wang--Guo foundation and the publication
that introduces the frozen-SCBA extension. The latter citation is currently a
TODO pending author-supplied publication metadata. Cite Zhang--Yu--Gull when
using MPM, and cite this software separately when using this implementation.
