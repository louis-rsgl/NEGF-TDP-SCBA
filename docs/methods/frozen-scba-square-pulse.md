# Frozen-SCBA finite square-pulse matching

## Problem addressed

This method computes the transient current produced by switching a finite
voltage pulse on at \(t=0\) and off at \(t=s\), retaining the state and memory
accumulated during the pulse rather than resetting the device at turnoff.

## Physical/mathematical formulation

The initial boundary is the unbiased stationary interacting state. The same
stationary electron--phonon self-energy is frozen across both switching events.
For \(0\le t\le s\), the square-pulse solution is exactly the upward-step
solution restricted to that interval. For \(t>s\), the full upward history is
matched to propagation under the unbiased frozen propagator.

## Core method

The post-turnoff outer residue sum uses unbiased Green poles, while its
amplitudes retain the biased Green poles that describe the pulse-window
history. Internal meromorphic frequency integrals are evaluated with

\[
\mathfrak R_-[F]=-
\sum_{p\in\mathcal P_-(F)}\operatorname{Res}_{z=p}F(z).
\]

Candidate kinematic, lead, and biased Green poles are kept until the complete
integrand is assembled; removable cancellations are then detected by small
contour integrals.

## Key equations

For \(t>s\),

\[
A_\alpha^\square(\epsilon,t)=G_{\rm fr}^R(\epsilon)
-e^{i\Delta_\alpha s}\sum_rR_r^0e^{-i(\xi_r^0-\epsilon)t}
\mathcal S_\alpha^\square(\xi_r^0,\epsilon;s),
\]

with an analogous expression for \(C^\square\). Explicit \(B^>\) and \(D^>\)
finite sums provide the post-turnoff lesser histories. Full formulas are in
v26, sections “Square Pulse” and “Quantum Dot Model with Lorentzian Linewidth:
Square Pulse.”

## Algorithm

1. Build the unbiased frozen kernel and one minimal-pole self-energy fit.
2. Extract unbiased and biased Green-pole sets from that fit.
3. Precompute square-history amplitudes on the current energy grid.
4. Numerically evaluate and validate complete-integrand internal residues.
5. Use upward formulas through \(s\), then post-turnoff unbiased pole sums.
6. Combine stored upward and new post-turnoff lesser histories in the current.

## Assumptions

All assumptions of the stationary and step methods apply. The pulse duration is
finite and nonnegative, the off-state equals the initial one-body Hamiltonian,
and the time grid contains the turnoff time exactly.

## Applicability

Use this approach for pump/probe-like finite rectangular voltage histories in
the scalar Lorentzian-dot model when the frozen-kernel approximation is
acceptable and state continuity through the second switch matters.

## Limitations

Only a single square pulse is implemented; arbitrary pulse trains, ramps, and
smooth envelopes are unsupported. The second switch does not trigger a new
SCBA calculation, so the method is inappropriate when the phonon or
electron--phonon self-energy must respond dynamically. Internal residue work
can be expensive because it is performed across outer Green poles, energies,
and candidate singularities. Passing turnoff identities does not establish
accuracy outside the configured frequency and time grids.

## Relationship to existing methods

The construction retains the finite-bandwidth pulse framework of Maciejko,
Wang, and Guo, adds the frozen stationary electron--phonon kernel documented by
the repository authors, and uses the published MPM rational representation.

## What this work contributes

The v26 manuscript identifies the two-pole-set frozen-SCBA square construction,
including the internal lower-plane residue operator and continuity identities,
as its extension of the electronic square-pulse problem. Publication status,
priority, and comparison with other interacting square-pulse solvers require
author verification.

## Implementation in this repository

- [`backend/square_pulse.py`](../../backend/square_pulse.py): square kernels,
  complete-integrand residue evaluator, cached amplitudes, post-turnoff
  \(A,B,C,D\).
- [`backend/observables.py`](../../backend/observables.py): upward interval,
  stored histories, current, occupation, continuity, diagnostics.
- [`runner_square_full_scba.py`](../../runner_square_full_scba.py): dedicated
  self-consistent configuration.

## Validation and benchmarks

[`tests/test_transient_square.py`](../../tests/test_transient_square.py) checks
that the time grid contains \(s\), the square solution equals the upward one
through turnoff, \(A,C\) are continuous at \(s\), zero duration produces no
retarded transient, removable candidates cancel, lead-pole zeros remain
finite, and a small current calculation succeeds. Quality files report
internal-residue and turnoff scaled errors. The repository contains no
independently published square-pulse benchmark or experimental comparison.

## References

Use the Maciejko--Wang--Guo article and Zhang--Yu--Gull MPM article listed on
the other method pages. The frozen-SCBA square-pulse derivation is presently
available only as the v26 repository manuscript, without DOI or arXiv ID.

## How to cite this method

An independent implementation should cite the Maciejko--Wang--Guo foundation
and the authors' frozen-SCBA method publication once its identifier is supplied.
If MPM is used, cite Zhang--Yu--Gull. If this code is used, cite the software in
addition to those method papers.
