# Minimal-pole retarded reduction and Green-pole extraction

## Problem addressed

The transient residue formulas require meromorphic retarded propagators, while
the stationary electron--phonon self-energy is available as frequency-dependent
real-grid data. This method approximates only the causal dynamic retarded
self-energy, reconstructs the stationary propagators through Dyson's equation,
and extracts their physical poles and residues.

## Physical/mathematical formulation

Auxiliary upper-half-plane samples are spectral transforms of the stationary
electron--phonon density:

\[
\Sigma_{\rm ep,dyn}^R(i\nu_n)=\int\frac{d\omega}{2\pi}
\frac{\Gamma_{\rm ep}(\omega)}{i\nu_n-\omega},\qquad
\nu_n=(2n+1)\pi/\beta_{\rm fit}.
\]

## Core method

The published Minimal Pole Method (MPM) supplies

\[
\Sigma_{\rm ep,dyn}^R(z)\simeq\sum_j\frac{s_j}{z-\zeta_j},
\qquad \operatorname{Im}\zeta_j<0.
\]

The same fit is inserted into both unbiased and biased Dyson denominators. A
non-Hermitian arrowhead eigenproblem produces Green-function poles \(\xi_r\).
Their analytic residues are the reciprocal derivative of the Dyson
denominator. The self-energy poles \(\zeta_j\) and Lorentzian embedding poles
are zeros, not poles, of the reconstructed Green function and are not inserted
into transient Green-pole sums.

## Key equations

For grouped lead shifts \(d_m\) with numerators \(a_m\),

\[
R_r=\left[1+\sum_m\frac{a_m}{(\xi_r-d_m+iW)^2}
+\sum_j\frac{s_j}{(\xi_r-\zeta_j)^2}\right]^{-1}.
\]

## Algorithm

1. Build upper-half-plane spectral samples.
2. Run the pinned MiniPole implementation.
3. Reject non-negligible upper-half-plane self-energy poles.
4. If MiniPole fails validation, use SciPy AAA only to propose pole locations,
   discard upper-half-plane candidates, and causally refit lower-plane residues.
5. Build unbiased and biased arrowhead problems from the same fitted self-energy.
6. Remove negligible pole--zero pairs and reject causal or degeneracy failures.
7. Validate self-energy and both Green reconstructions against the full
   stationary grid.

## Assumptions

The dynamic self-energy must admit an accurate finite rational approximation
over the relevant window. Simple-residue formulas require retained Green poles
not to be numerically degenerate. The Hartree term remains an explicit static
shift. Lesser functions are never pole-fitted.

## Applicability

Use this reduction when a causal frequency-dependent retarded kernel must be
evaluated repeatedly in transient contour or residue formulas. It becomes
advantageous when a validated small pole set replaces expensive repeated
retarded frequency integrations.

## Limitations

This does not make the frozen-kernel physical approximation exact. A fit that
passes only on too narrow a frequency grid can still miss relevant structure.
The implementation aborts rather than returning a result when configured
reconstruction, causality, pole-conditioning, or boundary gates fail. It is a
scalar implementation; matrix-valued MPM is not exposed by this code.

## Relationship to existing methods

MPM originates with Zhang, Yu, and Gull, not with this repository. SciPy's AAA
rational approximation is used as a fallback candidate generator. The
arrowhead Dyson reconstruction and its use for the two frozen propagators are
implementation choices in this repository. Any claim that this combination is
methodologically novel requires author and literature verification.

## What this work contributes

Supported implementation contributions are: fitting only the dynamic retarded
sector; preserving real-grid lesser data; reconstructing two Green-pole sets
from one self-energy fit; excluding self-energy poles from Green sums; and
applying joint self-energy, Green, causality, conditioning, and boundary gates.

## Implementation in this repository

- [`backend/minimal_poles.py`](../../backend/minimal_poles.py): spectral
  samples, MiniPole/AAA fitting, arrowhead eigenproblem, residues, validation.
- [`backend/observables.py`](../../backend/observables.py): step-pulse residue
  sums and direct-quadrature checks.
- [`backend/square_pulse.py`](../../backend/square_pulse.py): square-pulse
  internal and outer residue evaluation.

## Validation and benchmarks

[`tests/test_minimal_poles.py`](../../tests/test_minimal_poles.py) tests Dyson
roots, residues, the two/three-pole phonon-free limits, causality and degeneracy
failures, adaptive AAA work limits, and nonzero-coupling reconstruction.
Transient tests compare selected residue expressions with defining real-axis
quadrature. Quality JSON records report fit method, pole counts, and scaled
errors. No performance benchmark against alternative analytic-continuation
methods is established in the repository.

## References

L. Zhang, Y. Yu, and E. Gull, “Minimal pole representation and analytic
continuation of matrix-valued correlation functions,” *Physical Review B* 110,
235131 (2024), DOI: [10.1103/PhysRevB.110.235131](https://doi.org/10.1103/PhysRevB.110.235131).

## How to cite this method

If MPM is used or reimplemented, cite Zhang, Yu, and Gull even if this software
is not used. If this repository's reduction or implementation is used, cite
the software separately. Citation guidance for the causal AAA adaptation is
an unresolved author-metadata item.
