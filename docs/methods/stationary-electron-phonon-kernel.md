# Stationary single-mode electron--phonon Born kernel

## Problem addressed

This method constructs the stationary retarded and lesser Green functions of
a scalar electronic level coupled to finite-bandwidth Lorentzian leads and one
stationary phonon mode. The resulting electron--phonon self-energy is the
boundary data frozen during the later pulse calculation.

## Physical/mathematical formulation

The implemented central region is a single, spinless level with energy
`e_0`, scalar coupling `g_q`, and phonon frequency `w_q`. Each lead has a
Lorentzian linewidth. A downward step uses a biased stationary reference;
upward and square pulses use an unbiased stationary reference.

The Dyson and Keldysh equations are

\[
G^R=[\omega-\epsilon_0-d_0-\Sigma_{\rm lead}^R-\Sigma_{\rm ep}^R+i\eta]^{-1},
\qquad
G^<=G^R(\Sigma_{\rm lead}^<+\Sigma_{\rm ep}^<)G^A.
\]

## Core method

For fixed occupation \(N_0\), shifted interpolation gives

\[
\Sigma_{\rm ep}^<(\omega)=g^2[(N_0+1)G^<(\omega+\omega_0)
+N_0G^<(\omega-\omega_0)].
\]

The code forms \(G^>=G^<+G^R-G^A\), constructs the corresponding greater
self-energy and spectral density
\(\Gamma_{\rm ep}=i(\Sigma_{\rm ep}^>-\Sigma_{\rm ep}^<)\), and obtains the
dynamic retarded part with a non-wrapping, linear-convolution principal-value
transform. The scalar Hartree term is
\(\Sigma_H=-2g^2\bar n/\omega_0\).

## Key equations

The full expressions and conventions are summarized in the root README and
derived in v26, Assumption A7 and the sections “Single-mode stationary SCBA
construction” and “Stationary SCBA construction of the frozen dot kernel.”

## Algorithm

`scba_mode="weak_born"` evaluates the complete \(O(g^2)\) kernel once from
the no-phonon stationary reference, followed by one dressed Dyson/Keldysh
update. `scba_mode="self_consistent"` iterates both \(G^R\) and \(G^<\), using
linear mixing and optional damped Pulay/DIIS acceleration. Absolute and
relative integrated-L2 residuals must pass together; spectral normalization
and frequency-window edge checks are then applied.

## Assumptions

- Scalar, spinless, single-level electronic region.
- One stationary phonon mode without anomalous coherence.
- Fixed phonon occupation: thermal when `N0=None`, otherwise prescribed and
  nonnegative.
- Common Lorentzian bandwidth parameter `W`; finite real-frequency grid.
- Energies are stored in units of the total linewidth scale \(\Gamma\).

## Applicability

Use this implementation when a stationary Born or self-consistent Born
electron--phonon boundary is appropriate for the scalar Lorentzian-dot model,
and when that boundary will remain fixed during a later voltage pulse. Use
`weak_born` for the explicitly noniterative \(O(g^2)\) construction; select
`self_consistent` only when iterative dressed SCBA is intended.

## Limitations

This is not a transient self-consistent electron--phonon solver. It does not
solve phonon kinetics, phonon heating, anomalous phonon coherence,
electron--electron interactions, spin, multiple device orbitals, multiple
phonon modes, or arbitrary lead spectral functions. Grid tails are modeled as
\(G^R(z)\to1/(z+i\eta)\) and \(G^<(z)\to0\); a window that truncates important
spectral weight is invalid. SCBA convergence is not evidence that the Born
approximation is physically accurate in a chosen strong-coupling regime.

## Relationship to existing methods

Born and self-consistent Born electron--phonon approximations predate this
repository. The v26 manuscript cites Frederiksen's 2004 and 2007 theses for
the strict \(O(g^2)\) specialization. The repository supplies an implementation
and integrates the stationary result with its frozen-pulse construction; it
does not establish that SCBA itself originated here.

## What this work contributes

The code contribution supported by inspection is a protocol-aware frozen
record (`FrozenSCBA`), explicit separation of Hartree and dynamic retarded
parts, storage of the exact lesser Green function used to form the kernel, and
failure gates for residuals, spectral weight, and grid edges. Whether any of
these implementation choices constitute a publishable methodological novelty
requires author verification.

## Implementation in this repository

- [`backend/SCBA.py`](../../backend/SCBA.py): `Solver`, `FrozenSCBA`, weak-Born
  update, self-consistent iteration, Hilbert transform, convergence gates.
- [`backend/system_classes.py`](../../backend/system_classes.py): `System`
  settings, reference selection, and cache lifecycle.
- [`backend/distribution.py`](../../backend/distribution.py): Fermi and Bose
  distributions.

## Validation and benchmarks

[`tests/test_frozen_scba.py`](../../tests/test_frozen_scba.py) checks gauge
shifts, the Hilbert-transform sign, the \(g=0\) solution, protocol-specific
references, thermal/nonthermal occupation, interpolation tails, and explicit
nonconvergence. Archived `*_scba` and `*_weak_born` directories contain run
artifacts, but the repository does not identify them as independently
published benchmarks or comparisons with experiment.

## References

See the Frederiksen theses and the general NEGF references listed in the v26
manuscript. A canonical citation specifically for the stationary approximation
used here has not been designated in repository metadata.

## How to cite this method

Do not cite this repository as the origin of Born or SCBA. Cite the appropriate
foundational electron--phonon SCBA literature for the stationary approximation
and, when the kernel is frozen during a pulse, also cite the frozen-SCBA method
paper once the authors provide its publication identifier. If this code is
used, cite the software separately. See the [citation guide](../citation-guide.md).
