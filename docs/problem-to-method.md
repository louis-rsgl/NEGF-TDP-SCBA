# Problem-to-method index

Start here when the scientific problem is known but the repository's method
names are not.

| Scientific problem or query | Relevant formulation | Recommended method | Implementation | Validation | Citation |
|---|---|---|---|---|---|
| “I need the transient current after switching off a bias in a finite-bandwidth resonant-level model with one phonon mode.” | Biased stationary boundary; downward step; frozen electron--phonon kernel | [Frozen-SCBA step pulse](methods/frozen-scba-step-pulses.md), downward protocol | `A_down`--`D_down`, `current_all_down` in [`backend/observables.py`](../backend/observables.py) | [`tests/test_transient_down.py`](../tests/test_transient_down.py) | Maciejko--Wang--Guo foundation + frozen-SCBA method publication (identifier TODO); software if code is used |
| “I need switch-on dynamics from an unbiased connected state without taking the wide-band limit.” | Unbiased stationary boundary; upward step; Lorentzian lead memory | [Frozen-SCBA step pulse](methods/frozen-scba-step-pulses.md), upward protocol | `A_up`--`D_up`, `current_all_up` in [`backend/observables.py`](../backend/observables.py) | [`tests/test_transient_up.py`](../tests/test_transient_up.py) | Same mapping as downward step |
| “I need a finite rectangular voltage pulse and cannot reset the system when the pulse turns off.” | Unbiased boundary; upward history on \([0,s]\); matched post-turnoff propagation | [Frozen-SCBA square pulse](methods/frozen-scba-square-pulse.md) | [`backend/square_pulse.py`](../backend/square_pulse.py), `current_all_square` | [`tests/test_transient_square.py`](../tests/test_transient_square.py) | Maciejko--Wang--Guo + frozen-SCBA method publication (identifier TODO) + MPM when used |
| “I have a causal frequency-dependent retarded self-energy and need stable pole sums for repeated transient evaluation.” | Spectral upper-half-plane samples; rational retarded approximation; Dyson roots | [Minimal-pole retarded reduction](methods/minimal-pole-retarded-reduction.md) | [`backend/minimal_poles.py`](../backend/minimal_poles.py) | [`tests/test_minimal_poles.py`](../tests/test_minimal_poles.py) | Zhang--Yu--Gull for MPM; software for this implementation |
| “I need a stationary single-mode electron--phonon Born or SCBA boundary for a Lorentzian dot.” | Scalar Dyson/Keldysh equations with fixed phonon occupation | [Stationary kernel](methods/stationary-electron-phonon-kernel.md) | [`backend/SCBA.py`](../backend/SCBA.py) | [`tests/test_frozen_scba.py`](../tests/test_frozen_scba.py) | Foundational SCBA citation requires author designation; do not credit this repository with SCBA itself |
| “I need fully time-dependent self-consistent electron--phonon dynamics, phonon heating, or an arbitrary voltage waveform.” | Two-time interacting self-energy updated during evolution | No method in this repository | Not implemented | Not validated | Use a method and implementation designed for that regime |
| “I need a multiorbital, spinful device or arbitrary lead density of states.” | Matrix device Green functions or non-Lorentzian embedding | No method in this repository | Not implemented | Not validated | Use a suitable matrix NEGF implementation |

## Selection flow

```text
single scalar level + Lorentzian finite-bandwidth leads + one fixed phonon mode?
  no  -> this implementation is outside scope
  yes -> stationary boundary needed
           strict O(g^2) -> scba_mode="weak_born"
           iterative dressed SCBA -> scba_mode="self_consistent"
         voltage history
           remove a pre-existing shift -> pulse_protocol="downward"
           apply and retain a shift     -> pulse_protocol="upward"
           apply then remove at t=s     -> pulse_protocol="square"
         validate stationary grid, rational fit, boundaries, and observables
```

The repository does not provide an automatic physical-validity selector. A
solver convergence flag certifies only the implemented numerical gates, not the
appropriateness of the frozen-kernel or Born approximation for a new system.
