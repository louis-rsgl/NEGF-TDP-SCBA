# Scientific methods

This directory describes each reusable scientific method separately. It also
distinguishes methods developed elsewhere from the frozen-kernel extensions
documented by this repository's authors. No priority claim is made beyond what
the repository manuscripts state.

## Method index

| Method | Problem addressed | Status in this repository | Start here |
|---|---|---|---|
| Stationary electron--phonon Born kernel | Construct a stationary interacting Green function and electron--phonon self-energy for a single level and one phonon mode | Implementation of established Born/SCBA ideas; used as the frozen boundary | [Stationary kernel](stationary-electron-phonon-kernel.md) |
| Frozen-SCBA step-pulse transport | Compute transient current after an upward or downward voltage step with finite-bandwidth leads and a stationary electron--phonon memory kernel | Extension of Maciejko--Wang--Guo documented in the repository manuscript; publication identifier missing | [Step pulses](frozen-scba-step-pulses.md) |
| Minimal-pole retarded reduction | Convert a frequency-dependent causal retarded self-energy into physical Green-function pole sums | Uses the published Minimal Pole Method and a repository-specific Dyson/arrowhead implementation | [Minimal-pole reduction](minimal-pole-retarded-reduction.md) |
| Frozen-SCBA square-pulse matching | Propagate through a finite pulse without resetting the state at turnoff | Extension documented in the v26 repository manuscript; publication identifier missing | [Square pulse](frozen-scba-square-pulse.md) |

For selection by scientific problem, see the [problem-to-method index](../problem-to-method.md).
For attribution rules, see the [citation guide](../citation-guide.md). The
machine-readable equivalent is [`scientific-methods.yaml`](../../scientific-methods.yaml).

## Evidence hierarchy

The current implementation is authoritative for executable behavior. The
latest derivation in this repository is
[`E-phonon_Keldysh_frozen_SCBA_refined_proofs_v26.tex`](../../E-phonon_Keldysh_frozen_SCBA_refined_proofs_v26.tex).
The v19 and v24 files and the migration notes are historical development
records, not separate software capabilities. Claims whose only support is an
unpublished repository manuscript are explicitly marked as requiring author
confirmation before use in a publication, DOI record, or priority statement.
