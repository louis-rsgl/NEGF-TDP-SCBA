# Repository scientific and metadata audit

Audit date: 2026-08-14. This audit is based only on files and Git metadata in
the repository. It does not assert external publication or registry state.

## Scientific scope established from the repository

The code treats time-dependent nonequilibrium Green-function transport through
a scalar, spinless electronic level connected to finite-bandwidth Lorentzian
leads and coupled to one stationary phonon mode. It supports downward voltage
steps, upward voltage steps, and one finite square pulse. Stationary
electron--phonon effects are computed either as strict \(O(g^2)\) weak Born or
iterative SCBA, then frozen during the pulse. The transient retarded sector is
evaluated with finite Green-pole sums derived from a rational approximation to
the dynamic retarded self-energy. Currents include lead and frozen
electron--phonon lesser histories, and the code reports occupation and a charge
continuity diagnostic.

The governing ingredients are the scalar Dyson/Keldysh equations, Lorentzian
embedding self-energies, single-mode Born electron--phonon self-energies, a
spectral/Hilbert construction of the retarded dynamic part, partition-free
pulse transforms, and lower-half-plane residue evaluation. The detailed
equations are in the root README and the v26 manuscript.

## Reusable methods identified

1. Stationary single-mode weak-Born/self-consistent-Born kernel construction.
2. Frozen electron--phonon extension of finite-bandwidth upward/downward pulse
   transport.
3. Minimal-pole retarded self-energy reduction plus physical Green-pole
   extraction from a Dyson arrowhead eigenproblem.
4. Two-pole-set frozen-kernel square-pulse matching and internal residue
   evaluation.

Stable supporting techniques include non-wrapping shifted interpolation,
linear-convolution Hilbert transformation, joint retarded/lesser DIIS mixing,
the removable-limit `expc` function, exact integration of a linear interpolant
against a Cauchy kernel, and failure-gated output. These are implementation
techniques; the repository does not establish them as independent novel
scientific methods.

## Assumptions and supported regimes

- Connected partition-free initial state and sudden piecewise-constant shifts.
- Scalar single level, one fixed phonon mode, and Lorentzian finite-bandwidth
  leads sharing `W`.
- Fixed thermal or prescribed nonthermal phonon occupation.
- Frozen stationary electron--phonon self-energy during the complete pulse.
- Finite real-frequency and current-integration windows.
- CPU NumPy/SciPy execution; optional process parallelism across independent
  parameter points.

The repository does not support a time-updated interaction self-energy,
phonon dynamics/heating, arbitrary waveforms, multiple orbitals or modes,
spin, electron--electron interactions, arbitrary lead spectra, GPU, or MPI.

## Prior art and repository-associated contributions

The finite-bandwidth partition-free electronic pulse foundation is explicitly
attributed to Maciejko, Wang, and Guo (2006). MPM is explicitly attributed to
Zhang, Yu, and Gull (2024). Born/SCBA is established prior art; the repository
manuscript cites Frederiksen theses for the strict \(O(g^2)\) specialization.

The v26 manuscript describes its work as extending the Maciejko construction
with a stationary electron--phonon SCBA kernel frozen at the pre-pulse boundary
and develops upward, downward, and square-pulse forms. This supports describing
the repository as an implementation of that stated extension, but publication
status, priority, novelty relative to all literature, and a canonical DOI or
arXiv identifier cannot be established from the repository.

## Implementation, validation, and artifacts

- `backend/SCBA.py`: stationary weak-Born/SCBA and frozen record.
- `backend/minimal_poles.py`: MPM/AAA reduction, Dyson reconstruction, Green
  poles, residues, and validation.
- `backend/observables.py`: step transforms, currents, occupation, diagnostics.
- `backend/square_pulse.py`: post-turnoff matching and internal residues.
- `runner.py`: physical-unit boundary, parameter sweeps, per-job failure and
  quality files, plots, and NPY output.
- `tests/`: stationary, pole, protocol, units, boundary, quadrature, current,
  and runner tests.

The repository includes hundreds of NPY, SVG, log, quality, and failure
artifacts for parameter sweeps. These demonstrate exercised configurations and
record validation gates, but no file declares an external benchmark dataset,
independent code comparison, experiment, uncertainty analysis, or peer-reviewed
validation result. The continuity residual is diagnostic rather than an
acceptance gate.

## Existing scholarly metadata

Established metadata found:

- Authors: Louis Rossignol and Hong Guo, McGill University.
- Repository URL: `https://github.com/louis-rsgl/NEGF-TDPSCBA`.
- Maciejko--Wang--Guo DOI: `10.1103/PhysRevB.74.085324`.
- Zhang--Yu--Gull DOI: `10.1103/PhysRevB.110.235131`.
- Zhang--Erpenbeck--Yu--Gull DOI in v26: `10.1063/5.0273763`.
- MiniPole source pinned to Git commit
  `15e4a541d652d2584fd4680413d11b5571f78d9b`.

Missing from the repository at audit time: frozen-SCBA paper DOI/arXiv/journal
metadata, software DOI, release version/tags, ORCIDs, license, package metadata,
documentation deployment, archive metadata, and authoritative benchmark
identifiers. `CITATION.cff` was added conservatively without inventing those
fields.

## Modification plan and disposition

### A. Safe documentation and metadata changes

Implemented: method pages, problem-to-method index, citation guide,
reproducibility/API guide, this audit, author-action checklist,
`scientific-methods.yaml`, `CITATION.cff`, and a more discoverable README entry
point.

### B. Scientific claims requiring author verification

Not asserted as fact: priority, novelty relative to the full literature,
quantitative physical validity boundaries, comparison with alternative
interacting transient solvers, and publication/benchmark status. These are
listed in the author-action checklist.

### C. Code/API improvements

Implemented only a backward-compatible provenance interface:
`System.provenance()`, `System.citations()`, and enriched future
`run_info.json` metadata. No equations, defaults, solver paths, or numerical
acceptance gates were changed. Larger packaging/API redesigns were not made.

### D. External actions

No external record was changed. Release, DOI, ORCID, repository-topic,
documentation-hosting, and registry work remains an author action.

## Discoverability obstacles that remain

The main obstacles are the missing method-publication identifier and release
DOI, absence of a license, no tagged releases, no installable package metadata,
no documentation website, historical manuscripts without a declared status
banner, and a repository dominated by generated output files. Search and
scholarly systems may also conflate the published prior methods with the
repository-associated frozen-kernel extension unless the authors supply a
precise contribution statement and citation record.
