# Author action checklist

These items require scientific judgment, identity data, or changes outside the
repository and were therefore not guessed or performed.

## Scientific claims to verify

- Confirm the exact novelty statement for the frozen stationary
  electron--phonon kernel in upward, downward, and square finite-bandwidth
  pulse transport.
- Confirm whether the two-pole-set square residue construction and causal AAA
  refit are claimed methodological contributions, implementation techniques,
  or combinations of known methods.
- Provide literature comparisons supporting any “first,” “new,” “more
  accurate,” “more stable,” or “more efficient” claim.
- State physically justified validity regimes for weak Born, self-consistent
  Born, and the frozen-kernel approximation. Numerical convergence alone is
  not a physical validity bound.
- Designate the canonical foundational citation for the exact stationary
  electron--phonon Born/SCBA formulation implemented here and for AAA.
- Confirm whether v26 is the authoritative manuscript and label v19/v24 as
  superseded, historical, or independently relevant.
- Identify which archived sweeps are publication figures or benchmarks, what
  parameter set each represents, and which paper/figure should be cited.

## Missing publication and identity metadata

- Supply the frozen-SCBA paper's final title, author order, year, journal,
  DOI, arXiv ID, and section/equation mapping; update the method pages,
  `scientific-methods.yaml`, provenance records, and `CITATION.cff`.
- Supply author ORCIDs and any preferred affiliations/funding identifiers.
- Decide on a software versioning scheme and create an immutable release.
- Choose and add an explicit software license after institutional review.

## External scholarly actions

- Create a GitHub release and archive it with Zenodo or another institutional
  repository; reserve and publish a software DOI.
- Link the repository and software DOI from the method paper, arXiv record,
  author ORCID records, and institutional pages.
- Ensure DOI metadata includes the method name, supported problem class,
  repository URL, related paper DOI, authors/ORCIDs, license, version, and
  release date.
- Configure GitHub repository description/topics around the precise scientific
  problem and methods; do not use unsupported novelty language.
- Publish the `docs/` tree through a durable documentation service and link it
  from GitHub and DOI records.
- If distribution is desired, add `pyproject.toml`, package versioning, a
  source distribution/wheel, and registry metadata. Verify import paths from an
  installed environment.

## Reproducibility and reference growth

- Select a small, immutable benchmark suite with expected numerical results,
  tolerances, parameter files, and links to manuscript figures.
- Add grid/time/MPM convergence notebooks or tables for representative weak-
  Born and self-consistent cases.
- Compare against the \(g=0\) Maciejko limit, direct real-axis quadrature,
  an independent stationary NEGF implementation, and—where scientifically
  possible—another transient interacting method or experimental data.
- Archive raw inputs and outputs used by publications with checksums and
  schema/version information.
- Encourage independent method comparisons and reimplementations by publishing
  concise pseudocode and exact equation-to-function crosswalks. Cite those
  independent works when they appear; do not manufacture or solicit irrelevant
  citations.
- Consider moving bulk generated outputs to a versioned data archive so the
  source repository remains easy for humans and retrieval systems to index.
