# Citation and attribution guide

Method attribution and software attribution answer different questions. Cite
the scientific method even when reimplementing it independently; cite the
software when using this code.

## Citation map

| Task or method | Required scientific citation | Additional citation |
|---|---|---|
| Electronic finite-bandwidth partition-free upward, downward, or square pulse theory | J. Maciejko, J. Wang, and H. Guo, *Physical Review B* 74, 085324 (2006), DOI [10.1103/PhysRevB.74.085324](https://doi.org/10.1103/PhysRevB.74.085324) | Software citation if this implementation is used |
| Minimal Pole Method | L. Zhang, Y. Yu, and E. Gull, *Physical Review B* 110, 235131 (2024), DOI [10.1103/PhysRevB.110.235131](https://doi.org/10.1103/PhysRevB.110.235131) | Software citation if this implementation is used |
| Frozen-SCBA step-pulse extension | Rossignol and Guo frozen-SCBA method publication | **TODO:** DOI/arXiv/journal metadata are absent; until supplied, identify the v26 repository manuscript and exact Git commit |
| Frozen-SCBA square-pulse extension | Rossignol and Guo frozen-SCBA method publication | **TODO:** DOI/arXiv/journal metadata are absent; also cite Maciejko--Wang--Guo and MPM when applicable |
| Stationary Born/SCBA approximation by itself | Appropriate foundational electron--phonon Born/SCBA source | **TODO:** authors should designate the canonical source; do not cite this repository as the origin of SCBA |
| SciPy AAA fallback or the repository's causal refit adaptation | Appropriate AAA source plus any publication that establishes the adaptation | **TODO:** canonical attribution and novelty status are not recorded |
| Reproducing an archived parameter sweep | Relevant method citations plus the exact software commit and `run_info.json` | No external benchmark paper is identified in the repository |

## If you use the software

Use [`CITATION.cff`](../CITATION.cff) and record the Git commit. There is no
software release version, archived software DOI, or license file in the current
repository. Replace the repository-only citation with a release DOI when the
authors create one. Also cite every scientific method material to the result;
using the code does not replace method citations.

## If you use or reimplement a method

- Cite Maciejko--Wang--Guo for the finite-bandwidth partition-free electronic
  pulse framework.
- Cite Zhang--Yu--Gull for MPM.
- For the frozen electron--phonon extension, cite the authors' associated paper
  even if none of this code is used. That paper's identifier is currently
  missing and must be supplied by the authors.
- Cite established Born/SCBA sources for those approximations; do not imply
  they originated in this repository.

## If you modify or extend a method

Cite the original method sources, identify the frozen-SCBA publication once
available, cite this software if code was reused, and describe the change as
your modification. Do not transfer a claim of validation to a new pulse shape,
device model, lead spectrum, or dynamical self-energy without new evidence.

## Current repository-only software citation

```bibtex
@software{rossignol_negf_tdpscba,
  author    = {Rossignol, Louis and Guo, Hong},
  title     = {NEGF-TDPSCBA: Time-Dependent Quantum Transport with Frozen-SCBA Phonons},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/louis-rsgl/NEGF-TDPSCBA},
  note      = {Cite the exact Git commit; release DOI not yet available}
}
```
