# Milestone 6 — First PyPI release

**Status:** done — `0.1.0` published to [PyPI](https://pypi.org/project/vorflow/) on 2026-09-30 · **Risk:** low · **Behavior change:** none (packaging/metadata only)
**Back to** [ROADMAP.md](../../ROADMAP.md)

## Goal

Publish `vorflow` to PyPI so users can `pip install vorflow`. The name is
**available on PyPI** (verified 2026-07-18: the registry returns 404 for
`vorflow`).

This milestone is being rehearsed on TestPyPI first. The rehearsal stops after
verifying `0.1.0rc1`; real PyPI publication remains a separate approval gate.

## Current state (release candidate prepared locally)

- Explicit PEP 621/639 metadata for `0.1.0rc1`, with Oscar Sanchez as first
  author and Oscar and rhugman as maintainers. rhugman is listed by name only
  (no contact email in package metadata).
- MIT SPDX metadata, dependency floors, keywords, repository links, changelog,
  and absolute README links suitable for package-index rendering.
- One canonical Basic Usage script that runs against an installed wheel.
- Cross-platform CI plus a Python 3.10 job for the six exact dependency floors.
- Automated wheel/sdist content and built-metadata validation.
- One tag-triggered workflow (`.github/workflows/release.yml`) that builds,
  tests and validates once, then publishes `vX.Y.ZrcN` tags to TestPyPI
  (protected `testpypi` environment) and `vX.Y.Z` tags to PyPI (protected
  `pypi` environment), both with short-lived OIDC credentials. Any other tag
  fails `scripts/check_dist.py` before upload.
- `__version__` resolved from installed metadata, with a neutral source-tree
  fallback instead of a duplicated release number.

A local rehearsal on 2026-09-30 (`develop` at 427e30e plus the release
workflow changes) passed Ruff, the full test suite (331 tests), an isolated
archive build, Twine checks, archive validation against `v0.1.0rc1`,
fresh-wheel installation, `pip check`, and the Basic Usage example outside the
source tree. `0.1.0rc1` has not been tagged or published yet, so the changes
listed under `[Unreleased]` since the first release-candidate preparation were
folded into its changelog section.

## Completed release preparation

- [x] Integrate `gmshflow_missing` into local `main` after full verification.
- [x] Prepare and validate the `0.1.0rc1` wheel and source distribution.
- [x] Smoke-test the wheel in a fresh environment outside the repository.
- [x] Add PyPI-facing installation documentation and absolute links.
- [x] Add and test runtime dependency floors without upper caps.
- [x] Record Oscar's primary authorship and name-only co-maintainer metadata
  for rhugman.
- [x] Add the changelog, modern licence metadata, and package keywords.
- [x] Add a Trusted Publishing workflow: rc tags to TestPyPI, final tags to
  PyPI.

## TestPyPI rehearsal (verified 2026-09-30)

`v0.1.0rc1` (33fcd30) was built, tested and published by `release.yml` after a
manual approval. The TestPyPI page renders the README, banner and links, and
its metadata matches `pyproject.toml`. A fresh venv installed
`vorflow==0.1.0rc1` from TestPyPI (dependencies from PyPI): `pip check` was
clean, `vorflow.__version__` was `0.1.0rc1`, and `examples/basic_usage.py`
generated 2372 cells.


- [x] Re-enable the `pytest` workflow on GitHub. It was disabled for
  inactivity (weekly `schedule` trigger), so it did not run on the
  `develop` -> `main` PR: `gh workflow enable python-app.yml`.
- [x] Get a green `pytest` run on the release PR and integrate `develop` into
  `main`.
- [x] Create the upstream `testpypi` GitHub environment (with a required
  reviewer) and the TestPyPI pending Trusted Publisher (workflow
  `release.yml`, environment `testpypi`; see the header of
  `.github/workflows/release.yml`).
- [x] Create and push the annotated `v0.1.0rc1` tag.
- [x] Review the GitHub build and manually approve the protected `testpypi`
  deployment.
- [x] Inspect the TestPyPI project page and install `0.1.0rc1` independently.
- [x] Record the result as **TestPyPI verified**. Real PyPI publication remains
  a separate approval gate.

## Final PyPI release (`0.1.0`, published 2026-09-30)

`v0.1.0` (46b07df) was built and tested by `release.yml` and published to PyPI
after a manual approval. PyPI refused the first upload attempt
(`invalid-publisher`: no Trusted Publisher matched the workflow's claims), so
nothing was uploaded. After the pypi.org publisher was set up, re-running the
failed job published the artifacts from the original build. A
fresh venv installed `vorflow` from PyPI: `pip check` was clean,
`vorflow.__version__` was `0.1.0`, and `examples/basic_usage.py` generated 2372
cells.

Only after the TestPyPI rehearsal is verified:

- [x] Create the upstream `pypi` GitHub environment (with a required
  reviewer and a `v*` tag rule).
- [x] Add the PyPI pending Trusted Publisher (workflow `release.yml`,
  environment `pypi`).
- [x] Bump `version` in `pyproject.toml` to `0.1.0`.
- [x] In `README.md`, replace the "not yet published on PyPI" installation text
  with `pip install vorflow`; keep the GLU note for Linux.
- [x] In `CHANGELOG.md`, add a dated `## [0.1.0] - YYYY-MM-DD` section above
  `[0.1.0rc1]` (listing any changes since the candidate, or stating that there
  were none) and its compare link.
- [x] Merge to `main`, then create and push the annotated `v0.1.0` tag from
  `main`.
- [x] Review the build and approve the protected `pypi` deployment.
- [x] Install `vorflow==0.1.0` from PyPI in a fresh environment, check
  `vorflow.__version__`, and mark this milestone **Done**.

## Phase 3 — Nice-to-have (can follow in later 0.x releases)

- [ ] **Python 3.13/3.14 in the CI matrix and classifiers.** The local dev
  env is already on 3.14, so it is implicitly supported but untested in CI.
- [ ] **Docs site** (mkdocs-material + API reference on GitHub Pages or
  ReadTheDocs) and a `Documentation` URL in `[project.urls]`. The README is
  sufficient for an alpha.
- [ ] **Citation/DOI** (`CITATION.cff` + Zenodo) — worthwhile for a
  research-audience package.
- [ ] **conda-forge feedstock.** The MODFLOW/flopy user base is heavily
  conda-based. Do this after the PyPI release stabilizes, since conda-forge
  builds from the published sdist.

## Release sequence

1. Integrate the reviewed release branch into `main`.
2. Tag `v0.1.0rc1`; `release.yml` routes rc tags to TestPyPI only.
3. Let GitHub rebuild, retest, and validate one wheel and one sdist.
4. Review the build results, then manually approve the protected `testpypi`
   deployment.
5. Verify the rendered TestPyPI page and an independent installation.
6. Follow **Final PyPI release** above; the `v0.1.0` tag goes to PyPI only.

## Verification

- `twine check dist/*` passes; sdist/wheel contain only the package and
  standard metadata.
- README Basic Usage example runs against the installed wheel in a clean venv.
- TestPyPI project page renders with no broken links before the real upload.
- After the rehearsal: install `vorflow==0.1.0rc1` from TestPyPI in a fresh
  environment and confirm `vorflow.__version__` matches the tag.
- Real PyPI publication remains pending a separate review and approval.
