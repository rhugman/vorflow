# Contributing to vorflow

Contributions are welcome: bug reports, questions, documentation fixes, new
examples and code. By taking part you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Getting help

Ask questions about using `vorflow` in the
[issue tracker](https://github.com/rhugman/vorflow/issues). Check the
[documentation](https://rhugman.github.io/vorflow/) and the
[examples](https://github.com/rhugman/vorflow/tree/main/examples) first; if
they do not answer the question, that is worth an issue too.

## Reporting bugs

Open an [issue](https://github.com/rhugman/vorflow/issues) with:

- the `vorflow`, `gmsh`, `shapely` and Python versions, and the operating system;
- a minimal script that reproduces the problem, with the input geometry inline
  or attached (a small shapefile or GeoPackage is fine);
- what happened, including the full traceback, and what you expected.

Meshing problems are often geometry-specific. A plot of the conceptual mesh
(`blueprint.plot()`) or of the resulting grid helps a lot.

## Suggesting features

Open an issue describing the modelling problem the feature would solve. Check the
[open issues](https://github.com/rhugman/vorflow/issues) first; the feature
may already be planned.

## Development setup

The Conda environment matches CI and installs the package in editable mode
with the `dev` and `examples` extras, including FloPy for the model examples:

```bash
micromamba env create -f etc/environment.yml
```

To update an existing environment, run this from the checkout or worktree
you want to experiment with:

```bash
conda env update -n vorflow -f etc/environment.yml
```

The editable `..` path is relative to the environment file's `etc/`
directory. It follows that checkout, not other worktrees automatically.
Check the active source with `python -c "import vorflow; print(vorflow.__file__)"`
and restart any running notebook kernel after changing the installation.
The MODFLOW executables are installed separately as described below.

If Gmsh and the geospatial stack are already installed:

```bash
pip install -e ".[dev,examples]"
```

If another `vorflow` is installed in the environment, run with
`PYTHONPATH=src` so the checkout is imported.

## Running the tests and linter

```bash
pytest                 # full suite
pytest -m "not slow"   # skip the Gmsh-heavy end-to-end tests
ruff check src tests scripts
```

CI runs the full suite on Linux, Windows and macOS with Python 3.10 to 3.12,
a minimum-dependencies job, and `ruff`. Pull requests must pass all of them.

The documentation executes the example notebooks, so it needs Gmsh, `mf6`
and `mfusg`. After installing the Python dependencies, create a binary
directory and run `get-modflow /path/to/bin --subset mf6,mfusg`. Add that
directory to PATH; the MFUSG notebook also accepts `MFUSG_EXE` as a full path.
The notebooks do not download executables. Then build:

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
```

## Pull requests

1. Open an issue first for anything beyond a small fix, so the approach can be
   agreed before you write the code.
2. Branch from `main` and keep each pull request to one change.
3. Add or update tests. Mark Gmsh-heavy end-to-end tests with
   `@pytest.mark.slow`.
4. Add an entry under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md) for any
   user-visible change.
5. Update the docstrings, README or user guide if behaviour changes.

### Conventions

- Progress and diagnostics go through the `vorflow` logger, never `print()`.
- Raise `ValueError` for bad user input; use `assert` for internal invariants.
- The user-facing keyword is `resolution`; the internal column is `lc`.
- Commit example notebooks without outputs (`tests/test_notebooks.py` checks
  this).
- The index of the cleaned GeoDataFrames is the feature ID that links
  features to their Gmsh entities and refinement fields. Do not reindex or
  reorder them between `ConceptualMesh.generate()` and
  `MeshGenerator.generate()`.

## License

Contributions are released under the project's [MIT License](LICENSE).
