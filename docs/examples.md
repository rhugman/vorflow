# Examples

The notebooks below are run on every documentation build. Their sources are in
[`examples/`](https://github.com/rhugman/vorflow/tree/main/examples), together
with standalone scripts for field-based refinement, mesh quality diagnostics,
structured quad buffers, active-domain workflows and triangular element-grid
export.

```{toctree}
:maxdepth: 1

examples/basic_example
examples/comprehensive_demo
examples/point_centring_demo
examples/plotting_demo
examples/cleaning_limitations_demo
examples/basic_example_modflow
```

## Barrier layout and optional repairs

The [MF6 notebook](examples/basic_example_modflow.ipynb) uses the existing
`add_line()` API to show straddle sites and the one-row quad-buffer alternative,
with optional mirrors, clipping and fragment merging. It checks cell/face
quality, tracks both sides of the fault, plots the selected generator
connections and uses those pairs to construct the FloPy HFB package.

Hydraulic resistance is assigned separately from mesh generation. Run
`python scripts/validate_barrier_flow.py --engines mf6` for analytical head and
flow checks, including a resistive HFB case.
