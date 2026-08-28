# TSG WLS-UQ numerical experiments

This directory is the PGM-owned source of the numerical experiments, compact
results, and generated numerical figures for the TSG WLS-UQ manuscript.  The
paper build assumes sibling checkouts with this layout:

```text
<workspace>/
├── power-grid-model/
└── TSG-WLS-UQ/
```

## Setup

Run the experiment commands from the `power-grid-model` repository root.  Its
Python environment must contain a current native PGM build and the example
dependencies:

```sh
UV_CACHE_DIR=/tmp/uv-cache uv sync --group example
```

If the native PGM build changes, recreate or update the environment before
rerunning the experiments.

## Rebuild the paper tables

The fastest reproducible route uses the retained machine-readable summaries:

```sh
.venv/bin/python docs/examples/tsg_wls_uq/build_paper_artifacts.py --tables-only
```

This cross-checks the retained native and standardized interval summaries,
then regenerates `results/one_sigma_interval_rows.tex`. It regenerates
`results/timing_power_law_rows.tex` from `results/timing_power_laws.csv`.
Both paths are relative to this directory. Synchronize the paper-facing
copies as shown below before building the manuscript.

## Reproduce all experiments

The full workflow reruns the notebook-derived accuracy, interval, and runtime
experiments, then rebuilds the manuscript artifacts:

```sh
.venv/bin/python docs/examples/tsg_wls_uq/run_accuracy_notebooks.py
.venv/bin/python docs/examples/tsg_wls_uq/run_interval_coverage.py
.venv/bin/python docs/examples/tsg_wls_uq/run_runtime_benchmark.py --repeats 3
MPLCONFIGDIR=/tmp/tsg-wls-uq-mplconfig \
    .venv/bin/python docs/examples/tsg_wls_uq/build_paper_artifacts.py
```

Synchronize the paper-facing copies of the generated figures and table rows:

```sh
mkdir -p ../TSG-WLS-UQ/tables
cp docs/examples/tsg_wls_uq/figures/cigre_lv_power_sigmas.pdf \
   docs/examples/tsg_wls_uq/figures/gain_inverse_accuracy_vs_zi.pdf \
   docs/examples/tsg_wls_uq/figures/ternary_tree_runtime.pdf \
   ../TSG-WLS-UQ/figs/
cp docs/examples/tsg_wls_uq/results/one_sigma_interval_rows.tex \
   docs/examples/tsg_wls_uq/results/timing_power_law_rows.tex \
   ../TSG-WLS-UQ/tables/
```

Then rebuild the paper:

```sh
cd ../TSG-WLS-UQ
latexmk -pdf main.tex
```

The paper runners override the smaller direct-run notebook defaults and use
1,000 Monte Carlo samples for the accuracy and interval workflows. The
runtime workflow includes ternary trees through one million buses and can be
expensive.  For a smoke test, use the network selectors accepted by the
accuracy and interval runners and `--max-buses`/`--mc-max-buses` with a small
value in the runtime runner; reduced runs do not reproduce the reported
numbers.

The notebook runners extract and execute the relevant cells from the PGM
examples without modifying the notebooks.  The source cases are:

* `CIGRE MV State Estimation UQ Example.ipynb`
* `CIGRE LV State Estimation UQ Example.ipynb`
* `IEEE33 State Estimation UQ Example.ipynb`
* `MV Oberrhein State Estimation UQ Example.ipynb`
* `LV Schutterwald State Estimation UQ Example.ipynb`
* `Synthetic MV Ternary Tree UQ Scaling Example.ipynb`

The shared helpers are `gain_matrix.py`, `gain_covariance.py`,
`network_summary.py`, `pandapower_mc_uq.py`, and `pandapower_mc_uq.md`.  The
external case files are `data/cigre_lv_pgm.json`,
`data/mv_oberrhein_pgm.json`, and `data/lv_schutterwald_pgm.json`, relative to
`docs/examples/` in the PGM checkout.

## Inputs and outputs

The scripts write temporary raw outputs under

```text
docs/examples/output/tsg_wls_uq_accuracy/
docs/examples/output/tsg_wls_uq_runtime/
```

The retained outputs are under this directory:

- `results/` contains the compact CSV summaries, LaTeX table rows, runtime
  data, and `experiment_metadata.json`.
- `figures/` contains the three generated numerical PDF figures.

The historical producing checkout and source hashes remain in
`results/experiment_metadata.json`.  Because relocation did not rerun the
experiments, `results/migration_manifest.json` separately records the current
PGM script hashes and retained-artifact hashes.  A complete rerun refreshes the
experiment metadata with current PGM paths and hashes; only then should the
migration manifest be replaced or removed.

After verifying the retained summaries and figures, the two raw manuscript
output directories above may be removed.  The full workflow recreates them.
Do not remove other directories under `docs/examples/output/`, since they may
belong to unrelated examples.
