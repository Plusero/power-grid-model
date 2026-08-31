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
then regenerates
`docs/examples/tsg_wls_uq/results/one_sigma_interval_rows.tex`. It regenerates
`docs/examples/tsg_wls_uq/results/timing_power_law_rows.tex` from
`docs/examples/tsg_wls_uq/results/timing_power_laws.csv`. Synchronize the
paper-facing copies as shown below before building the manuscript.

## Reproduce all experiments

The full workflow reruns the notebook-derived accuracy, interval, and runtime
experiments, then rebuilds the manuscript artifacts:

```sh
.venv/bin/python docs/examples/tsg_wls_uq/run_accuracy_notebooks.py
.venv/bin/python docs/examples/tsg_wls_uq/run_interval_coverage.py
.venv/bin/python docs/examples/tsg_wls_uq/run_runtime_benchmark.py --repeats 3
MPLCONFIGDIR=/tmp/tsg-wls-uq-mplconfig \
    .venv/bin/python docs/examples/tsg_wls_uq/build_paper_artifacts.py
MPLCONFIGDIR=/tmp/tsg-wls-uq-mplconfig \
    .venv/bin/python docs/examples/tsg_wls_uq/build_matrix_sparsity.py --case ieee33
```

The matrix builder is case-driven rather than IEEE33-specific. Its built-in
non-IEEE regression case can be generated without adding retained paper
artifacts to the repository:

```sh
MPLCONFIGDIR=/tmp/tsg-wls-uq-mplconfig \
    .venv/bin/python docs/examples/tsg_wls_uq/build_matrix_sparsity.py \
    --case cigre-mv --output-root /tmp/cigre-mv-sparsity
```

`SparsityCase` and `build_sparsity_artifacts()` are the programmatic interface
for supplying another row-oriented topology, PGM state-estimation input
dataset, solver tolerance and iteration settings, artifact prefix, and optional
regression expectations. All matrix dimensions, row-group counts, block
boundaries, angle references, and artifact names are derived from the complete
case; the real-gain angle references are selected only after all finite
measurement rows, including bus injections, have been assembled. PGM's fixed
1-MVA three-phase normalization is used throughout. The current complex
reconstruction supports one connected radial symmetric network containing
ordinary lines and two-winding transformers. It accepts voltage, global-angle
branch-current, branch-power, and PGM-style aggregated
load/generator/source/direct-node power measurements. Static shunts and other
unsupported components or terminals fail explicitly. The legacy
`build_ieee33_sparsity.py` remains a compatibility wrapper for the default
case.

Synchronize the paper-facing copies of the generated figures and table rows:

```sh
mkdir -p ../TSG-WLS-UQ/tables
cp docs/examples/tsg_wls_uq/figures/cigre_lv_power_sigmas.pdf \
   docs/examples/tsg_wls_uq/figures/gain_inverse_accuracy_vs_zi.pdf \
   docs/examples/tsg_wls_uq/figures/ieee33_wls_uq_matrix_sparsity.pdf \
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

## Input locations

All paths in this section are relative to the `power-grid-model` repository
root. The case notebooks under `docs/examples/` define the sensor layouts,
uncertainties, Monte Carlo settings, and output quantities. Their PGM network
models are stored or constructed as follows:

| Case | PGM network-model source | Sensor and experiment definition |
|---|---|---|
| CIGRE MV | The `workshop_model` dictionary is embedded in `docs/examples/CIGRE MV State Estimation UQ Example.ipynb`; it reproduces the CIGRE MV model from the state-estimation workshop. | The same notebook defines the three sensor layouts and UQ calculations. |
| CIGRE LV | `docs/examples/data/cigre_lv_pgm.json`, generated from pandapower's CIGRE LV network with the PGM-IO converter. | `docs/examples/CIGRE LV State Estimation UQ Example.ipynb` loads the JSON and constructs the sensors. |
| IEEE 33 | The `workshop_model` dictionary is embedded in `docs/examples/IEEE33 State Estimation UQ Example.ipynb`; it is sourced from the `dsse-uq-non-gaussian` IEEE33 PGM model. | The same notebook defines the three sensor layouts and UQ calculations. |
| MV Oberrhein | `docs/examples/data/mv_oberrhein_pgm.json`, converted from pandapower's default high-load MV Oberrhein network. | `docs/examples/MV Oberrhein State Estimation UQ Example.ipynb` loads the JSON and constructs the sensors. |
| LV Schutterwald | `docs/examples/data/lv_schutterwald_pgm.json`, converted from pandapower's LV Schutterwald network. | `docs/examples/LV Schutterwald State Estimation UQ Example.ipynb` loads the JSON and constructs the sensors. |
| Synthetic MV ternary tree | Generated in memory by `build_ternary_mv_network()` in `docs/examples/Synthetic MV Ternary Tree UQ Scaling Example.ipynb`; no network JSON is read. | The same notebook generates the loads and sensors used by the runtime benchmark. |

The runners extract and execute the relevant notebook cells without modifying
the notebooks. Shared calculation and comparison helpers are
`docs/examples/gain_matrix.py`, `docs/examples/gain_covariance.py`,
`docs/examples/network_summary.py`, `docs/examples/pandapower_mc_uq.py`, and
`docs/examples/pandapower_mc_uq.md`.

## Output locations

The workflow has three output levels. The authoritative paper-oriented outputs
are the retained `results/` and `figures/` directories; the raw run directories
are reproducible working data, and the sibling manuscript contains synchronized
copies.

| Output level | Location | Contents and role |
|---|---|---|
| Temporary accuracy and interval output | `docs/examples/output/tsg_wls_uq_accuracy/<case>/` | Per-case comparison and baseline-metric CSV files, one-sigma interval scores, figures, and execution logs. The case directories are `cigre_mv_radial`, `cigre_lv`, `ieee33`, `mv_oberrhein`, and `lv_schutterwald`. |
| Temporary runtime output | `docs/examples/output/tsg_wls_uq_runtime/` | `ternary_tree_runtime.csv` and `runtime_metadata.json` produced by the runtime runner. |
| Retained paper-oriented results | `docs/examples/tsg_wls_uq/results/` | Compact CSV summaries, runtime and fitted-scaling data, the two LaTeX table fragments, provenance metadata, and the IEEE33 matrix NPZ/CSV artifacts. These are the authoritative numerical results used to rebuild the paper artifacts. |
| Retained generated figures | `docs/examples/tsg_wls_uq/figures/` | The four paper PDF figures and the IEEE33 PNG preview. These are the authoritative generated figure files. |
| Manuscript-facing copies | `../TSG-WLS-UQ/tables/` and `../TSG-WLS-UQ/figs/` | Copies consumed by `../TSG-WLS-UQ/main.tex`. They must be synchronized after regeneration and are not a second authoritative results source. |

Within `docs/examples/tsg_wls_uq/`, for example,
`results/accuracy_median_ape.csv`,
`results/one_sigma_interval_scores_standardized.csv`, and
`results/timing_power_laws.csv` contain paper-oriented numerical summaries.
The paper table rows are `results/one_sigma_interval_rows.tex` and
`results/timing_power_law_rows.tex`. The default IEEE33 sparsity case
additionally writes `results/ieee33_wls_uq_matrices.npz`,
`results/ieee33_wls_uq_matrix_sparsity.csv`, and
`results/ieee33_pgm_bus_order.csv`, plus the corresponding PDF and PNG under
`figures/`. Its real-polar `H` contains every active finite-variance voltage,
current, branch-power, and aggregated load-injection measurement for the
`[theta; voltage magnitude]` state. The workflow verifies `G = H.T @ W @ H`
and compares `diag(G^-1)` with PGM's analytical NRSE output. It separately
checks the complex augmented inverse against PGM's analytical ILSE output.
For other cases, the same filenames use the selected case prefix; exact
zero-injection constraints are represented in the complex augmented system but
are intentionally excluded from the conventional finite-weight real gain. The
CIGRE MV regression therefore validates that real gain algebraically and by
finite differences, while its end-to-end covariance comparison with PGM uses
the complex augmented formulation.

The historical producing checkout and source hashes remain in
`results/experiment_metadata.json`.  Because relocation did not rerun the
experiments, `results/migration_manifest.json` separately records the current
PGM script hashes and retained-artifact hashes.  A complete rerun refreshes the
experiment metadata with current PGM paths and hashes; only then should the
migration manifest be replaced or removed.

After verifying the retained summaries and figures, the two temporary raw-output
roots above may be removed. The full workflow recreates them.
Do not remove other directories under `docs/examples/output/`, since they may
belong to unrelated examples.
