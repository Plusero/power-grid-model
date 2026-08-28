# ruff: noqa: PLR0912, PLR2004, S603, S607
"""Build manuscript tables and figures from the fresh numerical experiments."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

TopologyValue = int | float | str
NetworkTopology = dict[str, list[dict[str, TopologyValue]]]

WORKFLOW_ROOT = Path(__file__).resolve().parent
PGM_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = PGM_ROOT / "docs" / "examples"
RAW_ACCURACY_ROOT = EXAMPLE_ROOT / "output" / "tsg_wls_uq_accuracy"
RAW_RUNTIME_ROOT = EXAMPLE_ROOT / "output" / "tsg_wls_uq_runtime"
RESULTS_ROOT = WORKFLOW_ROOT / "results"
FIGURE_ROOT = WORKFLOW_ROOT / "figures"

SOURCE_FILES = (
    EXAMPLE_ROOT / "CIGRE MV State Estimation UQ Example.ipynb",
    EXAMPLE_ROOT / "CIGRE LV State Estimation UQ Example.ipynb",
    EXAMPLE_ROOT / "IEEE33 State Estimation UQ Example.ipynb",
    EXAMPLE_ROOT / "MV Oberrhein State Estimation UQ Example.ipynb",
    EXAMPLE_ROOT / "LV Schutterwald State Estimation UQ Example.ipynb",
    EXAMPLE_ROOT / "Synthetic MV Ternary Tree UQ Scaling Example.ipynb",
    EXAMPLE_ROOT / "gain_matrix.py",
    EXAMPLE_ROOT / "gain_covariance.py",
    EXAMPLE_ROOT / "network_summary.py",
    EXAMPLE_ROOT / "pandapower_mc_uq.py",
    EXAMPLE_ROOT / "pandapower_mc_uq.md",
    EXAMPLE_ROOT / "data" / "cigre_lv_pgm.json",
    EXAMPLE_ROOT / "data" / "mv_oberrhein_pgm.json",
    EXAMPLE_ROOT / "data" / "lv_schutterwald_pgm.json",
    WORKFLOW_ROOT / "run_accuracy_notebooks.py",
    WORKFLOW_ROOT / "run_interval_coverage.py",
    WORKFLOW_ROOT / "run_runtime_benchmark.py",
    WORKFLOW_ROOT / "build_paper_artifacts.py",
)

NETWORKS = {
    "CIGRE MV": (
        "cigre_mv_radial",
        "cigre_mv_uq_comparison.csv",
        "voltage + current + power",
    ),
    "CIGRE LV": ("cigre_lv", "cigre_lv_uq_comparison.csv", "voltage + current + power"),
    "IEEE 33": ("ieee33", "ieee33_uq_comparison.csv", "voltage + current + power"),
    "MV Oberrhein": (
        "mv_oberrhein",
        "mv_oberrhein_uq_comparison.csv",
        "voltage + current + power",
    ),
    "LV Schutterwald": ("lv_schutterwald", "lv_schutterwald_uq_comparison.csv", None),
}

NETWORK_STATISTICS = {
    "CIGRE MV": {"buses": 15, "zero-injection buses": 1},
    "CIGRE LV": {"buses": 41, "zero-injection buses": 25},
    "IEEE 33": {"buses": 33, "zero-injection buses": 0},
    "MV Oberrhein": {"buses": 179, "zero-injection buses": 18},
    "LV Schutterwald": {"buses": 2_940, "zero-injection buses": 1_420},
}

TOPOLOGY_SOURCES = {
    "CIGRE MV": ("notebook", "CIGRE MV State Estimation UQ Example.ipynb"),
    "CIGRE LV": ("json", "data/cigre_lv_pgm.json"),
    "IEEE 33": ("notebook", "IEEE33 State Estimation UQ Example.ipynb"),
    "MV Oberrhein": ("json", "data/mv_oberrhein_pgm.json"),
    "LV Schutterwald": ("json", "data/lv_schutterwald_pgm.json"),
}

METHODS = {
    "Analytical UQ ILSE sigma": "Analytical ILSE",
    "Analytical UQ NRSE sigma": "Analytical NRSE",
    "Monte Carlo UQ ILSE sigma": "MC ILSE",
    "Pandapower MC WLS sigma": "Pandapower MC WLS",
    "UQ gain inv sigma": "Gain inverse NumPy",
}
BASELINE_COLUMN = "Monte Carlo UQ NRSE sigma"

ABSOLUTE_SIGMA_METHODS = {
    "Analytical UQ ILSE sigma": "Analytical UQ ILSE",
    "Analytical UQ NRSE sigma": "Analytical UQ NRSE",
    "Monte Carlo UQ ILSE sigma": "Monte Carlo UQ ILSE",
    BASELINE_COLUMN: "Monte Carlo UQ NRSE",
    "Pandapower MC WLS sigma": "Pandapower MC WLS",
    "UQ gain inv sigma": "NumPy gain inverse",
}

INTERVAL_METHODS = (
    "Analytical UQ ILSE",
    "Monte Carlo UQ ILSE",
    "Analytical UQ NRSE",
    "Monte Carlo UQ NRSE",
    "Gain inverse NumPy",
)

TIMING_METHODS = (
    ("Deterministic ILSE", "Deterministic ILSE"),
    ("Deterministic NRSE", "Deterministic NRSE"),
    ("Analytical UQ ILSE", "Analytical UQ ILSE"),
    ("Analytical UQ NRSE", "Analytical UQ NRSE"),
    ("Monte Carlo ILSE", r"MC UQ ILSE ($N=10$)"),
    ("Monte Carlo NRSE", r"MC UQ NRSE ($N=10$)"),
    ("Gain inverse NumPy", "Gain inv. NumPy"),
    ("Gain inverse SciPy", "Gain inv. SciPy"),
)

ONE_SIGMA_COVERAGE = math.erf(1.0 / math.sqrt(2.0))
ONE_SIGMA_ALPHA = 1.0 - ONE_SIGMA_COVERAGE

QUANTITIES = {
    "Voltage magnitude": lambda values: values.eq("voltage magnitude"),
    "Current magnitude": lambda values: values.eq("current"),
    "Active power": lambda values: values.str.startswith("active power"),
    "Reactive power": lambda values: values.str.startswith("reactive power"),
}

ABSOLUTE_SIGMA_QUANTITIES = {
    "Voltage magnitude": (r"sigma_|U|", "V", 1.0),
    "Current magnitude": (r"sigma_|I|", "A", 1.0),
    "Active power": (r"sigma_P", "kW", 1.0e-3),
    "Reactive power": (r"sigma_Q", "kvar", 1.0e-3),
}


def load_accuracy_comparison(network: str) -> pd.DataFrame:
    directory, filename, sensor_setting = NETWORKS[network]
    comparison = pd.read_csv(RAW_ACCURACY_ROOT / directory / filename)
    if sensor_setting is not None:
        comparison = comparison[comparison["sensor setting"] == sensor_setting]
    if network == "LV Schutterwald":
        node_counts = comparison.loc[comparison["component"] == "node"].groupby("quantity", sort=False).size()
        expected_node_count = NETWORK_STATISTICS[network]["buses"]
        for quantity in (
            "voltage magnitude",
            "active power injection",
            "reactive power injection",
        ):
            if int(node_counts.get(quantity, 0)) != expected_node_count:
                raise ValueError(f"LV Schutterwald comparison must contain {expected_node_count} node {quantity} rows")
    return comparison.copy()


def common_quantity_subsets(
    comparison: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    method_columns = [*METHODS, BASELINE_COLUMN]
    missing_columns = set(method_columns).difference(comparison.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Accuracy comparison is missing sigma columns: {missing}")

    subsets: dict[str, pd.DataFrame] = {}
    for quantity, selector in QUANTITIES.items():
        quantity_rows = comparison[selector(comparison["quantity"])].copy()
        common_finite = np.isfinite(quantity_rows[method_columns]).all(axis=1)
        nonzero_reference = quantity_rows[BASELINE_COLUMN].abs() > np.finfo(float).eps
        subsets[quantity] = quantity_rows[common_finite & nonzero_reference]
    return subsets


def load_network_topology(network: str) -> NetworkTopology:
    source_type, relative_path = TOPOLOGY_SOURCES[network]
    source_path = EXAMPLE_ROOT / relative_path
    if source_type == "json":
        return json.loads(source_path.read_text(encoding="utf-8"))

    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    matches: list[NetworkTopology] = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        try:
            syntax_tree = ast.parse("".join(cell.get("source", [])))
        except SyntaxError:
            continue
        for statement in syntax_tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            is_workshop_model = any(
                isinstance(target, ast.Name) and target.id == "workshop_model" for target in statement.targets
            )
            if is_workshop_model:
                matches.append(ast.literal_eval(statement.value))
    if len(matches) != 1:
        raise ValueError(f"Expected one literal workshop_model in {source_path}, found {len(matches)}")
    return matches[0]


def attach_rated_voltage(
    comparison: pd.DataFrame,
    topology: NetworkTopology,
) -> pd.DataFrame:
    nodes = {int(row["id"]): float(row["u_rated"]) for row in topology["node"]}
    branches = {
        (component, int(row["id"])): row for component in ("line", "transformer") for row in topology.get(component, [])
    }

    component_classes: list[str] = []
    rated_voltages: list[float] = []
    for row in comparison.itertuples(index=False):
        component_id = int(row.id)
        if row.component == "node":
            if row.terminal != "—":
                raise ValueError(f"Unexpected node terminal {row.terminal!r}")
            terminal_node = component_id
            component_class = "Node"
        else:
            if row.component not in {"line", "transformer"}:
                raise ValueError(f"Unexpected output component {row.component!r}")
            if row.terminal not in {"from", "to"}:
                raise ValueError(f"Unexpected branch terminal {row.terminal!r}")
            branch = branches[(row.component, component_id)]
            terminal_node = int(branch[f"{row.terminal}_node"])
            component_class = "Branch terminal"
        component_classes.append(component_class)
        rated_voltages.append(nodes[terminal_node] / 1.0e3)

    result = comparison.copy()
    result["component class"] = component_classes
    result["rated voltage [kV]"] = rated_voltages
    if result[["component class", "rated voltage [kV]"]].isna().any().any():
        raise ValueError("Rated-voltage join contains missing values")
    return result


def accuracy_table() -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    rows: list[dict[str, object]] = []
    sample_counts: dict[str, dict[str, int]] = {}

    for network, (directory, filename, sensor_setting) in NETWORKS.items():
        comparison = load_accuracy_comparison(network)

        baseline_metrics_path = (
            RAW_ACCURACY_ROOT / directory / filename.replace("_comparison.csv", "_baseline_metrics.csv")
        )
        baseline_metrics = pd.read_csv(baseline_metrics_path)
        if sensor_setting is not None:
            baseline_metrics = baseline_metrics[baseline_metrics["sensor setting"] == sensor_setting]
        sample_counts[network] = {
            "attempted": int(baseline_metrics["baseline attempted samples"].iloc[0]),
            "converged": int(baseline_metrics["baseline converged samples"].iloc[0]),
        }

        quantity_subsets = common_quantity_subsets(comparison)

        for method_column, method in METHODS.items():
            statistics = NETWORK_STATISTICS[network]
            row: dict[str, object] = {
                "network": network,
                "buses": statistics["buses"],
                "zero-injection buses": statistics["zero-injection buses"],
                "zero-injection buses [%]": 100.0 * statistics["zero-injection buses"] / statistics["buses"],
                "method": method,
            }
            for quantity, subset in quantity_subsets.items():
                if method_column not in subset:
                    row[f"{quantity} median APE [%]"] = np.nan
                    row[f"{quantity} finite pairs"] = 0
                    continue
                errors = 100.0 * (subset[method_column] - subset[BASELINE_COLUMN]).abs() / subset[BASELINE_COLUMN].abs()
                row[f"{quantity} median APE [%]"] = float(errors.median()) if len(errors) else np.nan
                row[f"{quantity} finite pairs"] = len(errors)
            rows.append(row)

    return pd.DataFrame(rows), sample_counts


def absolute_sigma_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expected_raw_units = {
        "Voltage magnitude": "V",
        "Current magnitude": "A",
        "Active power": "W",
        "Reactive power": "var",
    }

    for network in NETWORKS:
        comparison = attach_rated_voltage(load_accuracy_comparison(network), load_network_topology(network))
        for quantity, subset in common_quantity_subsets(comparison).items():
            if subset.empty:
                continue
            raw_units = set(subset["unit"])
            if raw_units != {expected_raw_units[quantity]}:
                raise ValueError(f"Unexpected units for {network} {quantity}: {raw_units}")

            symbol, output_unit, scale = ABSOLUTE_SIGMA_QUANTITIES[quantity]
            group_columns = ["component class", "rated voltage [kV]"]
            for (component_class, rated_voltage), group in subset.groupby(group_columns, sort=False):
                row: dict[str, object] = {
                    "network": network,
                    "component class": component_class,
                    "rated voltage [kV]": float(rated_voltage),
                    "quantity": quantity,
                    "symbol": symbol,
                    "unit": output_unit,
                    "finite outputs": len(group),
                }
                for method_column, method in ABSOLUTE_SIGMA_METHODS.items():
                    row[f"{method} median sigma"] = float(group[method_column].median()) * scale
                rows.append(row)

    result = pd.DataFrame(rows)
    network_order = {network: index for index, network in enumerate(NETWORKS)}
    component_order = {"Node": 0, "Branch terminal": 1}
    quantity_order = {quantity: index for index, quantity in enumerate(QUANTITIES)}
    result["network order"] = result["network"].map(network_order)
    result["component order"] = result["component class"].map(component_order)
    result["quantity order"] = result["quantity"].map(quantity_order)
    result = result.sort_values(
        [
            "network order",
            "component order",
            "rated voltage [kV]",
            "quantity order",
        ],
        ascending=[True, True, False, True],
    ).drop(columns=["network order", "component order", "quantity order"])
    return result.reset_index(drop=True)


def validate_absolute_sigma_table(absolute_sigmas: pd.DataFrame, accuracy: pd.DataFrame) -> None:
    key_columns = ["network", "component class", "rated voltage [kV]", "quantity"]
    if absolute_sigmas.duplicated(key_columns).any():
        raise ValueError("Absolute-sigma groups are not unique")

    median_columns = [f"{method} median sigma" for method in ABSOLUTE_SIGMA_METHODS.values()]
    medians = absolute_sigmas[median_columns].to_numpy(dtype=float)
    if not np.isfinite(medians).all() or (medians < 0.0).any():
        raise ValueError("Absolute-sigma medians must be finite and nonnegative")

    expected_counts = accuracy.groupby("network", sort=False).first()
    actual_counts = absolute_sigmas.groupby(["network", "quantity"], sort=False)["finite outputs"].sum()
    for network in NETWORKS:
        for quantity in QUANTITIES:
            expected = int(expected_counts.loc[network, f"{quantity} finite pairs"])
            actual = int(actual_counts.get((network, quantity), 0))
            if actual != expected:
                raise ValueError(
                    f"Count mismatch for {network} {quantity}: absolute table {actual}, accuracy table {expected}"
                )


def load_interval_scores(network: str) -> pd.DataFrame:
    directory, _, sensor_setting = NETWORKS[network]
    scores = pd.read_csv(RAW_ACCURACY_ROOT / directory / f"{directory}_one_sigma_interval_scores.csv")
    if sensor_setting is not None:
        scores = scores[scores["sensor setting"] == sensor_setting]
    return scores.copy()


def eligible_interval_scores(network: str) -> pd.DataFrame:
    """Return the common five-method interval-score output population."""
    key_columns = ["component", "id", "terminal", "quantity"]
    comparison = attach_rated_voltage(load_accuracy_comparison(network), load_network_topology(network))
    if comparison.duplicated(key_columns).any():
        raise ValueError(f"Duplicate comparison outputs for {network}")

    interval_scores = load_interval_scores(network)
    expected_methods = set(INTERVAL_METHODS)
    if set(interval_scores["method"]) != expected_methods:
        raise ValueError(f"Unexpected interval methods for {network}")
    if interval_scores.duplicated(["method", *key_columns]).any():
        raise ValueError(f"Duplicate interval-score outputs for {network}")

    expected_sources = {
        "Analytical UQ ILSE": ("ILSE", "scenario-specific analytical", 0),
        "Monte Carlo UQ ILSE": ("ILSE", "independent Monte Carlo", 1_000),
        "Analytical UQ NRSE": ("NRSE", "scenario-specific analytical", 0),
        "Monte Carlo UQ NRSE": ("NRSE", "independent Monte Carlo", 1_000),
        "Gain inverse NumPy": ("NRSE", "clean reference-point gain inverse", 1),
    }
    for method, (center_source, sigma_source, sigma_samples) in expected_sources.items():
        method_rows = interval_scores[interval_scores["method"] == method]
        if set(method_rows["center source"]) != {center_source}:
            raise ValueError(f"Unexpected center source for {network} {method}")
        if set(method_rows["sigma source"]) != {sigma_source}:
            raise ValueError(f"Unexpected sigma source for {network} {method}")
        if not (method_rows["sigma calibration samples"] == sigma_samples).all():
            raise ValueError(f"Unexpected sigma calibration count for {network} {method}")

    np.testing.assert_allclose(
        interval_scores["MWS"],
        2.0 * interval_scores["mean interval sigma"],
        rtol=1.0e-12,
        atol=1.0e-9,
    )
    if (interval_scores["MWIS"] + 1.0e-12 < interval_scores["MWS"]).any():
        raise ValueError(f"MWIS is smaller than MWS for {network}")
    mc_rows = interval_scores[interval_scores["method"].str.startswith("Monte Carlo")]
    np.testing.assert_allclose(
        mc_rows["mean interval sigma"],
        mc_rows["empirical MC sigma"],
        rtol=1.0e-12,
        atol=1.0e-9,
    )

    for method, center_method in (
        ("Monte Carlo UQ ILSE", "Analytical UQ ILSE"),
        ("Monte Carlo UQ NRSE", "Analytical UQ NRSE"),
        ("Gain inverse NumPy", "Analytical UQ NRSE"),
    ):
        left = interval_scores.loc[interval_scores["method"] == method, [*key_columns, "mean estimate"]]
        right = interval_scores.loc[
            interval_scores["method"] == center_method,
            [*key_columns, "mean estimate"],
        ]
        paired = left.merge(
            right,
            on=key_columns,
            how="inner",
            suffixes=("", " center"),
            validate="one_to_one",
        )
        if paired.empty:
            raise ValueError(f"No shared centers for {network} {method}")
        np.testing.assert_allclose(
            paired["mean estimate"],
            paired["mean estimate center"],
            rtol=0.0,
            atol=1.0e-12,
        )

    metadata_columns = [
        *key_columns,
        "true value",
        "component class",
        "rated voltage [kV]",
    ]
    merged = interval_scores.merge(
        comparison[metadata_columns],
        on=key_columns,
        how="inner",
        suffixes=("", " comparison"),
        validate="many_to_one",
    )
    np.testing.assert_allclose(
        merged["true value"],
        merged["true value comparison"],
        rtol=1.0e-12,
        atol=1.0e-9,
    )
    native_score_columns = ["MCS", "MWS", "MWIS", "empirical MC sigma"]
    native_finite = np.isfinite(merged[native_score_columns]).all(axis=1)
    nonzero = merged["empirical MC sigma"].abs() > np.finfo(float).eps
    deterministic = native_finite & ~nonzero
    if deterministic.any() and not np.allclose(merged.loc[deterministic, "MCS"], 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"A zero-variance output does not have full coverage for {network}")
    standardized_score_columns = ["standardized MWS", "standardized MWIS"]
    finite = native_finite & np.isfinite(merged[standardized_score_columns]).all(axis=1)
    merged = merged[finite & nonzero].copy()
    method_counts = merged.groupby(key_columns, sort=False)["method"].nunique()
    common_keys = method_counts[method_counts == len(INTERVAL_METHODS)].index
    merged = merged.set_index(key_columns).loc[common_keys].reset_index()
    expected_rows = len(common_keys) * len(INTERVAL_METHODS)
    if len(merged) != expected_rows:
        raise ValueError(f"Incomplete common interval-score population for {network}")
    merged["network"] = network
    return merged.drop(columns="true value comparison")


def _aggregate_interval_group(group: pd.DataFrame, *, network: str, quantity: str, method: str) -> dict[str, object]:
    symbol, output_unit, scale = ABSOLUTE_SIGMA_QUANTITIES[quantity]
    attempted = group["attempted samples"].unique()
    method_converged = group["method converged samples"].unique()
    common_converged = group["common converged samples"].unique()
    sigma_calibration = group["sigma calibration samples"].unique()
    if len(attempted) != 1 or len(method_converged) != 1 or len(common_converged) != 1 or len(sigma_calibration) != 1:
        raise ValueError(f"Inconsistent scenario counts for {network} {quantity} {method}")
    mean_width = float(group["MWS"].mean())
    mean_winkler = float(group["MWIS"].mean())
    mean_empirical_sigma = float(group["empirical MC sigma"].mean())
    if not np.isfinite(mean_empirical_sigma) or mean_empirical_sigma <= 0.0:
        raise ValueError(f"Invalid empirical scale for {network} {quantity} {method}")
    return {
        "network": network,
        "quantity": quantity,
        "symbol": symbol,
        "unit": output_unit,
        "method": method,
        "finite outputs": len(group),
        "attempted samples": int(attempted[0]),
        "method converged samples": int(method_converged[0]),
        "common converged samples": int(common_converged[0]),
        "sigma calibration samples": int(sigma_calibration[0]),
        "MCS [%]": 100.0 * float(group["MCS"].mean()),
        "MWS": scale * mean_width,
        "MWIS": scale * mean_winkler,
        "standardized MWS": mean_width / mean_empirical_sigma,
        "standardized MWIS": mean_winkler / mean_empirical_sigma,
    }


def interval_score_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return native-unit/rated-voltage and standardized interval tables."""
    native_rows: list[dict[str, object]] = []
    standardized_rows: list[dict[str, object]] = []
    expected_raw_units = {
        "Voltage magnitude": "V",
        "Current magnitude": "A",
        "Active power": "W",
        "Reactive power": "var",
    }

    for network in NETWORKS:
        scores = eligible_interval_scores(network)
        for quantity, selector in QUANTITIES.items():
            quantity_scores = scores[selector(scores["quantity"])].copy()
            if quantity_scores.empty:
                continue
            if set(quantity_scores["unit"]) != {expected_raw_units[quantity]}:
                raise ValueError(f"Unexpected interval units for {network} {quantity}")

            for (component_class, rated_voltage, method), group in quantity_scores.groupby(
                ["component class", "rated voltage [kV]", "method"], sort=False
            ):
                row = _aggregate_interval_group(group, network=network, quantity=quantity, method=method)
                row["component class"] = component_class
                row["rated voltage [kV]"] = float(rated_voltage)
                native_rows.append(row)

            for (component_class, method), group in quantity_scores.groupby(["component class", "method"], sort=False):
                row = _aggregate_interval_group(group, network=network, quantity=quantity, method=method)
                row["component class"] = component_class
                standardized_rows.append(row)

    native = pd.DataFrame(native_rows)
    standardized = pd.DataFrame(standardized_rows).drop(columns=["unit", "MWS", "MWIS"])
    network_order = {network: index for index, network in enumerate(NETWORKS)}
    component_order = {"Node": 0, "Branch terminal": 1}
    quantity_order = {quantity: index for index, quantity in enumerate(QUANTITIES)}
    method_order = {method: index for index, method in enumerate(INTERVAL_METHODS)}
    for table in (native, standardized):
        table["network order"] = table["network"].map(network_order)
        table["component order"] = table["component class"].map(component_order)
        table["quantity order"] = table["quantity"].map(quantity_order)
        table["method order"] = table["method"].map(method_order)
    native = native.sort_values(
        [
            "network order",
            "component order",
            "rated voltage [kV]",
            "quantity order",
            "method order",
        ],
        ascending=[True, True, False, True, True],
    )
    standardized = standardized.sort_values(["network order", "component order", "quantity order", "method order"])
    order_columns = [
        "network order",
        "component order",
        "quantity order",
        "method order",
    ]
    native = native.drop(columns=order_columns).reset_index(drop=True)
    standardized = standardized.drop(columns=order_columns).reset_index(drop=True)
    return native, standardized


def validate_interval_score_tables(
    native_scores: pd.DataFrame,
    standardized_scores: pd.DataFrame,
) -> None:
    native_keys = [
        "network",
        "component class",
        "rated voltage [kV]",
        "quantity",
        "method",
    ]
    standardized_keys = ["network", "component class", "quantity", "method"]
    if native_scores.duplicated(native_keys).any():
        raise ValueError("Native-unit interval-score groups are not unique")
    if standardized_scores.duplicated(standardized_keys).any():
        raise ValueError("Standardized interval-score groups are not unique")

    for table, group_columns in (
        (native_scores, native_keys[:-1]),
        (standardized_scores, standardized_keys[:-1]),
    ):
        method_sets = table.groupby(group_columns, sort=False)["method"].agg(set)
        if not method_sets.map(lambda methods: methods == set(INTERVAL_METHODS)).all():
            raise ValueError("An interval-score group does not contain all methods")
        output_counts = table.groupby(group_columns, sort=False)["finite outputs"].nunique()
        if not (output_counts == 1).all():
            raise ValueError("Interval methods use different output counts")

    actual_counts = native_scores.groupby(["network", "quantity", "method"], sort=False)["finite outputs"].sum()
    expected_counts = standardized_scores.groupby(["network", "quantity", "method"], sort=False)["finite outputs"].sum()
    if not actual_counts.sort_index().equals(expected_counts.sort_index()):
        raise ValueError("Native and standardized interval-score counts disagree")

    native_numeric = native_scores[["MCS [%]", "MWS", "MWIS", "standardized MWS", "standardized MWIS"]].to_numpy(
        dtype=float
    )
    standardized_numeric = standardized_scores[["MCS [%]", "standardized MWS", "standardized MWIS"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(native_numeric).all() or not np.isfinite(standardized_numeric).all():
        raise ValueError("Interval scores must be finite")

    for table in (native_scores, standardized_scores):
        coverage = table["MCS [%]"]
        if (coverage < 0.0).any() or (coverage > 100.0).any():
            raise ValueError("Coverage is outside [0, 100]")
        if (table["standardized MWS"] < 0.0).any() or (
            table["standardized MWIS"] + 1.0e-12 < table["standardized MWS"]
        ).any():
            raise ValueError("Invalid standardized interval score")
        if not (table["attempted samples"] == 1_000).all():
            raise ValueError("Interval experiment did not use 1,000 samples")
        if (table["common converged samples"] > table["method converged samples"]).any():
            raise ValueError("Common convergence count exceeds a method count")
        if (table["common converged samples"] < 0.9 * table["attempted samples"]).any():
            raise ValueError("Fewer than 90% of interval scenarios converged")

        expected_sigma_samples = table["method"].map(
            {
                "Analytical UQ ILSE": 0,
                "Monte Carlo UQ ILSE": 1_000,
                "Analytical UQ NRSE": 0,
                "Monte Carlo UQ NRSE": 1_000,
                "Gain inverse NumPy": 1,
            }
        )
        if not table["sigma calibration samples"].equals(expected_sigma_samples):
            raise ValueError("Unexpected interval sigma calibration sample count")

    mc_standardized_widths = standardized_scores.loc[
        standardized_scores["method"].str.startswith("Monte Carlo"),
        "standardized MWS",
    ]
    if not np.allclose(mc_standardized_widths, 2.0, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("Monte Carlo one-sigma intervals do not have width 2 sigma")

    if (native_scores["MWS"] < 0.0).any() or (native_scores["MWIS"] + 1.0e-12 < native_scores["MWS"]).any():
        raise ValueError("Invalid native interval width or Winkler score")


def write_interval_table_rows(standardized_scores: pd.DataFrame) -> None:
    """Write pooled MCS rows for the merged node/branch manuscript table."""
    quantity_labels = {
        "Voltage magnitude": r"$|U|$",
        "Current magnitude": r"$|I|$",
        "Active power": r"$P$",
        "Reactive power": r"$Q$",
    }
    quantity_groups = {
        "Node": ("Voltage magnitude", "Active power", "Reactive power"),
        "Branch terminal": (
            "Current magnitude",
            "Active power",
            "Reactive power",
        ),
    }

    def validate_group(network: str, component: str) -> pd.DataFrame:
        rows = standardized_scores[
            (standardized_scores["network"] == network) & (standardized_scores["component class"] == component)
        ]
        expected_quantities = set(quantity_groups[component])
        actual_quantities = set(rows["quantity"])
        if actual_quantities != expected_quantities:
            raise ValueError(
                f"Incomplete pooled {component} interval quantities for {network}: "
                f"expected {sorted(expected_quantities)}, "
                f"found {sorted(actual_quantities)}"
            )
        return rows

    def quantity_cells(network: str, component: str, quantity: str, rows: pd.DataFrame) -> list[str]:
        quantity_rows = rows[rows["quantity"] == quantity]
        counts = quantity_rows["finite outputs"].unique()
        if len(counts) != 1:
            raise ValueError(f"Inconsistent pooled output count for {network} {component} {quantity}")
        cells = [quantity_labels[quantity], f"{int(counts[0]):,}"]
        for method in INTERVAL_METHODS:
            method_rows = quantity_rows[quantity_rows["method"] == method]
            if len(method_rows) != 1:
                raise ValueError(
                    f"Expected one pooled MCS row for {network} "
                    f"{component} {quantity} {method}, found {len(method_rows)}"
                )
            cells.append(f"{method_rows['MCS [%]'].iloc[0]:.2f}")
        return cells

    lines: list[str] = []
    for network in NETWORKS:
        node_rows = validate_group(network, "Node")
        branch_rows = validate_group(network, "Branch terminal")
        for quantity_index, (node_quantity, branch_quantity) in enumerate(
            zip(
                quantity_groups["Node"],
                quantity_groups["Branch terminal"],
                strict=True,
            )
        ):
            network_cell = f"\\multirow{{3}}{{*}}{{{network}}}" if quantity_index == 0 else ""
            cells = [
                *quantity_cells(network, "Node", node_quantity, node_rows),
                *quantity_cells(network, "Branch terminal", branch_quantity, branch_rows),
            ]
            lines.append(f"        {network_cell} & " + " & ".join(cells) + r" \\")
        lines.append(r"        \midrule")

    content = "\n".join([*lines[:-1], r"        \bottomrule"]) + "\n"
    (RESULTS_ROOT / "one_sigma_interval_rows.tex").write_text(content, encoding="utf-8")


def power_law_table(runtime: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, method_rows in runtime.groupby("method", sort=False):
        subset = method_rows.sort_values("buses")
        buses = subset["buses"].to_numpy(dtype=float)
        elapsed = subset["median time [s]"].to_numpy(dtype=float)
        exponent, log_prefactor = np.polyfit(np.log(buses), np.log(elapsed), deg=1)
        fitted = log_prefactor + exponent * np.log(buses)
        residual = np.sum((np.log(elapsed) - fitted) ** 2)
        total = np.sum((np.log(elapsed) - np.log(elapsed).mean()) ** 2)
        rows.append(
            {
                "method": method,
                "minimum buses": int(buses.min()),
                "maximum buses": int(buses.max()),
                "fit points": len(subset),
                "C [s]": float(np.exp(log_prefactor)),
                "p": float(exponent),
                "R2 log time": float(1.0 - residual / total),
            }
        )
    return pd.DataFrame(rows)


def _latex_scientific(value: float) -> str:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Expected a finite positive coefficient, received {value}")
    exponent = math.floor(math.log10(value))
    mantissa = value / 10.0**exponent
    return rf"${mantissa:.2f}\!\times\!10^{{{exponent}}}$"


def write_timing_table_rows(timing_laws: pd.DataFrame) -> None:
    """Write the empirical timing-law rows used by the manuscript."""
    required_columns = {"method", "C [s]", "p", "R2 log time"}
    missing_columns = required_columns.difference(timing_laws.columns)
    if missing_columns:
        raise ValueError("Timing-law table is missing columns: " + ", ".join(sorted(missing_columns)))
    if timing_laws["method"].duplicated().any():
        raise ValueError("Timing-law methods are not unique")

    expected_methods = {method for method, _ in TIMING_METHODS}
    actual_methods = set(timing_laws["method"])
    if actual_methods != expected_methods:
        raise ValueError(
            f"Unexpected timing-law methods: expected {sorted(expected_methods)}, found {sorted(actual_methods)}"
        )

    indexed = timing_laws.set_index("method")
    lines: list[str] = []
    for method, label in TIMING_METHODS:
        row = indexed.loc[method]
        coefficient = float(row["C [s]"])
        exponent = float(row["p"])
        r_squared = float(row["R2 log time"])
        if not np.isfinite([coefficient, exponent, r_squared]).all():
            raise ValueError(f"Non-finite timing-law value for {method}")
        lines.append(f"        {label} & {_latex_scientific(coefficient)} & {exponent:.2f} & {r_squared:.3f} " + r"\\")

    content = "\n".join([*lines, r"        \bottomrule"]) + "\n"
    (RESULTS_ROOT / "timing_power_law_rows.tex").write_text(content, encoding="utf-8")


def plotting_module() -> ModuleType:
    """Load the optional plotting stack only when figures are requested."""
    importlib.import_module("scienceplots")
    return importlib.import_module("matplotlib.pyplot")


def plot_runtime(runtime: pd.DataFrame) -> None:
    plt = plotting_module()
    specifications = [
        ("Deterministic ILSE", "Det. ILSE", "#003C64", "-", "o"),
        ("Deterministic NRSE", "Det. NRSE", "#8C2D04", "-", "s"),
        ("Analytical UQ ILSE", "Analytical UQ ILSE", "#0072B2", "--", "^"),
        ("Analytical UQ NRSE", "Analytical UQ NRSE", "#D55E00", "--", "v"),
        ("Monte Carlo ILSE", "MC UQ ILSE", "#56B4E9", "-.", "D"),
        ("Monte Carlo NRSE", "MC UQ NRSE", "#E69F00", "-.", "P"),
        ("Gain inverse NumPy", "Gain inv. NumPy", "#7B3294", ":", "X"),
        (
            "Gain inverse SciPy",
            "Gain inv. SciPy",
            "#009E73",
            (0, (5, 1, 1, 1, 1, 1)),
            "*",
        ),
    ]
    with plt.style.context(["science", "ieee"]):
        plt.rcParams["savefig.bbox"] = None
        figure, axis = plt.subplots(figsize=(3.5, 3.35))
        for method, label, color, line_style, marker in specifications:
            subset = runtime[runtime["method"] == method].sort_values("buses")
            axis.plot(
                subset["buses"],
                subset["median time [s]"],
                color=color,
                linestyle=line_style,
                marker=marker,
                markersize=3,
                linewidth=0.9,
                label=label,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Number of buses")
        axis.set_ylabel("Median wall time (s)")
        axis.grid(visible=True, which="both", alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        legend_order = (0, 2, 4, 6, 1, 3, 5, 7)
        figure.legend(
            [handles[index] for index in legend_order],
            [labels[index] for index in legend_order],
            loc="upper center",
            ncol=2,
            fontsize=7,
            frameon=False,
            columnspacing=0.8,
            handlelength=2.0,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.78))
        figure.savefig(FIGURE_ROOT / "ternary_tree_runtime.pdf", bbox_inches=None)
        plt.close(figure)


def plot_cigre_lv_power_sigmas() -> None:
    plt = plotting_module()
    comparison = pd.read_csv(RAW_ACCURACY_ROOT / "cigre_lv" / "cigre_lv_uq_comparison.csv")
    comparison = comparison[
        (comparison["sensor setting"] == "voltage + current + power")
        & (comparison["component"] == "line")
        & (comparison["terminal"] == "from")
        & comparison["quantity"].isin(["active power", "reactive power"])
    ]
    line_ids = sorted(comparison["id"].unique())[:8]
    comparison = comparison[comparison["id"].isin(line_ids)]
    method_columns = [
        ("Analytical UQ ILSE sigma", "Analytical ILSE", "#0072B2"),
        ("Analytical UQ NRSE sigma", "Analytical NRSE", "#D55E00"),
        ("UQ gain inv sigma", "Gain inverse NumPy", "#009E73"),
        ("Monte Carlo UQ ILSE sigma", "MC ILSE", "#CC79A7"),
        ("Monte Carlo UQ NRSE sigma", "MC NRSE", "#56B4E9"),
        ("Pandapower MC WLS sigma", "pandapower MC WLS", "#E69F00"),
    ]

    with plt.style.context(["science", "ieee"]):
        plt.rcParams["savefig.bbox"] = None
        figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.5), sharey=False)
        positions = np.arange(len(line_ids), dtype=float)
        width = 0.13
        for axis, quantity, ylabel, panel in zip(
            axes,
            ("active power", "reactive power"),
            (r"Active-power sigma (kW)", r"Reactive-power sigma (kvar)"),
            ("(a)", "(b)"),
            strict=True,
        ):
            subset = comparison[comparison["quantity"] == quantity].set_index("id").loc[line_ids]
            for index, (column, label, color) in enumerate(method_columns):
                axis.bar(
                    positions + (index - (len(method_columns) - 1) / 2) * width,
                    subset[column].to_numpy(dtype=float) / 1_000.0,
                    width=width,
                    color=color,
                    edgecolor="black",
                    linewidth=0.35,
                    label=label,
                )
            axis.set_xlabel("Line ID")
            axis.set_ylabel(ylabel)
            axis.set_xticks(positions)
            axis.set_xticklabels([str(line_id) for line_id in line_ids])
            axis.grid(axis="y", alpha=0.25)
            axis.text(0.02, 0.96, panel, transform=axis.transAxes, va="top", fontweight="bold")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=6, fontsize=7, frameon=False)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        figure.savefig(FIGURE_ROOT / "cigre_lv_power_sigmas.pdf", bbox_inches=None)
        plt.close(figure)


def plot_gain_inverse_accuracy(accuracy: pd.DataFrame) -> None:
    plt = plotting_module()
    gain_inverse = (
        accuracy[accuracy["method"] == "Gain inverse NumPy"]
        .sort_values("zero-injection buses [%]")
        .reset_index(drop=True)
    )
    series = [
        ("Voltage magnitude median APE [%]", r"$\sigma_{|U|}$", "#0072B2", "o"),
        ("Current magnitude median APE [%]", r"$\sigma_{|I|}$", "#D55E00", "s"),
        ("Active power median APE [%]", r"$\sigma_P$", "#009E73", "^"),
        ("Reactive power median APE [%]", r"$\sigma_Q$", "#CC79A7", "D"),
    ]
    percentages = gain_inverse["zero-injection buses [%]"].to_numpy(dtype=float)
    values = gain_inverse[[column for column, *_ in series]].to_numpy(dtype=float)
    if values.shape != (len(NETWORKS), len(series)) or not np.isfinite(values).all():
        raise ValueError("Expected finite gain-inverse errors for all accuracy entries")

    with plt.style.context(["science", "ieee"]):
        plt.rcParams["savefig.bbox"] = None
        figure, axis = plt.subplots(figsize=(3.5, 2.7))
        for column, label, color, marker in series:
            axis.scatter(
                percentages,
                gain_inverse[column],
                color=color,
                marker=marker,
                s=25,
                edgecolor="black",
                linewidth=0.35,
                label=label,
                zorder=3,
            )
        axis.set_xlabel(r"Zero-injection buses (\%)")
        axis.set_ylabel(r"Gain-inverse MdAPE (\%)")
        axis.set_yscale("log")
        axis.set_xlim(-3.0, 64.0)
        axis.set_ylim(3.0, 110.0)
        axis.set_xticks(percentages)
        axis.set_xticklabels(
            [f"{percentage:.2f}" for percentage in percentages],
            rotation=35,
            ha="right",
        )
        axis.set_yticks([4.0, 10.0, 20.0, 50.0, 100.0])
        axis.set_yticklabels(["4", "10", "20", "50", "100"])
        axis.grid(visible=True, which="both", axis="y", alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            fontsize=7,
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.3,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
        figure.savefig(FIGURE_ROOT / "gain_inverse_accuracy_vs_zi.pdf", bbox_inches=None)
        plt.close(figure)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(root: Path) -> dict[str, object]:
    def run_git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = run_git("status", "--porcelain", "--untracked-files=normal")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status entries": len(status.splitlines()) if status else 0,
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in (
        "power-grid-model",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "pandapower",
        "scienceplots",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def source_provenance() -> dict[str, object]:
    missing = [str(path) for path in SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing experiment source files: " + ", ".join(missing))
    return {
        "repositories": {
            "power-grid-model": git_provenance(PGM_ROOT),
        },
        "python": sys.version,
        "packages": package_versions(),
        "source file sha256": {str(path.relative_to(PGM_ROOT)): file_sha256(path) for path in SOURCE_FILES},
    }


def rebuild_table_fragments() -> None:
    """Regenerate both LaTeX tables from the retained compact CSV summaries."""
    native_scores = pd.read_csv(RESULTS_ROOT / "one_sigma_interval_scores_by_voltage_level.csv")
    standardized_scores = pd.read_csv(RESULTS_ROOT / "one_sigma_interval_scores_standardized.csv")
    validate_interval_score_tables(native_scores, standardized_scores)
    write_interval_table_rows(standardized_scores)

    timing_laws = pd.read_csv(RESULTS_ROOT / "timing_power_laws.csv")
    write_timing_table_rows(timing_laws)


def main(*, tables_only: bool = False) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    if tables_only:
        rebuild_table_fragments()
        return

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    accuracy, sample_counts = accuracy_table()
    accuracy.to_csv(RESULTS_ROOT / "accuracy_median_ape.csv", index=False)
    absolute_sigmas = absolute_sigma_table()
    validate_absolute_sigma_table(absolute_sigmas, accuracy)
    absolute_sigmas.to_csv(RESULTS_ROOT / "absolute_sigma_medians_by_voltage_level.csv", index=False)
    native_interval_scores, standardized_interval_scores = interval_score_tables()
    validate_interval_score_tables(native_interval_scores, standardized_interval_scores)
    native_interval_scores.to_csv(RESULTS_ROOT / "one_sigma_interval_scores_by_voltage_level.csv", index=False)
    standardized_interval_scores.to_csv(RESULTS_ROOT / "one_sigma_interval_scores_standardized.csv", index=False)
    write_interval_table_rows(standardized_interval_scores)
    interval_sample_counts: dict[str, dict[str, object]] = {}
    for network, network_scores in standardized_interval_scores.groupby("network", sort=False):
        common_counts = network_scores["common converged samples"].unique()
        if len(common_counts) != 1:
            raise ValueError(f"Inconsistent common convergence count for {network}")
        interval_sample_counts[network] = {
            "attempted": int(network_scores["attempted samples"].iloc[0]),
            "common converged": int(common_counts[0]),
            "method converged": {
                method: int(
                    network_scores.loc[
                        network_scores["method"] == method,
                        "method converged samples",
                    ].iloc[0]
                )
                for method in INTERVAL_METHODS
            },
            "sigma calibration samples": {
                method: int(
                    network_scores.loc[
                        network_scores["method"] == method,
                        "sigma calibration samples",
                    ].iloc[0]
                )
                for method in INTERVAL_METHODS
            },
        }

    runtime_source = RAW_RUNTIME_ROOT / "ternary_tree_runtime.csv"
    runtime_destination = RESULTS_ROOT / "ternary_tree_runtime.csv"
    shutil.copyfile(runtime_source, runtime_destination)
    runtime = pd.read_csv(runtime_destination)
    timing_laws = power_law_table(runtime)
    timing_laws.to_csv(RESULTS_ROOT / "timing_power_laws.csv", index=False)
    write_timing_table_rows(timing_laws)

    runtime_metadata = json.loads((RAW_RUNTIME_ROOT / "runtime_metadata.json").read_text(encoding="utf-8"))
    metadata = {
        "accuracy Monte Carlo samples": sample_counts,
        "accuracy seed": 2026,
        "accuracy baseline": "Monte Carlo UQ NRSE",
        "accuracy metric": (
            "median absolute percentage error over the common finite-output "
            "intersection of all reported methods with nonzero baseline sigma"
        ),
        "absolute sigma metric": (
            "median output standard deviation over the same common finite-output "
            "intersection as the accuracy table, grouped by node or branch "
            "terminal and terminal-node rated voltage"
        ),
        "absolute sigma units": {
            "voltage magnitude": "V",
            "current magnitude": "A",
            "active power": "kW",
            "reactive power": "kvar",
        },
        "one-sigma interval scores": {
            "truth": "noise-free Newton-Raphson power-flow output",
            "centers": {
                "Analytical UQ ILSE": "scenario-specific ILSE estimate",
                "Monte Carlo UQ ILSE": "scenario-specific ILSE estimate",
                "Analytical UQ NRSE": "scenario-specific NRSE estimate",
                "Monte Carlo UQ NRSE": "scenario-specific NRSE estimate",
                "Gain inverse NumPy": "scenario-specific NRSE estimate",
            },
            "half-widths": {
                "Analytical UQ ILSE": "scenario-specific analytical sigma",
                "Monte Carlo UQ ILSE": "independent 1,000-sample ILSE sigma",
                "Analytical UQ NRSE": "scenario-specific analytical sigma",
                "Monte Carlo UQ NRSE": "independent 1,000-sample NRSE sigma",
                "Gain inverse NumPy": "clean reference-point gain-inverse sigma",
            },
            "lower endpoint": "center - half-width",
            "upper endpoint": "center + half-width",
            "evaluation samples": 1_000,
            "evaluation seed": 2026,
            "Monte Carlo sigma training samples": 1_000,
            "Monte Carlo sigma training seed": 2027,
            "sample counts": interval_sample_counts,
            "nominal coverage": ONE_SIGMA_COVERAGE,
            "alpha": ONE_SIGMA_ALPHA,
            "aggregation": (
                "mean over the output specifications exported for the accuracy "
                "study, restricted to the common finite five-method intersection "
                "with nonzero empirical variance and common converged scenarios; "
                "native scores are grouped by node or branch terminal, terminal-node "
                "rated voltage, and output quantity"
            ),
            "excluded outputs": (
                "deterministic outputs with zero empirical variance are excluded "
                "after verifying 100 percent repeated-sampling coverage; "
                "outputs not exported by the accuracy notebooks are outside the "
                "reported population"
            ),
            "standardization": (
                "group-mean MWS and MWIS divided by the group-mean empirical "
                "standard deviation of the matched independent truth-centered "
                "1,000-sample training batch"
            ),
            "MWS and MWIS units": {
                "voltage magnitude": "V",
                "current magnitude": "A",
                "active power": "kW",
                "reactive power": "kvar",
            },
        },
        "network statistics": NETWORK_STATISTICS,
        "runtime": runtime_metadata,
        "artifact storage": {
            "root": str(WORKFLOW_ROOT.relative_to(PGM_ROOT)),
            "relocated without recomputation": False,
        },
        "source provenance": source_provenance(),
    }
    (RESULTS_ROOT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    plot_runtime(runtime)
    plot_cigre_lv_power_sigmas()
    plot_gain_inverse_accuracy(accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="regenerate LaTeX table fragments from retained summary CSV files",
    )
    arguments = parser.parse_args()
    main(tables_only=arguments.tables_only)
