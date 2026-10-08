"""Extract the symmetric Y_bus that Power Grid Model builds internally.

The reusable API accepts a PGM ``input`` dataset and runs the compiled
``power_grid_model_c_example_extract_y_bus`` tool, which rebuilds PGM's internal
model state and dumps every ``math_solver::YBus`` in CSR form together with the
math-bus to input-node coupling.  The helper converts that dump into dense
complex matrices and, for a single connected sub-network, also returns the
matrix permuted into input node order.

The extractor binary is located through the ``PGM_EXTRACT_Y_BUS`` environment
variable, the executable search path, or the standard repository build output
directories, in that order.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from power_grid_model import AttributeType, ComponentType, DatasetType
from power_grid_model.utils import json_serialize

EXTRACTOR_ENV_VAR = "PGM_EXTRACT_Y_BUS"
EXTRACTOR_BINARY_NAME = "power_grid_model_c_example_extract_y_bus"
PGM_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PgmSubNetworkYBus:
    """One isolated mathematical network's Y_bus in PGM math-bus order."""

    bus_node_ids: np.ndarray
    row_indptr: np.ndarray
    col_indices: np.ndarray
    y_bus: np.ndarray
    is_radial: bool
    slack_bus: int


@dataclass(frozen=True)
class PgmYBusExtraction:
    """All sub-network Y_bus dumps plus the single-network input-order matrix."""

    version: str
    system_frequency: float
    sub_networks: tuple[PgmSubNetworkYBus, ...]
    y_bus_input_order: np.ndarray | None


def find_extractor(repository_root: Path = PGM_ROOT) -> Path | None:
    """Locate the compiled Y_bus extractor binary, or return ``None``."""
    environment_path = os.environ.get(EXTRACTOR_ENV_VAR)
    if environment_path:
        candidate = Path(environment_path).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        raise FileNotFoundError(f"{EXTRACTOR_ENV_VAR} does not point to an executable file: {candidate}")
    which_result = shutil.which(EXTRACTOR_BINARY_NAME)
    if which_result is not None:
        return Path(which_result)
    search_roots = [*sorted(repository_root.glob("cpp_build/*/bin")), repository_root / "build" / "bin"]
    for search_root in search_roots:
        candidate = search_root / EXTRACTOR_BINARY_NAME
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _dense_from_csr(row_indptr: np.ndarray, col_indices: np.ndarray, admittance: np.ndarray) -> np.ndarray:
    """Convert one CSR admittance dump into a dense complex matrix."""
    n_bus = row_indptr.size - 1
    y_bus = np.zeros((n_bus, n_bus), dtype=np.complex128)
    rows = np.repeat(np.arange(n_bus), np.diff(row_indptr))
    y_bus[rows, col_indices] = admittance
    return y_bus


def extract_pgm_y_bus(
    inputs: dict[ComponentType, np.ndarray],
    system_frequency: float,
    *,
    extractor: Path | None = None,
) -> PgmYBusExtraction:
    """Run the extractor on ``inputs`` and return the parsed Y_bus matrices.

    ``y_bus_input_order`` is only populated when the model has exactly one
    sub-network that covers every input node; otherwise it is ``None``.
    """
    if extractor is None:
        found = find_extractor()
        if found is None:
            raise FileNotFoundError(
                f"Could not locate {EXTRACTOR_BINARY_NAME}; build the C++ examples with PGM_ENABLE_DEV_BUILD=ON "
                f"or point {EXTRACTOR_ENV_VAR} at the binary"
            )
        extractor = found

    with tempfile.TemporaryDirectory(prefix="pgm_y_bus_") as temporary_directory:
        input_path = Path(temporary_directory) / "input.json"
        output_path = Path(temporary_directory) / "y_bus.json"
        input_path.write_text(
            json_serialize(data=inputs, dataset_type=DatasetType.input, use_compact_list=False, indent=-1),
            encoding="utf-8",
        )
        completed = subprocess.run(  # noqa: S603  # controlled binary and temporary-file arguments
            [str(extractor), str(input_path), str(output_path), repr(float(system_frequency))],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{extractor.name} failed: {completed.stderr.strip() or completed.stdout.strip()}")
        extracted = json.loads(output_path.read_text(encoding="utf-8"))

    sub_networks = tuple(
        PgmSubNetworkYBus(
            bus_node_ids=np.asarray(raw["bus_node_ids"], dtype=int),
            row_indptr=np.asarray(raw["row_indptr"], dtype=int),
            col_indices=np.asarray(raw["col_indices"], dtype=int),
            y_bus=_dense_from_csr(
                np.asarray(raw["row_indptr"], dtype=int),
                np.asarray(raw["col_indices"], dtype=int),
                np.asarray(raw["admittance_real"], dtype=float)
                + 1.0j * np.asarray(raw["admittance_imag"], dtype=float),
            ),
            is_radial=bool(raw["is_radial"]),
            slack_bus=int(raw["slack_bus"]),
        )
        for raw in extracted["sub_networks"]
    )

    y_bus_input_order: np.ndarray | None = None
    node_input = inputs.get(ComponentType.node)
    if (
        len(sub_networks) == 1
        and node_input is not None
        and node_input.size
        and sub_networks[0].bus_node_ids.size == node_input.size
    ):
        input_node_ids = node_input[AttributeType.id].astype(int)
        input_position = {int(node_id): position for position, node_id in enumerate(input_node_ids)}
        if set(sub_networks[0].bus_node_ids.tolist()) == set(input_node_ids.tolist()):
            math_positions = np.asarray([input_position[int(node_id)] for node_id in sub_networks[0].bus_node_ids])
            input_positions = np.argsort(math_positions)
            y_bus_input_order = sub_networks[0].y_bus[np.ix_(input_positions, input_positions)]

    return PgmYBusExtraction(
        version=str(extracted["version"]),
        system_frequency=float(extracted["system_frequency"]),
        sub_networks=sub_networks,
        y_bus_input_order=y_bus_input_order,
    )
