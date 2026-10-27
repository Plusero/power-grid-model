<!--
SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

# Delta-transformer observability

Reproducer for [issue #1668](https://github.com/PowerGridModel/power-grid-model/issues/1668).

The network contains a source on an MV bus, a delta/grounded-wye transformer,
and a load on an LV bus. The only measurements are the LV phase-to-ground
voltage magnitudes and the per-phase P/Q at the transformer's LV terminal.
The transformer and measurements use synthetic, rounded values.

An equal complex shift of all three MV phase voltages leaves the MV line-to-line
voltages and the LV measurements unchanged. These measurements therefore cannot
determine the MV common-mode voltage in asymmetric state estimation. The expected
behavior is to reject the configuration with `NotObservableError`.

The current topological observability check accepts this configuration. On the
tested original build, iterative linear SE subsequently raises `IterationDiverge`
and Newton-Raphson SE raises `SparseMatrixError`. Both methods are marked `xfail`
for those errors until the observability check is corrected. Numerical failure
details may vary by platform.

`asym_output.json` is an empty dataset used to register the asymmetric exception
case with the validation harness. No valid asymmetric estimate is asserted.
Symmetric state estimation assumes a balanced positive-sequence model, so it
does not test this missing common-mode information.

Run the Python validation cases from the repository root:

```sh
python -m pytest tests/unit/test_0Z_model_validation.py -k observability-delta-trafo --no-cov -rx
```

When implementing a fix, remove the `xfail` entries. The existing `raises`
expectation should then pass for both methods.
