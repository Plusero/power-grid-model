<!--
SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

# Native pandapower networks in the UQ examples

The PGM input remains the experiment definition: it supplies component IDs and ordering, sensors, Monte Carlo
measurement samples, zero-injection buses, and the output layout. Pandapower supplies the electrical network used by
its WLS estimator.

## Network selection

When the benchmark is available from `pandapower.networks`, the notebook creates that native network and passes it to
`run_pandapower_monte_carlo()`. The helper deep-copies it, so the imported object is not mutated. If no native network
is supplied, the helper falls back to constructing a pandapower network from PGM components.

The native-network path is used for CIGRE MV, IEEE33, MV Oberrhein, and LV Schutterwald. The meshed CIGRE case
additionally closes the native network's line switches.

## Alignment and validation

PGM IDs are mapped to pandapower table indices using the component row order preserved by the original conversion.
Before estimation, the helper verifies component counts, bus rated voltages, source buses, branch endpoints, line
parameters, transformer endpoints and sides, and terminal switch states. Any mismatch raises an error instead of
comparing different networks.

Branches that are already open are disabled for WLS, and their omitted measurement channels are reported.

## Monte Carlo flow

For every sample, the same seeded PGM measurement realization is copied into pandapower with unit conversion: V to
per unit, radians to degrees, W/var to MW/Mvar, and A to kA. PGM zero-injection buses are applied as equality
constraints. Disconnected feeders are estimated separately and merged. Finally, pandapower results are converted back
to PGM order, terminal orientation, and SI units before their empirical standard deviations are compared with Monte
Carlo PGM NRSE.

Pandapower WLS does not consume the current-angle channel used by these PGM examples. If removing that channel makes
a layout under-observable, the result is reported as unavailable rather than as zero uncertainty.
