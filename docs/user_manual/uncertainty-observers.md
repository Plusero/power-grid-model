<!-- SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org> -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# State estimation uncertainty observers

Uncertainty observers request first-order standard deviations of estimated
voltage, terminal power/current, or net node injection. Existing sensors supply
the measurements and their modeled variances. Observers add no measurements or
constraints and do not change the estimated electrical quantities.

Use symmetric or asymmetric state estimation with
`CalculationMethod.iterative_linear` or `CalculationMethod.newton_raphson`. Add
the required observer input records and call the existing `PowerGridModel.calculate_state_estimation`; no additional
calculation flag is needed. A complete executable network is provided in
[the symmetric example](../../examples/python/observer_uncertainty.py):

```sh
python examples/python/observer_uncertainty.py --method iterative_linear
python examples/python/observer_uncertainty.py --method newton_raphson
```

The example constructs ordinary voltage measurements, all three observer types,
both line terminals and a node-injection observer, then writes compact JSON
through the standard serializer. For a coupled, unequal-phase network, use
[the asymmetric example](../../examples/python/asymmetric_observer_uncertainty.py):

```sh
python examples/python/asymmetric_observer_uncertainty.py --method iterative_linear
python examples/python/asymmetric_observer_uncertainty.py --method newton_raphson
```

Asymmetric calls set the existing `symmetric=False` argument. The observer
components, input fields and status updates are the same for either symmetry.

## Components and units

All observer inputs require a globally unique `id`, an existing
`observed_object` ID and `status` (`0` disabled, `1` enabled). Power and current
observers additionally require `observed_terminal_type`, using
`ObservedTerminalType`. Voltage observers accept nodes and have no terminal
selector.

| Component | Sigma attributes | Symmetric units | Asymmetric units |
| --- | --- | --- | --- |
| `voltage_uncertainty_observer` | `u_sigma`, `u_angle_sigma` | line-to-line V, rad | phase-to-neutral V, rad |
| `power_uncertainty_observer` | `p_sigma`, `q_sigma` | total three-phase W, var | per-phase W, var |
| `current_uncertainty_observer` | `i_sigma`, `i_angle_sigma` | selected terminal A, rad | per-phase terminal A, rad |

`id` and `energized` are scalar in both output datasets. Sigma attributes are
scalar in `sym_output` and three-entry arrays in `asym_output`, ordered a/b/c.
Each selected sigma attribute requests all three phases. Asymmetric voltage
uses the phase base `u_rated / sqrt(3)` and power uses one third of the
three-phase power base; current uses the selected terminal's existing current
base. These outputs do not provide line-to-line, total-power or sequence
uncertainty. Those quantities require covariance cross terms and cannot be
recovered by combining phase sigmas alone.

Power/current targets support `line`, `asym_line`, `generic_branch` and
`transformer` with `branch_from`/`branch_to` (codes 0/1). Three-winding
transformers use `branch3_1`/`branch3_2`/`branch3_3` (codes 6/7/8), including the
physical terminal's voltage/current base. Branch quantities point into the
selected terminal. Power observers also support `node_injection` (code 9), in
injection direction, where the physical node maps one-to-one to a mathematical
bus. Closed links that fuse physical nodes do not determine individual node
injection attribution; requesting an enabled injection sigma there raises an
argument error. Voltage observers remain supported on fused nodes.

Appliance, ideal-link, node-current and other target/selector combinations are
unsupported. Disabled inputs still require valid IDs and selectors. An observer
ID in output identifies the observer, not its target.

## Selection, updates and batches

Default state-estimation output includes observers in the model. Select output
components and attributes through `output_component_types` as usual. For
example, requesting only `id` and `energized` reads metadata without computing
uncertainty:

```python
result = model.calculate_state_estimation(
    calculation_method=CalculationMethod.newton_raphson,
    output_component_types={ComponentType.voltage_uncertainty_observer: ["id", "energized"]},
)
```

An explicit empty observer attribute list omits that component. No observers,
all observers disabled, disconnected targets, or no requested sigma fields
perform no additional uncertainty factorization. Duplicate observers retain
input row order and IDs while sharing one numerical request per target and
quantity. An island shares one covariance workspace and selective-inverse
sweep, independent of duplicate observer count.

Observer updates contain only `id` and `status`; targets and terminal selectors
are immutable. Missing status means unchanged. Existing full-component positional
updates without IDs, sparse/dense batches, Cartesian-product batches and
`continue_on_batch_error` retain their usual semantics. Topology, status,
measurement and reference changes are resolved independently for each scenario;
failed scenarios use the normal batch error/output conventions.

Observer output is unsupported for power flow, short circuit and other
estimation methods, including metadata-only or disabled observer output
requests. Explicitly exclude observer output in those calls.
Default power-flow/short-circuit output already excludes observers.

## Validity and interpretation

`energized=1` means that the observer is enabled and its selected target terminal
is energized. It is not a covariance-validity, exactness or physical-observability
certificate. Disabled/disconnected outputs have `energized=0` and missing sigmas.
An undefined or invalid quantity remains energized but has a missing sigma.
Asymmetric validity is per phase and quantity: a row can retain valid phase a/c
sigmas while phase b is missing.

Exactly zero voltage/current makes the corresponding magnitude and angle
first-order derivatives undefined. Signed P/Q maps can remain valid at zero
nominal power. There is no arbitrary near-zero cutoff; first-order accuracy can
deteriorate close to zero. Nonfinite or materially negative variances produce
missing output; only scale-relative roundoff negatives are clipped. A physical
unit conversion that overflows or turns a positive value into floating-point
zero also produces a missing value.

Node-injection sigmas are reported as zero only when processed P and Q and both
modeled measurement variances are exactly zero. For asymmetric output, this
predicate applies separately to each processed phase. A nominal zero load or
noisy zero measurement does not meet that predicate. A numerical failure of the
uncertainty calculation keeps the state-estimation result and reports every
requested sigma as missing (NaN); no other covariance algorithm is tried.

Angles use an absolute reference when one is available; otherwise they are
relative to the island's reported slack angle. In asymmetric calculations,
the same slack-phase-a angle is subtracted from every phase; slack phases b/c
can retain relative-angle uncertainty. Current angle means the current
phasor angle in that reporting frame, including the terminal/slack covariance;
it is not a current sensor's local voltage-relative angle.

These sigmas describe the estimator's stated local approximation. ILSE uses
frozen transformed independent measurements and proper complex voltage
deviations, including its factor of one half in real projections. Power outputs
retain their induced pseudo-covariance. NRSE uses the returned-state real polar
model with exact constraints and the specified virtual-angle correction. The
methods can therefore give different sigmas for the same measured network.
The asymmetric implementation preserves coupled phase and bus covariance;
it does not run three independent scalar estimators. Its statistical scope
remains the prepared diagonal measurement-noise model. In particular,
broadcast measurements use that model's weights and do not introduce a
physical common-channel covariance. General correlated measurement errors and
improper ILSE state covariance are outside this calculation.
Missing injection covariance blocks use the first sufficiently conditioned
full augmented common-middle block in mathematical bus order. Separating
middles certify an exact-arithmetic identity; nonseparating middles produce an
approximation. Only conditioning failure for every candidate invokes an
equal-weight mean of original physical two-branch paths. Missing inputs, failed
arithmetic and empty path pools produce invalid results. Finite residual or
pair-bound warnings remain diagnostics and do not select another method.
Approximate results are neither guaranteed
exact nor conservative. See the internal
[symmetric uncertainty algorithm](../algorithms/symmetric-state-estimation-uncertainty.md)
and its [asymmetric extension](../algorithms/asymmetric-state-estimation-uncertainty.md)
for numerical conventions and diagnostic provenance. These conventions do not
establish empirical calibration or coverage for an arbitrary network.

## Serialization

Standard JSON and MessagePack serialization support observer input, update and
output datasets in row or column form, using default maps or compact lists.
Native/Python missing doubles are NaN. Compact serialization writes `null` for
missing fields; default map serialization omits wholly missing attributes.
A retained asymmetric array always has three positions, with `null` for each
missing phase. For example, `[1.2, null, 3.4]` retains valid a/c values and the
missing b position in both JSON and MessagePack roundtrips. The existing dataset
envelope and calculation signatures are unchanged.

Native tests verify that absent/disabled/unrequested sigma output sends no UQ
context, duplicates share requests, and voltage-only requests perform no
injection-pair completion.
