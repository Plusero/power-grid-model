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

An explicitly empty attribute list for an observer component omits that
component. No uncertainty is calculated when the model has no observers, when
all of them are disabled or disconnected, or when no sigma attribute is
requested. Observers of the same target keep their own output rows and share
one calculation, and each island is factorized once however many observers it
has.

Observer updates contain only `id` and `status`; targets and terminal selectors
are immutable. Missing status means unchanged. Existing full-component positional
updates without IDs, sparse/dense batches, Cartesian-product batches and
`continue_on_batch_error` retain their usual semantics. Topology, status,
measurement and reference changes are resolved independently for each scenario;
failed scenarios use the normal batch error/output conventions.

Observer output is not available for power flow, short circuit and other
state-estimation methods. Default power-flow and short-circuit output excludes
observers. Explicitly requesting observer output in those calculations raises
an error, also for disabled observers or when only `id` and `energized` are
requested.

## Validity and interpretation

`energized=1` means that the observer is enabled and its target terminal is
energized. It says nothing about whether a sigma is valid or exact. Disabled or
disconnected observers have `energized=0` and missing sigmas. An energized
observer can also have a missing sigma. In asymmetric output this is per phase
and per attribute: phases a and c can be valid while phase b is missing.

A sigma is missing (NaN) if

- the voltage or current is exactly zero, where its magnitude and angle have no
  derivative. Active and reactive power stay defined at zero power. There is no
  cutoff near zero, where a first-order value becomes inaccurate;
- the calculated variance is not finite, or is negative by more than rounding
  error. A negative variance within rounding error is reported as zero;
- the uncertainty calculation of the island fails numerically. The state
  estimation result is kept and every requested sigma of the island is missing.

A node-injection sigma is zero where the injection is an exact constraint: the
measured active and reactive power and both of their variances are exactly
zero, in asymmetric calculations per phase. A zero load, or a zero reading with
a nonzero variance, is not such a constraint.

Angles use the reference of the estimated voltages: absolute if the island has
a measured voltage angle, otherwise relative to the angle of the island's
reference bus. In asymmetric calculations that is the phase-a angle of the
reference bus for all three phases, so phases b and c of the reference bus
still have an angle uncertainty. A current angle is the angle of the current
phasor in this reference, not the angle relative to the local voltage that a
current sensor can measure.

### What the sigmas represent

A sigma is the first-order standard deviation of the estimated quantity under
the measurement model of the calculation method, with independent measurement
errors. It is not a confidence interval and it is not calibrated against
measured data. The two methods use different measurement models and give
different sigmas for the same network.

- `newton_raphson` uses the sensor variances as given. Its variances agree with
  the first-order propagation of the sensor variances through the estimator,
  with one exception: in an asymmetric calculation without a measured voltage
  angle, the angles of phases b and c relative to phase a are overestimated. On
  the test network their variance is too large by about twice the voltage
  sensor variance, and the variance of current angles by up to 15%. Magnitudes,
  power and injections are not affected.
- `iterative_linear` treats every measurement as a complex number whose error
  has the same variance in every direction of the complex plane. A voltage
  measurement gets twice the sensor variance: the sensor variance for the
  magnitude, as in `newton_raphson`, and the same variance across the phasor
  for a measured angle, which is the `newton_raphson` angle variance at a
  voltage of 1 p.u. On the balanced test network its voltage variances are
  then within 10% of the first-order propagation when the angle is measured.
  Two differences remain.
  The active and reactive part of a power, or the two components of a current,
  share one variance, so its power and current variances lie between 0.5 and
  1.7 of the first-order propagation, depending on the quantity. And without a
  measured voltage angle, the angle relative to the reference is reported at
  0.5 to 0.7 of it.

A node-injection sigma needs the covariance between every two neighbours of the
node. Where the calculation does not provide it directly, it is derived through
a bus that both neighbours are connected to. This is exact if every path
between the two neighbours passes through that bus, as in a radial network, and
an approximation otherwise. An approximated value is neither exact nor an upper
bound, and it is missing if the approximation makes the variance negative.

The asymmetric calculation keeps the coupling between phases and between buses.
Correlated measurement errors are not modelled: a measurement that is given
once for all three phases counts as three independent measurements.

The [uncertainty algorithm](../algorithms/state-estimation-uncertainty.md)
and its [asymmetric extension](../algorithms/asymmetric-state-estimation-uncertainty.md)
describe the calculation.

## Serialization

Standard JSON and MessagePack serialization support observer input, update and
output datasets in row or column form, using default maps or compact lists.
Native/Python missing doubles are NaN. Compact serialization writes `null` for
missing fields; default map serialization omits wholly missing attributes.
A retained asymmetric array always has three positions, with `null` for each
missing phase. For example, `[1.2, null, 3.4]` retains valid a/c values and the
missing b position in both JSON and MessagePack roundtrips. The existing dataset
envelope and calculation signatures are unchanged.
