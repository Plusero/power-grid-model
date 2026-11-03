<!-- SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org> -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Internal asymmetric state estimation uncertainty

The uncertainty calculation of the
[uncertainty algorithm](state-estimation-uncertainty.md) is written once
for both symmetries. This page lists what differs in an asymmetric (three-phase)
calculation. The [observer workflow](../user_manual/uncertainty-observers.md)
describes selection, targets, updates and serialization, and the
[asymmetric example](../../examples/python/asymmetric_observer_uncertainty.py)
shows a complete `calculate_state_estimation(symmetric=False, ...)` call.

## Blocks and coordinates

Every bus has three phases, so each matrix has one block per pair of buses.

| Object | Coordinates of one bus | Block of two buses |
| --- | --- | --- |
| ILSE augmented unknown | `[U_a, U_b, U_c, phi_a, phi_b, phi_c]` | complex 6 by 6 |
| ILSE voltage state | `[U_a, U_b, U_c]` | complex 3 by 3 |
| NRSE augmented unknown | `[theta_abc, v_abc, phi_P_abc, phi_Q_abc]` | real 12 by 12 |
| NRSE voltage state | `[theta_a, theta_b, theta_c, v_a, v_b, v_c]` | real 6 by 6 |
| Rectangular voltages | `[Re(U)_abc, Im(U)_abc]` | real 6 by 6 |

The blocks keep the coupling between phases and between buses. A phase sigma is
the standard deviation of that phase in the coupled model, not the result of
three single-phase calculations.

The measurement weights and the variance normalization are those of the
estimator, with one normalization factor for the island and not one per phase.
A sensor that provides one value for all three phases enters as three
independent measurements. Correlated measurement errors are not modelled.

For ILSE, the covariance of the rectangular voltages follows from a complex
block `C` as in the symmetric case:

```text
0.5 * [[Re(C), -Im(C)], [Im(C), Re(C)]]
```

NRSE converts its polar covariance with the Jacobian of `U = v exp(j theta)`
and has no factor of one half.

## Angle reference

If the island has a measured voltage angle, angles are absolute. Otherwise all
three phases of every bus are reported relative to the phase-a angle `theta_s`
of the reference bus. Only that one angle is zero by definition: phases b and c
of the reference bus have an uncertainty relative to phase a. As in the
estimator, a voltage sensor without an angle in one of its phases is used as a
magnitude measurement.

An angle that is reported relative to `theta_s` has variance

```text
Var(angle) + Var(theta_s) - 2 Cov(angle, theta_s)
```

for voltage angles and current angles of both estimators. One additional solve
gives the covariance of every state with `theta_s`. Voltage magnitude, power and
current magnitude do not depend on the reference. A current angle is the angle
of the current phasor in this reference, not the angle relative to the local
voltage that a current sensor can measure.

Without a measured voltage angle, NRSE fixes the reference with three virtual
angle measurements, one per phase, at a bus that need not be the reference bus.
The state covariance is taken from the inverse of the matrix that includes
these rows, and their contribution is not removed afterwards. On the three-phase
test network the reported angles of phases b and c relative to phase a therefore
exceed the first-order variance by about twice the voltage sensor variance, and
current angles by up to 15%, while magnitudes, power and injections match it.
The estimate itself is not affected.

## Branch and injection outputs

The current of phase `p` at a branch side or at a bus sums over all phases of
all buses involved:

```text
I_p = sum_kq Y_pkq U_kq
S_p = U_p conj(I_p)
```

The derivative of one phase output therefore has entries for all three phases of
both buses of a branch, or of the bus and all its neighbours for an injection.
The variance is `d C d^T` with the rectangular covariance blocks of those buses.

Observer output is converted per phase: voltage with `u_rated / sqrt(3)`, power
with one third of the three-phase base power, and current with the base current
of the selected terminal, also for three-winding transformers. Angles stay in
radians. Total power, line-to-line voltage and sequence components are not
provided, because they need the covariance between phases.

## Completion and validity

Covariance completion uses whole blocks: the complex 6 by 6 or real 12 by 12
blocks of the inverse for the augmented path, and the real 6 by 6 rectangular
blocks for the mean path. The order of the common neighbours, the `1e-12`
conditioning limit and the rounding tolerance of `1000 * epsilon` are those of
the symmetric algorithm. As there, a pair that is completed along a path that is
not a separator is an approximation, and it can make an injection variance
negative, which is then reported as invalid.

Validity is per phase and per output. A voltage or current that is exactly zero
in one phase has no magnitude or angle derivative in that phase, while power
stays defined. An exact zero-injection constraint is recognised per phase, where
the measured P and Q and both of their variances are exactly zero.

The tests compare the results with a dense inverse, with finite differences of
the output functions, and with first-order propagation of the sensor variances
through the estimator. They do not establish calibration against sampled
measurement noise for a particular network.
