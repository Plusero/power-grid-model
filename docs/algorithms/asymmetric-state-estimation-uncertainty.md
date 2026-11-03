<!-- SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org> -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Internal asymmetric state estimation uncertainty

Asymmetric ILSE and NRSE reuse the request-driven covariance workspace,
selective-inverse kernel, validity rules and completion policy described in the
[symmetric algorithm](symmetric-state-estimation-uncertainty.md). They retain
full phase and bus covariance blocks. Phase outputs are marginal standard
deviations from that coupled model, not results from independent scalar solves.
The [observer workflow](../user_manual/uncertainty-observers.md) describes
selection, targets, updates and serialization. The executable
[asymmetric example](../../examples/python/asymmetric_observer_uncertainty.py)
uses the existing `calculate_state_estimation(symmetric=False, ...)` call.

## Coordinates and statistical scope

| Object | Coordinate order within one bus | Bus-pair block |
| --- | --- | --- |
| ILSE augmented unknown | `[U_a, U_b, U_c, phi_a, phi_b, phi_c]` | complex 6 by 6 |
| ILSE voltage state | `[U_a, U_b, U_c]` | complex 3 by 3 |
| NRSE augmented unknown | `[theta_abc, v_abc, phi_P_abc, phi_Q_abc]` | real 12 by 12 |
| NRSE voltage state | `[theta_a, theta_b, theta_c, v_a, v_b, v_c]` | real 6 by 6 |
| Physical rectangular state | `[Re(U)_abc, Im(U)_abc]` | real 6 by 6 |

The prepared independent-noise model and exact constraints remain the source
of covariance. One shared prepared variance normalization is restored exactly
once; phases are not normalized independently. Broadcast sensors retain the
existing prepared diagonal weights. These outputs do not model physical
common-channel errors or arbitrary correlated measurement noise.

ILSE retains its proper-complex state approximation, including for transformed
power/current measurements and magnitude-only voltage measurements. For a
complex covariance block `C`, the equivalent rectangular block is

```text
0.5 * [[Re(C), -Im(C)], [Im(C), Re(C)]]
```

Zero state pseudo-covariance does not imply zero power-output pseudo-covariance.
The widely linear power derivative retains both terms. General improper state
propagation is outside this model. NRSE uses the returned-state real polar
Jacobian and has no ILSE factor of one half.

## Angles and physical output maps

Prepared absolute voltage angles retain an absolute reporting reference.
Otherwise every reported phase angle uses the same slack-phase-a reference.
Only the slack-a angle is an algebraic zero coordinate; slack b/c may have
uncertainty relative to slack a. Existing measurement preparation is unchanged:
a voltage sensor with any missing phase angle becomes magnitude-only.

Without absolute angles, NRSE removes the noise contribution of all three
virtual angle selectors before applying the one reporting projection. The
virtual bus may differ from the reporting slack. If `Z_ij` is the raw inverse
state corner, `T_i` contains its three virtual selector columns, and `Wv` is the
normalized virtual weight matrix, then

```text
C0_ij = alpha * (Z_ij - T_i Wv T_j^T)
e = [1, 0, 0, 0, 0, 0]^T
d = [1, 1, 1, 0, 0, 0]^T
r_i = C0_i,slack e
c = e^T C0_slack,slack e
C_report_ij = C0_ij - r_i d^T - d r_j^T + c d d^T
```

ILSE retains its proper covariance before reporting maps. Its voltage/current
angle derivatives subtract the slack-a derivative and include the corresponding
cross covariance. A current angle is the current phasor angle in this reporting
frame, rather than a sensor's local voltage-relative angle.

Branch maps use all four endpoint blocks and the full coupled terminal
admittance row. Injection P/Q maps use the complete closed-neighborhood
covariance, including cross-phase, cross-bus and shunt terms. For phase `p`,

```text
I_p = sum_kq Y_pkq U_kq
S_p = U_p conj(I_p)
dS_p = conj(I_p) dU_p + U_p conj(dI_p)
```

The observer adapter converts each sigma once. Phase voltage uses
`u_rated / sqrt(3)`, phase power uses one third of the three-phase power base,
and phase current uses the selected physical terminal's current base.
Three-winding transformers use the selected terminal's mapping and base.
Angles remain in radians. There are no public total-power, line-to-line,
sequence-component or covariance fields.

## Completion and validity

Missing-pair completion uses full raw complex 6 by 6 or real 12 by 12 augmented
blocks. Original filled-pattern donors remain immutable; LU fill does not
create augmented graph adjacency. Common middles are visited in mathematical
bus order, and the first conditioned middle is selected. A separating middle
supports the exact raw-inverse identity. A nonseparating middle yields an
approximation. Only exhaustion of a nonempty candidate set by conditioning
failures permits the terminal mean of original physical paths. Physical support
and pseudoinverse operations retain phase coupling.

The inherited `1e-12` conditioning threshold, `sqrt(double_epsilon)` covariance
symmetry bound, `1000 * double_epsilon` roundoff/support rules and checked double
arithmetic remain unchanged. Invalid inputs or arithmetic do not select another
covariance engine. Completion validity does not certify approximation accuracy
or conservativeness.

Zero nominal voltage/current makes its magnitude and angle maps unavailable;
signed P/Q maps can remain valid. Exact-zero injection reporting applies only
to a phase whose processed P, Q and both variances are exactly zero. It preserves
raw numerical status and does not enforce zeros on other phases. Each public
sigma phase can therefore be valid or missing independently; `energized` still
describes enablement and connectivity.

Independent dense references and physical finite differences check local
algebra and API behavior. They do not establish nonlinear Monte Carlo
calibration, empirical coverage or a runtime guarantee. Those require separate
network- and noise-specific measurements.
