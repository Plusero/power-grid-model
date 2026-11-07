<!-- SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org> -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Internal symmetric state estimation uncertainty

The optional `StateEstimationUQContext` is an internal, synchronous math-solver
interface. It leaves `SolverOutput` and public calculation signatures unchanged.
Requests use island-local mathematical indices, checked against the bus count.
Voltage requests may be unordered or repeated; result rows follow request order.
A row selecting no quantities retains the default `not_requested` fields.
Null and empty requests perform no uncertainty work. Callers supply fresh UQ
result storage for each calculation. Results are published only after the
ordinary solve and uncertainty calculation finish; empty requests, invalid
requests and estimator failures leave the supplied result unchanged. A gain
that fails a numerical check of the uncertainty calculation does not discard
the estimate: the estimate is returned and every requested uncertainty is
reported invalid.
This first implementation supports voltage requests for both symmetries; branch
and injection requests are declared but fail explicitly.

## Prepared model and normalization

The calculation reuses `MeasuredValues`, including aggregation, appliance status,
processed exact zero injections, independently modeled sensor variances and the
existing measurement conversions. Values and standard deviations are per unit;
angles use radians. `variance_normalization()` retains the minimum finite,
strictly positive prepared voltage variance or complex P/Q (or current real/imag)
variance sum. Exact zero constraints and missing/infinite rows do not select the
scale. If no positive stochastic row exists the normalization is one. The
ordinary finite reciprocal normalization preserves existing rounding; division
handles reciprocal overflow. This rule is shared by ordinary estimation.

ILSE freezes the estimator's transformed independent complex measurements. Its
power-to-current conversion retains the existing near-unit-voltage variance
approximation and combines real and imaginary variances. Proper complex voltage
deviations (zero state pseudo-covariance) are an explicit approximation, including
magnitude-only inputs. This is not general physical-noise sandwich propagation.
NRSE uses its local real polar Jacobian at the returned phasors, with coordinates
`[arg(U), abs(U)]`, independently of signed radii during iteration.

## Sparse covariance calculation

ILSE reuses the estimator's state-independent LU factors and matching block
permutations. It copies the factors for the destructive selected-inverse sweep;
there is no second gain assembly, reuse eligibility check or refactorization.
NRSE assembles a dedicated augmented matrix at the returned state.
Each bus has a complete complex 2 by 2 ILSE block, or a real 4 by 4 NRSE
block, including multipliers. Exact injection constraints remain in this matrix;
an unconstrained gain inverse is not substituted. The symbolic LU pattern,
including ordinary fill supplied by topology preparation, is retained.

The dedicated NRSE assembly must be finite and Hermitian to relative Frobenius
tolerance `1e-12`. Accepted assembly roundoff is canonicalized with overflow-safe
double midpoint operations before factorization, which disables pivot
perturbation. ILSE uses the ordinary factors directly, without an additional
assembly check or canonicalization; the existing selected-inverse kernel rejects
perturbed factors. For both estimators, every packed diagonal LU pivot is
reconstructed as `L U`, equilibrated by row then column maxima, and checked for
a singular-value ratio of at least `1e-12`. These are numerical checks, not a
physical observability certificate or a full-island condition estimate.

Covariance symmetry uses a separate bounded tolerance `sqrt(double_epsilon)`.
This bounds symmetry projection size; it is not a covariance forward-error or
conditioning certificate.
It does not change `round_rtol = 1000*double_epsilon`, used for negative variance,
rank and scalar validity checks. No factor residual, inverse-condition estimate
or additional diagnostic solves are computed.

The upstream block selective-inverse sweep runs after all selector solves. The
workspace's `augmented_matrix` buffer starts with NRSE's assembled matrix or
ILSE's copied LU factors. After NRSE factorization and all selector solves, the
sweep overwrites it with complete raw inverse blocks on the original filled pattern.
Physical covariance is derived separately, restoring normalization
exactly once. No dense island covariance or physical-noise response matrix is
allocated. Raw blocks, selectors and results live only for this calculation.
The UQ factorization count is zero for ILSE and one for NRSE.
Stored covariance views compare the raw state corners in double precision,
excluding the algebraically removed slack-angle row and column; positive
normalization cancels from the symmetry ratio.
The skew must be at most `sqrt(double_epsilon)` times the state-corner Frobenius
scale (twice the mean orientation scale for off-diagonal pairs). Accepted state
corners sum the two orientations in a local copy, apply any shared
virtual/reference corrections, and restore normalization and the one-half
orientation factor.
Construction scales sum the Frobenius norms of the normalized correction
terms. Reverse access returns the adjoint. Raw augmented donors remain unchanged.

For ILSE the state covariance is `V_ij = alpha B_ij[U,U]`. Nonzero voltage
magnitude variance is `Re(V_ii)/2`; proper-model angle covariance is
`Re(V_ij/(U_i conj(U_j)))/2`. Relative angles use the slack-angle difference,
including bus/slack cross covariance obtained with one augmented selector solve.
The covariance stays in the final estimator phasor frame.

For NRSE a normalized deterministic virtual angle, when present, is removed by
`C0_ij = alpha (B_ij[q,q] - w t_i t_j^T)`. Its bus can differ from the reporting
slack. Relative reporting projects out the slack-angle column using the cached
selector; coincident virtual/slack selectors share one solve. Only the known
reported slack-angle row and column are set algebraically to zero. Magnitude
and angle variances are the canonical polar diagonal entries, without an ILSE
factor of one half.

### Why the virtual-angle correction is needed

Without an absolute voltage-angle measurement, the usual magnitude and power
measurements leave a common phasor rotation undetermined. Existing NRSE adds a
deterministic virtual angle at the voltage-measured slack bus, or at the first
voltage-measured bus if necessary. It uses that bus's prepared voltage weight.
The reference remains part of the estimator; its target has no random error.

A plain inverse gain would include the covariance associated with treating that
weighted virtual target as a noisy measurement. To remove this fictitious noise,
let `Z` denote the inverse's state block, `E` select the virtual angle coordinates,
and `Wv` contain their normalized weights. With real sensor Jacobian `H`,
unnormalized prepared noise covariance `R`, and `W = alpha R^-1`, the equivalent
constrained normal gain is `G = H^T W H + E Wv E^T`. The constrained inverse
identities give `Z G Z = Z`. Thus propagation of only the real sensor errors gives

```math
C^0=Z H^T W R W H Z
   =\alpha\left(Z-TW_vT^T\right),\qquad T=ZE.
```

Exact constraints remain in the augmented system and carry no sensor noise.
This identity uses the equivalent normal equations only for explanation;
production retains the sparse augmented formulation. It computes each column
of `T` by an LU solve with a unit selector, taking the state rows of the result,
before the selected-inverse sweep overwrites the factors.

For balanced NRSE there is one virtual column. Three-phase NRSE has three,
with `Wv = w I3`, so each bus pair subtracts
`alpha * w * sum_p T_i[:,p] T_j[:,p]^T`. These rows can affect relative-phase
directions as well as the common angle; the correction retains their
deterministic effect on the estimator and removes only their noise contribution.

Reporting relative angles is a separate transformation of `C^0`: subtract the
slack phase-a angle error from every reported phase-angle error, retaining the
bus/slack cross covariance. Only the reported slack-a angle is identically zero;
slack b/c relative angles and voltage magnitudes can remain uncertain. The
virtual reference bus and reporting slack need not be the same bus.

## Validity and arithmetic

All UQ arithmetic uses ordinary rounded binary64 (`double`) products and sums,
including contractions, norms, construction scales and standard deviations.
There is no compensated summation, error-free product expansion, high/low
residue storage or wider-precision retry.

There is no blanket input/gain magnitude prescreen. Covariance construction
retains its assembly, pivot and finiteness checks. Final scalar contractions
retain their variance and total-underflow checks. These do not guarantee
arbitrary-range safety: an intermediate rounded to zero before it reaches a
tracked contraction can be indistinguishable from a structural zero.
Covariance symmetry checks use the
bounded tolerance and raw state-corner scale above. Negative-eigenvalue checks
continue to use `round_rtol` times the correction-term construction scale.
Neither check introduces a unit floor.

This arithmetic choice trades precision for runtime. Cancellation may erase a
small contribution or change a near-threshold status relative to an extended or
compensated calculation. The validity comparisons use the rounded values and
their existing thresholds directly; they do not certify the sign of an exact
real-arithmetic result. No fixed cross-platform ulp agreement is promised.

A scalar contraction retains its signed raw variance and sum of absolute terms.
With `round_rtol = 1000 * double_epsilon`, only a negative value within
`round_rtol * absolute_scale` is clipped to zero. Larger negativity or nonfinite
arithmetic is invalid and has missing sigma. Voltage variances are real by
construction, so no imaginary residual arises.
Exactly zero nominal voltage makes magnitude and angle derivatives invalid.
There is no arbitrary near-zero cutoff; first-order accuracy can be poor near
zero even when the derivative exists. A failed assembly, pivot, perturbed-factor
or inverse-finiteness check marks every requested uncertainty invalid and keeps
the estimate; no other covariance algorithm is tried. This includes ILSE factors
that the estimator had to perturb, which are rejected before any pivot check or
solve.

The independent dense test references are test-only. They exercise both concrete
block sizes, radial and meshed fill, internal block pivots, normalization,
virtual/reference selectors, exact injection constraints, request behavior
and unchanged estimates with uncertainty enabled.
