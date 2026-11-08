<!-- SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org> -->
<!-- SPDX-License-Identifier: MPL-2.0 -->

# Internal symmetric state estimation uncertainty

The optional `StateEstimationUQContext` is an internal, synchronous math-solver
interface for quantifying the uncertainty of state-estimation results. It leaves
`SolverOutput` and public calculation signatures unchanged.
Requests use island-local mathematical indices. Bus/branch bounds and branch
sides are checked. Voltage and branch rows may be unordered or repeated; result
rows follow request order.
A row selecting no quantities retains the default `not_requested` fields.
Null and empty requests perform no uncertainty work. Callers supply fresh UQ
result storage for each calculation. Results are published only after the
ordinary solve and uncertainty calculation finish; empty requests, invalid
requests and estimator failures leave the supplied result unchanged. A gain
that fails a numerical check of the uncertainty calculation does not discard
the estimate: the estimate is returned and every requested uncertainty is
reported invalid.
Voltage and branch requests are supported for both symmetries; nodal-injection
requests remain declared but fail explicitly.

## Prepared model and variance normalization

Uncertainty quantification reuses `MeasuredValues`, including sensor
aggregation, appliance status, exact zero-injection constraints, independent
modeled sensor variances, and per-unit/radian measurement conversions. Voltage
magnitudes and their standard deviations are in per unit; voltage angles and
their standard deviations are in radians.

To keep the augmented system well-scaled during factorization, ordinary state
estimation normalizes all prepared variances by a shared positive scalar
$\alpha = \texttt{variance\_normalization()}$, defined as the minimum finite,
strictly positive prepared voltage variance or complex power/current variance
sum across the island (or $1$ if no positive stochastic measurement exists).
Exact zero-injection constraints and missing or infinite-variance rows do not
participate in selecting $\alpha$. When $\alpha^{-1}$ is finite, weights scale by
multiplication; otherwise division by $\alpha$ avoids reciprocal overflow.

The two state estimators linearize the network differently:

- **Iterative Linear State Estimation (ILSE)** solves for complex nodal voltages
  $U$ using state-independent nodal admittance equations and transformed complex
  measurements. Power measurements are converted to equivalent current
  variances using the existing near-unit-voltage approximation, combining active
  and reactive variances. Voltage perturbations are modeled as proper complex
  random variables (zero pseudo-covariance), including when only voltage
  magnitudes are measured.
- **Newton-Raphson State Estimation (NRSE)** linearizes the real measurement
  equations at the converged voltage phasors $U$ in canonical polar coordinates
  $q = [\arg(U), |U|]$ per phase, regardless of whether signed voltage radii
  were used during the Newton iterations.

## Sparse covariance calculation

For each island, PGM formulates state estimation as a bus-interleaved,
variance-normalized augmented KKT system

```math
\bar{A} = \Pi \begin{bmatrix} \bar{G} & Q^H \\ Q & -\bar{R} \end{bmatrix} \Pi^T,
\qquad B = \bar{A}^{-1},
```

where $\bar{G}$ contains normalized direct voltage-measurement weights, $Q$ is
the admittance or linearized branch/injection constraint matrix, $\bar{R}$
contains normalized auxiliary measurement variances (zero for exact injection
constraints), and $\Pi$ interleaves state coordinates $q$ and auxiliary
multipliers at each bus. Each bus block is $2 \times 2$ complex (balanced) or
$6 \times 6$ complex (three-phase) for ILSE, and $4 \times 4$ real (balanced) or
$12 \times 12$ real (three-phase) for NRSE. Exact injection constraints remain
in $\bar{A}$; an unconstrained normal-equation inverse is not substituted.

ILSE and NRSE obtain the LU factorization of $\bar{A}$ as follows:

- **ILSE** reuses the estimator's state-independent LU factors and block
  permutations without reassembling or refactorizing the gain (zero UQ
  factorizations). Because selective inversion overwrites its input buffer, the
  workspace copies the factors first. If ordinary ILSE had to perturb a nearly
  singular pivot during factorization, the existing selective-inverse routine
  rejects the factors.
- **NRSE** assembles $\bar{A}$ at the converged state phasors and performs one
  unperturbed LU factorization. Before factorization, the assembled matrix must
  be finite and Hermitian within a relative Frobenius tolerance of $10^{-12}$,
  and any accepted assembly roundoff is symmetrized using overflow-safe midpoint
  averaging.

For both estimators, every diagonal LU pivot block $L_{ii} U_{ii}$ is
reconstructed, equilibrated by row and column maxima, and required to have a
singular-value ratio $\sigma_{\min} / \sigma_{\max} \ge 10^{-12}$. These local
checks guard against ill-conditioned pivot blocks during selective inversion;
they are not a global observability certificate or a condition-number estimate
for the full island matrix.

Before selective inversion overwrites the LU factors in `augmented_matrix`, the
workspace executes any unit-selector solves ($\bar{A} t = e$) needed for nonlocal
angle-reference columns. The block Takahashi selective-inverse sweep
(`inplace_selective_inverse_with_prefactorized_matrix`) then computes the raw
inverse blocks $B_{ij}$ on the filled LU sparsity pattern, without allocating a
dense island inverse.

When a state covariance block is requested for bus pair $(i, j)$, the workspace
extracts the raw state corners $B_{ij}[q,q]$ and $B_{ji}[q,q]^H$ (excluding any
algebraically zeroed slack-angle row or column) and checks their Hermitian skew:
the Frobenius norm of $B_{ij}[q,q] - B_{ji}[q,q]^H$ must not exceed
$\sqrt{\varepsilon_{\text{mach}}}$ times the state-corner Frobenius scale (twice
the mean orientation norm for off-diagonal pairs). Accepted state
corners sum the two orientations in a local copy, apply any shared
virtual/reference corrections, and restore normalization and the one-half
orientation factor. This same small double-precision state corner supplies
rectangular values. Construction scales sum the Frobenius norms of the
normalized correction terms. Reverse access returns the adjoint. Stored
augmented-inverse blocks remain unchanged.

From the symmetrized state corner $Z_{ij} = \frac{1}{2}(B_{ij}[q,q] + B_{ji}[q,q]^H)$,
physical state covariance and voltage uncertainties are obtained as follows:

- **ILSE:** The complex voltage covariance block is $V_{ij} = \alpha Z_{ij}$ in
  the estimator's phasor frame. Under the proper complex Gaussian model at a
  nonzero estimated voltage $U_i$, the voltage-magnitude variance is
  $\operatorname{Var}(|U_i|) = \frac{1}{2}\operatorname{Re}(V_{ii})$ and the
  phase-angle covariance is
  $C_{\theta,ij} = \frac{1}{2}\operatorname{Re}(V_{ij} / (U_i U_j^*))$. When
  angles are reported relative to the slack bus $s$ (phase $a$), the relative
  angle variance is $C_{\theta,ii} + C_{\theta,ss} - 2 C_{\theta,is}$, where the
  cross covariance $V_{is}$ is obtained from one augmented selector solve for the
  slack column. Because this sum could mask a negative single-term variance,
  $\operatorname{Re}(V_{ii})$ and $\operatorname{Re}(V_{ss})$ must each be
  nonnegative for the relative angle to be valid.
- **NRSE:** When a deterministic virtual angle reference is present at bus $v$
  with normalized weight matrix $W_v = w I$, its fictitious measurement noise is
  removed by $C^0_{ij} = \alpha (Z_{ij} - T_i W_v T_j^T)$. When angles are
  reported relative to the slack bus, the slack phase-$a$ angle error is
  projected out using the cached slack selector column (reusing the virtual
  column when the virtual reference is on the slack bus), and the reported
  slack phase-$a$ angle row and column are set algebraically to zero. Magnitude
  and angle variances are the canonical polar diagonal entries, without an ILSE
  factor of one half.
Rectangular covariance uses the same returned phasors.

### Why the virtual-angle correction is needed

Without an absolute voltage-angle measurement, magnitude and power measurements
leave a common phasor rotation undetermined. To make the angle reference
well-posed, NRSE inserts a deterministic virtual-angle measurement at the
slack bus if it has a voltage sensor, or else at the first voltage-measured bus
$v$, borrowing that bus's normalized voltage weight $w = \alpha / \sigma_{U,v}^2$.

Directly inverting the augmented gain treats that virtual reference as if it
were a noisy angle measurement with variance $\sigma_{U,v}^2$. To propagate only
actual sensor noise, let $Z$ denote the state block of $\bar{A}^{-1}$, $E$ select
the virtual-angle coordinates, and $W_v = w I_m$ ($m = 1$ for balanced, $3$ for
three-phase). With real sensor Jacobian $H$, unnormalized sensor noise
covariance $R$, and normalized weights $W = \alpha R^{-1}$, the equivalent
constrained normal gain is $G = H^T W H + E W_v E^T$, and the KKT inverse
identities give $Z G Z = Z$. Propagating only the actual sensor errors yields

```math
C^0=Z H^T W R W H Z
   =\alpha\left(Z-TW_vT^T\right),\qquad T=ZE.
```

Exact zero-injection constraints remain in the sparse augmented system and carry
no sensor noise. Production code never forms normal equations; it computes each
column of $T$ before selective inversion by solving $\bar{A} t^{(p)} = e_{(v,\theta_p)}$
with a unit right-hand side and retaining the state rows at each bus:

```math
C^0_{ij} = \alpha \left(Z_{ij} - w \sum_{p=0}^{m-1} T_i[:,p]\, T_j[:,p]^T\right).
```

In three-phase NRSE ($m = 3$), the three virtual rows constrain all three phases
at bus $v$ and can affect inter-phase angle differences as well as the common
rotation; subtracting $\alpha T W_v T^T$ preserves their deterministic reference
effect on the state estimate while removing their fictitious noise contribution.

Reporting angles relative to the slack bus is a separate linear transformation
of $C^0$: subtracting the slack phase-$a$ angle perturbation from every phase
angle yields

```math
C^r_{ij} = C^0_{ij} - r_i d^T - d r_j^T + c\, d d^T,
```

where $d$ selects the angle coordinates of a bus block, $r_i = C^0_{i,s} e_0$ is
the bus-to-slack-phase-$a$ cross-covariance column, and $c = C^0_{s,s}(0,0)$ is
the slack phase-$a$ angle variance in $C^0$. Only the reported slack phase-$a$
angle variance is identically zero; slack phases $b$ and $c$ and all voltage
magnitudes retain their physical uncertainty.

## Validity and arithmetic

All UQ calculations use standard IEEE-754 binary64 (`double`) arithmetic. Each
state covariance block must be finite and Hermitian within
$\sqrt{\varepsilon_{\text{mach}}}$. Voltage outputs read single diagonal
entries, so the scalar rule below decides their validity; positive
semidefiniteness of whole blocks is checked where later calculations combine
their entries.

Each scalar variance contraction records its signed `raw_variance` $v$, the sum
of term magnitudes `absolute_scale` $s = \sum_k |t_k|$, and whether any term was
nonzero in exact arithmetic:

- If a calculation is undefined (e.g., nominal voltage is zero so magnitude and
  angle derivatives are undefined, or the source covariance block is invalid),
  nonfinite, lost all nonzero terms to subnormal underflow, or falls below
  $-\texttt{round\_rtol} \cdot s$ with $\texttt{round\_rtol} = 1000\,\varepsilon_{\text{mach}}$,
  it is reported as `invalid` with `sigma = NaN`.
- If $-\texttt{round\_rtol} \cdot s \le v < 0$, negative roundoff is clipped to
  `sigma = 0.0` with status `clipped_roundoff` while preserving the signed
  `raw_variance`.
- Otherwise, the variance is `valid` with $\sigma = \sqrt{v}$.

If the island's UQ system fails an assembly, perturbed-factor, pivot, or
inverse-finiteness check (`SparseMatrixError`), the converged state estimate is
returned unchanged and every requested uncertainty is marked `invalid`.

The unit tests verify the sparse workspace and voltage uncertainty calculations
against independent dense augmented inverses across both estimators and symmetry
modes, radial and meshed topologies with fill-in, variance normalization scales,
virtual and slack angle references, exact injection constraints, request
validation, UQ numerical-failure handling,
and unchanged estimates with uncertainty enabled.

## Branch propagation

Branch requests select an island-local mathematical branch and its `from` or
`to` terminal. The actual terminal admittance row gives `I = c U`; the signed
power entering that terminal is `S = U_terminal conj(I)`. Transformer phase
shifts and terminal shunts therefore use the same coefficients as ordinary
branch results. The four terminal covariance blocks are available on the
original filled pattern. Stored physical blocks consume the workspace's
canonical covariance view. Reference blocks use the same bounded symmetry
correction before their orientation average is used. All branch outputs use
the same normalized state corner; an already canonical view is not converted
or checked again. Because a contraction sums both terminals' entries, a
negative contribution from one bus could be masked by the other. Each terminal's
diagonal physical block $D$ must therefore be positive semidefinite within
roundoff, $\lambda_{\min}(D) \ge -\texttt{round\_rtol} \cdot S_D$, where the
construction scale $S_D$ sums the Frobenius norms of the mapped correction
terms; otherwise every output using that terminal is invalid. The check runs
once per bus and call.

ILSE applies the proper complex voltage covariance as an equivalent real
rectangular covariance with its required factor of one half. Power derivatives
retain both terms `conj(I) dU_terminal + U_terminal conj(c dU)`; the resulting
power perturbation can have nonzero pseudo-covariance even though voltage
perturbations are modeled proper. NRSE transforms the canonical polar
covariance to rectangular coordinates before the same physical output maps.

For nonzero current the magnitude derivative is
`Re(conj(I) dI / abs(I))`, and the angle derivative is `Im(dI / I)`.
ILSE slack-relative current angles additionally subtract the reporting slack
angle, including the terminal/slack cross covariance. NRSE uses its already
projected relative covariance without another slack subtraction. Exactly zero
current makes magnitude and angle uncertainty invalid; signed P/Q derivatives
remain defined. Arbitrarily small nonzero current has no numerical cutoff,
although first-order accuracy can deteriorate near zero. A disconnected
requested terminal is invalid; connected terminals ignore an absent opposite
bus. These outputs remain per unit, with physical conversion performed by the
observer adapter.

Independent central finite differences check both terminal output maps, joint
dense contractions, transformer/shunt coefficients, absolute and relative
references, virtual-angle correction, and the required ILSE power
pseudo-covariance. Solver integration checks preserve estimates and previously
computed voltage results while sharing one covariance workspace and selected-inverse sweep.
