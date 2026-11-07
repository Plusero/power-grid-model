// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#include <power_grid_model/math_solver/iterative_linear_se_solver.hpp>
#include <power_grid_model/math_solver/newton_raphson_se_solver.hpp>

#include <power_grid_model/common/dummy_logging.hpp>
#include <power_grid_model/common/statistics.hpp>

#include <Eigen/LU>
#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace power_grid_model::math_solver {
namespace {
using namespace state_estimation_uq;
using ILGain = iterative_linear_se::ILSEGainBlock<symmetric_t>;
using ILUnknown = iterative_linear_se::ILSEUnknown<symmetric_t>;
using NRGain = newton_raphson_se::NRSEGainBlock<symmetric_t>;
using NRUnknown = newton_raphson_se::NRSEUnknown<symmetric_t>;

MathModelTopology topology(Idx n, std::vector<BranchIdx> branches) {
    MathModelTopology topo;
    topo.slack_bus = n - 1;
    topo.is_radial = std::ssize(branches) == n - 1;
    topo.phase_shift.resize(n, 0.0);
    topo.branch_bus_idx = std::move(branches);
    // YBus receives the topology engine's symbolic fill; reproduce elimination
    // independently here because the fixture bypasses that engine.
    std::vector<std::vector<bool>> adjacency(n, std::vector<bool>(n, false));
    for (auto const& edge : topo.branch_bus_idx) {
        adjacency[edge[0]][edge[1]] = adjacency[edge[1]][edge[0]] = true;
    }
    for (Idx pivot = 0; pivot < n; ++pivot) {
        for (Idx first = pivot + 1; first < n; ++first) {
            for (Idx second = first + 1; second < n; ++second) {
                if (adjacency[pivot][first] && adjacency[pivot][second] && !adjacency[first][second]) {
                    adjacency[first][second] = adjacency[second][first] = true;
                    topo.fill_in.push_back({first, second});
                }
            }
        }
    }
    // One source at the slack bus and one voltage sensor at bus 0.
    IdxVector no_objects(n + 1, 0), source(n + 1, 0), voltage_sensor(n + 1, 1);
    source[n] = 1;
    voltage_sensor[0] = 0;
    topo.sources_per_bus = {from_sparse, source};
    topo.shunts_per_bus = {from_sparse, no_objects};
    topo.load_gens_per_bus = {from_sparse, no_objects};
    topo.voltage_sensors_per_bus = {from_sparse, voltage_sensor};
    topo.power_sensors_per_bus = {from_sparse, no_objects};
    topo.power_sensors_per_source = {from_sparse, {0, 0}};
    topo.power_sensors_per_load_gen = {from_sparse, {0}};
    topo.power_sensors_per_shunt = {from_sparse, {0}};
    IdxVector no_branch_measurements(topo.branch_bus_idx.size() + 1, 0);
    topo.power_sensors_per_branch_from = {from_sparse, no_branch_measurements};
    topo.power_sensors_per_branch_to = {from_sparse, no_branch_measurements};
    return topo;
}

/*
Network (v: voltage sensor):

bus0(v, P=Q=0) --branch0-- bus1(slack)
                             |
                          source0

Series admittance is 1 p.u.; there are no shunts or appliances at bus0.
At nonzero voltage, S0=0 implies I0=0 and U0=U1 (no branch voltage drop).
Relative NRSE places the virtual angle at measured bus0, away from slack bus1.
*/
MathModelTopology voltage_measurement_topology() { return topology(2, {{0, 1}}); }

/*
Network (v: voltage sensor, p: branch power sensor):

 source0                           source1
    |                                 |
 bus0(v) --(p)-- branch0 ---------- bus1(v, slack)

Branch0 runs from bus0 to bus1, with series admittance 1 p.u. and no shunts.
The current sensor is at the selected from/to terminal; source powers are unmeasured.
*/
MathModelTopology noisy_measurement_topology(BranchSide side) {
    auto topo = topology(2, {{0, 1}});
    topo.sources_per_bus = {from_sparse, {0, 1, 2}};
    topo.power_sensors_per_source = {from_sparse, {0, 0, 0}};
    topo.voltage_sensors_per_bus = {from_sparse, {0, 1, 2}};
    topo.power_sensors_per_branch_from = {from_sparse, {0, 1}};
    topo.current_sensors_per_branch_from = {from_sparse, {0, side == BranchSide::from ? 1 : 0}};
    topo.current_sensors_per_branch_to = {from_sparse, {0, side == BranchSide::to ? 1 : 0}};
    return topo;
}

StateEstimationInput<symmetric_t> voltage_measurement_input(bool absolute, double variance) {
    StateEstimationInput<symmetric_t> input;
    input.source_status = {1};
    input.measured_voltage = {
        {.value = absolute ? DoubleComplex{1.0, 0.0} : DoubleComplex{1.0, nan}, .variance = variance}};
    return input;
}

StateEstimationInput<symmetric_t> noisy_measurement_input(BranchSide side, AngleMeasurementType angle_type,
                                                          double variance_scale) {
    StateEstimationInput<symmetric_t> input;
    input.source_status = {1, 1};
    input.measured_voltage = {{std::polar(1.08, 0.13), 0.7 * variance_scale},
                              {std::polar(0.94, -0.08), 1.1 * variance_scale}};
    input.measured_branch_from_power = {{.real_component = {.value = 0.2, .variance = 0.4 * variance_scale},
                                         .imag_component = {.value = -0.15, .variance = 0.9 * variance_scale}}};
    DoubleComplex const recorded_current = (side == BranchSide::from ? 1.0 : -1.0) * DoubleComplex{0.17, 0.12};
    // Match the public polar-noise fixture used for the frozen MC reference.
    PolarComplexRandVar<symmetric_t> const polar_current{
        {std::abs(recorded_current), 0.3 * variance_scale},
        {std::arg(recorded_current), 0.8 * variance_scale / std::norm(recorded_current)}};
    CurrentSensorCalcParam<symmetric_t> const current{
        .angle_measurement_type = angle_type,
        .measurement = static_cast<DecomposedComplexRandVar<symmetric_t>>(polar_current)};
    if (side == BranchSide::from) {
        input.measured_branch_from_current = {current};
    } else {
        input.measured_branch_to_current = {current};
    }
    return input;
}

struct MonteCarloVoltageReference {
    BranchSide side;
    AngleMeasurementType angle_type;
    std::array<double, 2> ilse_variance, nrse_variance;
};

// Frozen physical-noise MC sample variances (p.u.^2, ddof=1), 1,000 draws per case.
// MonteCarloWithPGM run: focused-voltage-mc-20261105T202708Z in pgm-dev-helpers/se-uq-sym/campaign/results/.
// Same banks for both solvers; seed = 20261105 + 2000 * case index. Solver tolerance 1e-10, max iterations 100.
constexpr std::array<MonteCarloVoltageReference, 4> mc_voltage_reference{{
    {BranchSide::from,
     AngleMeasurementType::global_angle,
     {4.715011241299568e-5, 5.412894050860357e-5},
     {4.412549906100932e-5, 5.3412073837558703e-5}},
    {BranchSide::from,
     AngleMeasurementType::local_angle,
     {3.970789152499769e-5, 5.007326972753122e-5},
     {3.8115910903275474e-5, 4.995655819923004e-5}},
    {BranchSide::to,
     AngleMeasurementType::global_angle,
     {4.1196687495181515e-5, 5.147876957469861e-5},
     {4.012505034031924e-5, 5.014778098808568e-5}},
    {BranchSide::to,
     AngleMeasurementType::local_angle,
     {4.185494975238301e-5, 4.9655410338141935e-5},
     {4.005185441256372e-5, 4.763437404148403e-5}},
}};

MathModelParam<symmetric_t> parameters(Idx branches) {
    MathModelParam<symmetric_t> param;
    param.branch_param.resize(branches, {1.0, -1.0, -1.0, 1.0});
    return param;
}

std::vector<BranchIdx> four_bus_branches(bool mesh) {
    return mesh ? std::vector<BranchIdx>{{0, 1}, {1, 2}, {2, 3}, {3, 0}}
                : std::vector<BranchIdx>{{0, 2}, {1, 2}, {2, 3}};
}
ComplexValueVector<symmetric_t> four_bus_voltage() {
    return {1.01, std::polar(0.99, -0.02), std::polar(1.0, -0.01), 1.02};
}

// Voltage weights and noisy bus injections produce the same augmented block
// structure as SE: [G, Q^H; Q, -R], with bus-local measurement variances.
template <class Gain, int N>
std::vector<Gain> augmented_blocks(YBus<symmetric_t> const& y_bus, MeasuredValues<symmetric_t> const& measured,
                                   ComplexValueVector<symmetric_t> const& u,
                                   ComplexValueVector<symmetric_t> const& injection, bool relative, Idx virtual_bus) {
    std::vector<Gain> blocks(y_bus.nnz_lu());
    for (Idx row = 0; row != 4; ++row) {
        auto const& measured_injection = measured.bus_injection(row);
        for (Idx index = y_bus.row_indptr_lu()[row]; index != y_bus.row_indptr_lu()[row + 1]; ++index) {
            Idx const col = y_bus.col_indices_lu()[index];
            Idx const admittance_index = y_bus.map_lu_y_bus()[index];
            if (admittance_index < 0) {
                continue;
            }
            auto& block = blocks[index];
            auto const y = y_bus.admittance()[admittance_index];
            if constexpr (N == 1) {
                block.q() = y;
                if (row == col) {
                    if (measured.has_voltage(row)) {
                        block.g() = 1.0 / measured.voltage_var(row);
                    }
                    block.r() =
                        -(measured_injection.real_component.variance + measured_injection.imag_component.variance);
                }
            } else {
                // Analytic derivatives of S_i = U_i conj(sum_j Y_ij U_j)
                // in the estimator's [theta, magnitude] coordinates.
                DoubleComplex const cross = u[row] * std::conj(y * u[col]);
                DoubleComplex d_theta = -1.0i * cross;
                DoubleComplex d_magnitude = cross / std::abs(u[col]);
                if (row == col) {
                    d_theta += 1.0i * injection[row];
                    d_magnitude += injection[row] / std::abs(u[row]);
                    if (measured.has_voltage(row)) {
                        block(1, 1) = 1.0 / measured.voltage_var(row);
                        if (!relative || row == virtual_bus) {
                            block(0, 0) = block(1, 1);
                        }
                    }
                    block(2, 2) = -measured_injection.real_component.variance;
                    block(3, 3) = -measured_injection.imag_component.variance;
                }
                block.matrix().template bottomLeftCorner<2, 2>() << d_theta.real(), d_magnitude.real(), d_theta.imag(),
                    d_magnitude.imag();
            }
        }
    }
    for (Idx index = 0; index != y_bus.nnz_lu(); ++index) {
        blocks[index].matrix().template topRightCorner<N, N>() =
            blocks[y_bus.lu_transpose_entry()[index]].matrix().template bottomLeftCorner<N, N>().adjoint();
    }
    return blocks;
}

template <class Gain, class Unknown, int N>
void dense_workspace_reference(bool mesh, bool relative, Idx virtual_bus, double alpha) {
    constexpr int size = Gain::RowsAtCompileTime;
    using Scalar = typename Gain::Scalar;
    using Dense = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    auto topo = topology(4, four_bus_branches(mesh));
    topo.load_gens_per_bus = {from_sparse, {0, 1, 2, 3, 3}};
    topo.load_gen_type.assign(3, LoadGenType::const_pq);
    topo.power_sensors_per_load_gen = {from_sparse, {0, 0, 0, 0}};
    topo.power_sensors_per_bus = {from_sparse, {0, 1, 2, 3, 4}};
    IdxVector voltage_sensors(5, 0);
    for (Idx bus = 0; bus != 4; ++bus) {
        voltage_sensors[bus + 1] = voltage_sensors[bus] + Idx{!(virtual_bus == 0 && bus == topo.slack_bus)};
    }
    topo.voltage_sensors_per_bus = {from_sparse, voltage_sensors};
    YBus<symmetric_t> const y_bus{topo, parameters(topo.n_branch())};
    ComplexValueVector<symmetric_t> const u = four_bus_voltage();
    auto const injection = y_bus.calculate_injection(u);
    StateEstimationInput<symmetric_t> input;
    input.source_status = {1};
    input.load_gen_status = {1, 1, 1};
    for (Idx row = 0; row != 4; ++row) {
        if (voltage_sensors[row] != voltage_sensors[row + 1]) {
            input.measured_voltage.push_back({relative ? DoubleComplex{std::abs(u[row]), nan} : u[row], alpha});
        }
        input.measured_bus_injection.push_back(
            {.real_component = {.value = injection[row].real(), .variance = 0.5 * alpha},
             .imag_component = {.value = injection[row].imag(), .variance = 0.5 * alpha}});
    }
    MeasuredValues<symmetric_t> const measured{topo, input};
    CHECK_NOTHROW(observability::observability_check(measured, topo, y_bus.y_bus_structure()));
    CHECK(measured.variance_normalization() == alpha);
    auto const blocks = augmented_blocks<Gain, N>(y_bus, measured, u, injection, relative, virtual_bus);
    Dense dense = Dense::Zero(4 * size, 4 * size);
    for (Idx row = 0; row != 4; ++row) {
        for (Idx index = y_bus.row_indptr_lu()[row]; index != y_bus.row_indptr_lu()[row + 1]; ++index) {
            dense.block(row * size, y_bus.col_indices_lu()[index] * size, size, size) = blocks[index].matrix();
        }
    }
    double const weight = virtual_bus >= 0 ? 1.0 / measured.voltage_var(virtual_bus) : 0.0;
    SparseCovarianceWorkspace<Gain, Unknown, N> const workspace{y_bus,     blocks,      alpha, u,
                                                                !relative, virtual_bus, weight};
    Dense const inverse = dense.fullPivLu().inverse();
    CHECK((dense * inverse - Dense::Identity(4 * size, 4 * size)).norm() < 1e-12);
    PhysicalBlocks cache{workspace};
    for (Idx row = 0; row != 4; ++row) {
        for (Idx index = y_bus.row_indptr_lu()[row]; index != y_bus.row_indptr_lu()[row + 1]; ++index) {
            Idx const col = y_bus.col_indices_lu()[index];
            CHECK((workspace.augmented_matrix[index].matrix() - inverse.block(row * size, col * size, size, size))
                      .norm() < 1e-12);
            Eigen::Matrix<Scalar, N, N> expected = alpha * inverse.block(row * size, col * size, N, N);
            if constexpr (N == 2) {
                auto physical = [&](Idx a, Idx b) -> Eigen::Matrix2d {
                    Eigen::Matrix2d block = alpha * inverse.block(a * size, b * size, 2, 2);
                    if (virtual_bus >= 0) {
                        block -= alpha * weight * inverse.block(a * size, virtual_bus * size, 2, 1) *
                                 inverse.block(virtual_bus * size, b * size, 1, 2);
                    }
                    return block;
                };
                expected = physical(row, col);
                if (relative) {
                    Eigen::Vector2d const d{1, 0};
                    expected -= physical(row, 3).col(0) * d.transpose();
                    expected -= d * physical(3, col).row(0);
                    expected += physical(3, 3)(0, 0) * d * d.transpose();
                }
            }
            auto const actual = workspace.calculate_state_covariance_block(row, col);
            REQUIRE(actual.valid);
            CHECK((actual.value - expected).norm() < 1e-12 * alpha);
            if constexpr (N == 2) {
                Eigen::Matrix2d const left = workspace.rectangular_jacobian(row);
                Eigen::Matrix2d const right = workspace.rectangular_jacobian(col);
                Eigen::Matrix2d const rectangular = left * expected * right.transpose();
                auto const mapped = workspace.physical_block(row, col);
                CHECK((mapped.value - rectangular).norm() < 1e-12 * alpha);
                CHECK(std::isfinite(mapped.construction_scale));
                if (row <= col) {
                    auto const& stored = cache.stored(index, row, col);
                    CHECK(&cache.stored(index, row, col) == &stored);
                    CHECK((stored.value.array() == mapped.value.array()).all());
                    CHECK(stored.construction_scale == mapped.construction_scale);
                    CHECK(stored.valid == mapped.valid);
                }
                // Equivalent signed-radius coordinates require changing both
                // covariance sides and the radial Jacobian column together.
                Eigen::Matrix2d const sign = (Eigen::Vector2d{1, -1}).asDiagonal();
                CHECK(((left * sign) * (sign * expected * sign) * (right * sign).transpose() - rectangular).norm() <
                      1e-12 * alpha);
            }
        }
        if constexpr (N == 1) {
            if (relative) {
                CHECK(std::abs(workspace.variance_normalization * workspace.slack_column[row](0) -
                               alpha * inverse(row * size, 3 * size)) < 1e-12 * alpha);
            }
        }
    }
}
} // namespace

TEST_CASE("UQ concrete augmented gains agree with independent dense inverse") {
    for (bool const mesh : {false, true}) {
        for (bool const relative : {false, true}) {
            for (double const alpha : {1e-8, 1.0, 1e8}) {
                dense_workspace_reference<ILGain, ILUnknown, 1>(mesh, relative, -1, alpha);
                // NRSE has a virtual angle bus only without an angle measurement: off and on the slack bus 3.
                for (Idx const virtual_bus : relative ? IdxVector{0, 3} : IdxVector{-1}) {
                    dense_workspace_reference<NRGain, NRUnknown, 2>(mesh, relative, virtual_bus, alpha);
                }
            }
        }
    }
}

namespace {
// Synthetic four-bus workspace for physical-map and covariance-completion
// regressions. Its radial tree 0-1, 1-2, 1-3 leaves the pair (0, 2) outside the
// filled pattern, and its diagonal blocks force internal complete pivots.
std::vector<BranchIdx> synthetic_branches(bool mesh) {
    return mesh ? std::vector<BranchIdx>{{0, 1}, {1, 2}, {2, 3}, {3, 0}}
                : std::vector<BranchIdx>{{0, 1}, {1, 2}, {1, 3}};
}
ComplexValueVector<symmetric_t> synthetic_voltage() {
    return {std::polar(1.1, 0.1), std::polar(0.9, -0.2), std::polar(1.2, 0.3), 1.0};
}
template <class Gain, int N> std::vector<Gain> synthetic_blocks(YBus<symmetric_t> const& y_bus) {
    using Scalar = typename Gain::Scalar;
    std::vector<Gain> blocks(y_bus.nnz_lu());
    for (Idx row = 0; row != 4; ++row) {
        for (Idx index = y_bus.row_indptr_lu()[row]; index != y_bus.row_indptr_lu()[row + 1]; ++index) {
            Idx const col = y_bus.col_indices_lu()[index];
            Gain block;
            if (row == col) {
                // State/multiplier coupling forces nontrivial internal complete pivots.
                for (int k = 0; k != N; ++k) {
                    block(k, k) = 2.0 + row + k;
                    block(k + N, k + N) = -0.01;
                    block(k, k + N) = block(k + N, k) = 8.0 + k;
                }
            } else if (y_bus.map_lu_y_bus()[index] >= 0) {
                for (int k = 0; k != Gain::RowsAtCompileTime; ++k) {
                    block(k, k) = -0.15;
                }
                if constexpr (N == 1) {
                    block(0, 0) += Scalar{0.0, row < col ? 0.07 : -0.07};
                }
            }
            blocks[index] = block;
        }
    }
    return blocks;
}

template <class Gain, class Unknown, int N> struct WorkspaceCase {
    using Workspace = SparseCovarianceWorkspace<Gain, Unknown, N>;
    MathModelTopology topo;
    YBus<symmetric_t> y_bus;
    Workspace covariance;

    WorkspaceCase(bool mesh, bool relative, Idx virtual_bus, double alpha,
                  ComplexValueVector<symmetric_t> const& voltage = synthetic_voltage())
        : topo{topology(4, synthetic_branches(mesh))},
          y_bus{topo, parameters(topo.n_branch())},
          covariance{y_bus,       synthetic_blocks<Gain, N>(y_bus), alpha, voltage, !relative,
                     virtual_bus, virtual_bus >= 0 ? 0.3 : 0.0} {}
    WorkspaceCase(WorkspaceCase const&) = delete;
    WorkspaceCase& operator=(WorkspaceCase const&) = delete;
    WorkspaceCase(WorkspaceCase&&) = delete;
    WorkspaceCase& operator=(WorkspaceCase&&) = delete;
    ~WorkspaceCase() = default;
};
} // namespace

TEST_CASE("UQ physical map preserves an exactly cancelled corrected state") {
    using Fixture = WorkspaceCase<NRGain, NRUnknown, 2>;
    Fixture fixture{false, false, -1, 1.0, {1.1, 1.2, 1.0, 1.0}};
    Fixture::Workspace::StateCovarianceBlock state;
    state.terms[0](0, 0) = 3.0;
    state.terms[2](0, 0) = -1.0;
    state.terms[3](0, 0) = -2.0;
    auto const block = fixture.covariance.physical_from_state(0, 1, state);
    CHECK(block.valid);
    CHECK(block.value.isZero(0));
    CHECK(block.construction_scale == doctest::Approx(6.0 * 1.1 * 1.2));
}

TEST_CASE("UQ variance status thresholds and validity") {
    double const tolerance = round_rtol;
    CHECK(finalize_variance(-tolerance, 1.0, true).raw_status == UQVarianceStatus::clipped_roundoff);
    CHECK(finalize_variance(std::nextafter(-tolerance, -1.0), 1.0, true).raw_status == UQVarianceStatus::invalid);
    CHECK(variance({1.0, -1.0 - 1e-14}).raw_status == UQVarianceStatus::clipped_roundoff);
    CHECK(variance({1.0, -2.0}).raw_status == UQVarianceStatus::invalid);
    CHECK(variance({-1e-20}).raw_status == UQVarianceStatus::invalid);
    auto const zero = variance({0.0});
    CHECK(zero.raw_status == UQVarianceStatus::valid);
    CHECK(zero.sigma == 0.0);
    // Rounding cancellation of exactly nonzero terms at a normal scale is a valid zero, not underflow.
    auto const cancelled = variance({0x1p53, 1.0, -0x1p53});
    CHECK(cancelled.raw_variance == 0.0);
    CHECK(cancelled.raw_status == UQVarianceStatus::valid);
    CHECK(variance({1.0}, false).raw_status == UQVarianceStatus::invalid);
    CHECK(variance({nan}).raw_status == UQVarianceStatus::invalid);
    CHECK(variance({std::numeric_limits<double>::infinity()}).raw_status == UQVarianceStatus::invalid);
}

TEST_CASE("UQ relative ILSE angles reject a negative single-term variance inside the relative sum") {
    // Three buses in a line, slack bus 2; the diagonal gain makes every V_ii = 1 and V_is = 0.
    auto const topo = topology(3, {{0, 1}, {1, 2}});
    YBus<symmetric_t> const y_bus{topo, parameters(2)};
    std::vector<ILGain> gain(y_bus.nnz_lu());
    for (auto const d : y_bus.lu_diag()) {
        gain[d](0, 0) = 1.0;
        gain[d](1, 1) = -1.0;
    }
    StateEstimationUQRequest request;
    request.voltage = {{0, false, true}};
    for (Idx const negative_bus : {Idx{2}, Idx{0}}) {
        SparseCovarianceWorkspace<ILGain, ILUnknown, 1> workspace{y_bus, gain, 1.0, {1.0, 1.0, 1.0}, false};
        REQUIRE(workspace.report_relative_angle);
        // A small negative V_ss (slack) or V_00 would be masked by the other positive term.
        workspace.augmented_matrix[y_bus.lu_diag()[negative_bus]](0, 0) = -1e-8;
        auto const result = calculate_voltage_uq_results(workspace, request);
        CHECK(result.voltage[0].angle.phase[0].raw_status == UQVarianceStatus::invalid);
    }
}

TEST_CASE_TEMPLATE("UQ voltage integration and estimate invariance", Solver, IterativeLinearSESolver<symmetric_t>,
                   NewtonRaphsonSESolver<symmetric_t>) {
    constexpr double variance_scale = 1e-4;
    SUBCASE("Voltage measurements and exact constraints") {
        auto const topo = voltage_measurement_topology();
        YBus<symmetric_t> const y_bus{topo, parameters(1)};
        Solver solver{y_bus, topo};
        common::logging::NoLogger log;
        for (bool const absolute : {false, true}) {
            StateEstimationUQRequest request;
            request.voltage = {{0, true, true}, {1, false, true}};
            StateEstimationUQResult<symmetric_t> result;
            StateEstimationUQContext<symmetric_t> context{request, result};
            auto const input = voltage_measurement_input(absolute, variance_scale);
            auto const ordinary = solver.run_state_estimation(y_bus, input, 1e-12, 30, log);
            auto const enabled = solver.run_state_estimation(y_bus, input, 1e-12, 30, log, &context);
            CHECK(enabled.u == ordinary.u);
            REQUIRE(result.voltage.size() == 2);
            auto const factor = Solver::is_NRSE_solver ? 1.0 : 0.5;
            CHECK(result.voltage[0].magnitude.phase[0].raw_variance ==
                  doctest::Approx(variance_scale * factor).epsilon(1e-12));
            CHECK(result.voltage[0].magnitude.phase[0].raw_status == UQVarianceStatus::valid);
            CHECK(result.voltage[1].magnitude.phase[0].raw_status == UQVarianceStatus::not_requested);
            if (absolute) {
                CHECK(result.voltage[0].angle.phase[0].raw_variance ==
                      doctest::Approx(variance_scale * factor).epsilon(1e-12));
            } else {
                CHECK(result.voltage[1].angle.phase[0].raw_variance == 0.0);
            }
            StateEstimationUQRequest reordered_request;
            reordered_request.voltage = {{1, false, true}, {0, true, true}, {0, true, true}, {0, false, false}};
            StateEstimationUQResult<symmetric_t> reordered_result;
            StateEstimationUQContext<symmetric_t> reordered_context{reordered_request, reordered_result};
            auto const reordered = solver.run_state_estimation(y_bus, input, 1e-12, 30, log, &reordered_context);
            CHECK(reordered.u == ordinary.u);
            REQUIRE(reordered_result.voltage.size() == 4);
            CHECK(reordered_result.voltage[0].angle.phase[0].raw_variance ==
                  result.voltage[1].angle.phase[0].raw_variance);
            for (std::size_t row = 1; row != 3; ++row) {
                CHECK(reordered_result.voltage[row].magnitude.phase[0].raw_variance ==
                      result.voltage[0].magnitude.phase[0].raw_variance);
                CHECK(reordered_result.voltage[row].angle.phase[0].raw_variance ==
                      result.voltage[0].angle.phase[0].raw_variance);
            }
            CHECK(reordered_result.voltage[3].magnitude.phase[0].raw_status == UQVarianceStatus::not_requested);
            CHECK(reordered_result.voltage[3].angle.phase[0].raw_status == UQVarianceStatus::not_requested);
            // Adding branch rows shares the covariance workspace and leaves the voltage rows unchanged.
            auto const voltage_only = result.voltage;
            StateEstimationUQRequest branch_request = request;
            branch_request.branch = {{0, BranchSide::from, true, true, true, true},
                                     {0, BranchSide::to, true, true, true, true}};
            StateEstimationUQResult<symmetric_t> branch_result;
            StateEstimationUQContext<symmetric_t> branch_context{branch_request, branch_result};
            auto const branch_enabled = solver.run_state_estimation(y_bus, input, 1e-12, 30, log, &branch_context);
            CHECK(branch_enabled.u == ordinary.u);
            REQUIRE(branch_result.branch.size() == 2);
            CHECK(branch_result.voltage[0].magnitude.phase[0].raw_variance ==
                  voltage_only[0].magnitude.phase[0].raw_variance);
            CHECK(branch_result.voltage[0].angle.phase[0].raw_variance == voltage_only[0].angle.phase[0].raw_variance);
        }
        // An empty request must skip UQ work and leave the fresh result empty.
        auto const input = voltage_measurement_input(true, variance_scale);
        auto const ordinary = solver.run_state_estimation(y_bus, input, 1e-12, 30, log);
        StateEstimationUQRequest const empty;
        StateEstimationUQResult<symmetric_t> empty_result;
        StateEstimationUQContext<symmetric_t> empty_context{empty, empty_result};
        auto const bypass = solver.run_state_estimation(y_bus, input, 1e-12, 30, log, &empty_context);
        CHECK(bypass.u == ordinary.u);
        CHECK(empty_result.voltage.empty());
    }
    SUBCASE("Noisy current and power measurements") {
        // Frozen MC reference: at most 100% relative deviation in magnitude variance.
        // Angle uncertainties are checked for validity only.
        for (auto const& reference : mc_voltage_reference) {
            CAPTURE(reference.side);
            CAPTURE(reference.angle_type);
            auto const topo = noisy_measurement_topology(reference.side);
            YBus<symmetric_t> const y_bus{topo, parameters(1)};
            auto const input = noisy_measurement_input(reference.side, reference.angle_type, variance_scale);
            MeasuredValues<symmetric_t> const measured{topo, input};
            // The voltage channel's 0.7*scale is below the complex-channel variance sums.
            CHECK(measured.variance_normalization() / variance_scale == doctest::Approx(0.7));
            Solver solver{y_bus, topo};
            common::logging::NoLogger log;
            auto const ordinary = solver.run_state_estimation(y_bus, input, 1e-10, 100, log);
            StateEstimationUQRequest request;
            request.voltage = {{0, true, true}, {1, true, true}};
            request.branch = {{0, BranchSide::from, true, true, true, true},
                              {0, BranchSide::to, true, true, true, true}};
            StateEstimationUQResult<symmetric_t> result;
            StateEstimationUQContext<symmetric_t> context{request, result};
            auto const enabled = solver.run_state_estimation(y_bus, input, 1e-10, 100, log, &context);
            CHECK(enabled.u == ordinary.u);
            REQUIRE(result.voltage.size() == 2);
            REQUIRE(result.branch.size() == 2);
            for (auto const& branch : result.branch) {
                for (auto const* value :
                     {&branch.active_power, &branch.reactive_power, &branch.current_magnitude, &branch.current_angle}) {
                    CHECK(value->phase[0].raw_status == UQVarianceStatus::valid);
                }
            }
            auto const& mc_variance = Solver::is_NRSE_solver ? reference.nrse_variance : reference.ilse_variance;
            for (std::size_t bus = 0; bus != result.voltage.size(); ++bus) {
                auto const& voltage = result.voltage[bus];
                CHECK(voltage.angle.phase[0].raw_status == UQVarianceStatus::valid);
                CHECK(voltage.magnitude.phase[0].raw_status == UQVarianceStatus::valid);
                double const relative_deviation =
                    std::abs(voltage.magnitude.phase[0].raw_variance - mc_variance[bus]) / mc_variance[bus];
                CHECK(relative_deviation <= 1.0);
            }
        }
    }
}

TEST_CASE_TEMPLATE("UQ request bounds and unsupported injection requests", Sym, symmetric_t, asymmetric_t) {
    StateEstimationUQRequest request;
    StateEstimationUQResult<Sym> result;
    StateEstimationUQContext<Sym> context{request, result};
    CHECK_NOTHROW(validate_context<Sym>(nullptr, 2, 1));
    for (Idx const bus : {-1, 2}) {
        request.voltage = {{bus, true, false}};
        CHECK_THROWS_AS(validate_context(&context, 2, 1), InvalidArguments);
    }
    request = {};
    for (Idx const branch : {-1, 1}) {
        request.branch = {{branch, BranchSide::from, true, false, false, false}};
        CHECK_THROWS_AS(validate_context(&context, 2, 1), InvalidArguments);
    }
    request.branch = {{0, static_cast<BranchSide>(2), true, false, false, false}};
    CHECK_THROWS_AS(validate_context(&context, 2, 1), InvalidArguments);
    request = {};
    request.injection = {{0, true, false}};
    CHECK_THROWS_WITH_AS(validate_context(&context, 2, 1), "Injection uncertainty is not implemented",
                         InvalidArguments);
    request = {};
    CHECK_NOTHROW(validate_context(&context, 2, 1));
}

TEST_CASE_TEMPLATE("UQ numerical failure keeps the estimate and reports requested uncertainty invalid", Solver,
                   IterativeLinearSESolver<symmetric_t>, NewtonRaphsonSESolver<symmetric_t>) {
    // An unmeasured source at bus 0 and voltage sensor + exact zero injection at bus 1 leaves the
    // bus-0 pivot singular without perturbation. Ordinary SE perturbs the pivot and converges; the
    // selective inverse rejects perturbed ILSE factors, and NRSE's unperturbed UQ factorization fails.
    auto topo = voltage_measurement_topology();
    topo.sources_per_bus = {from_sparse, {0, 1, 1}};
    topo.voltage_sensors_per_bus = {from_sparse, {0, 0, 1}};
    YBus<symmetric_t> const y_bus{topo, parameters(1)};
    Solver solver{y_bus, topo};
    common::logging::NoLogger log;
    auto const input = voltage_measurement_input(true, 1e-4);
    auto const ordinary = solver.run_state_estimation(y_bus, input, 1e-12, 30, log);
    StateEstimationUQRequest request;
    request.voltage = {{0, true, true}, {1, false, false}};
    request.branch = {{0, BranchSide::to, true, false, true, false}};
    StateEstimationUQResult<symmetric_t> result;
    StateEstimationUQContext<symmetric_t> context{request, result};
    auto const enabled = solver.run_state_estimation(y_bus, input, 1e-12, 30, log, &context);
    CHECK(enabled.u == ordinary.u);
    REQUIRE(result.voltage.size() == 2);
    REQUIRE(result.branch.size() == 1);
    for (auto const* value : {&result.voltage[0].magnitude.phase[0], &result.voltage[0].angle.phase[0]}) {
        CHECK(value->raw_status == UQVarianceStatus::invalid);
        CHECK(std::isnan(value->sigma));
        // The failed workspace computes no raw variance.
        CHECK(std::isnan(value->raw_variance));
    }
    CHECK(result.voltage[1].magnitude.phase[0].raw_status == UQVarianceStatus::not_requested);
    CHECK(result.voltage[1].angle.phase[0].raw_status == UQVarianceStatus::not_requested);
    auto const& branch = result.branch[0];
    for (auto const* value : {&branch.active_power.phase[0], &branch.current_magnitude.phase[0]}) {
        CHECK(value->raw_status == UQVarianceStatus::invalid);
        CHECK(std::isnan(value->sigma));
        CHECK(std::isnan(value->raw_variance));
    }
    CHECK(branch.reactive_power.phase[0].raw_status == UQVarianceStatus::not_requested);
    CHECK(branch.current_angle.phase[0].raw_status == UQVarianceStatus::not_requested);
}

} // namespace power_grid_model::math_solver
