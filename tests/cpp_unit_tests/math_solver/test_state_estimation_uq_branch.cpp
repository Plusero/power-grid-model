// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#include <power_grid_model/math_solver/iterative_linear_se_solver.hpp>
#include <power_grid_model/math_solver/newton_raphson_se_solver.hpp>
#include <power_grid_model/math_solver/state_estimation_uq_branch.hpp>

#include <Eigen/LU>
#include <doctest/doctest.h>

namespace power_grid_model::math_solver {
namespace {
using namespace state_estimation_uq;
using ILGain = iterative_linear_se::ILSEGainBlock<symmetric_t>;
using ILUnknown = iterative_linear_se::ILSEUnknown<symmetric_t>;
using NRGain = newton_raphson_se::NRSEGainBlock<symmetric_t>;
using NRUnknown = newton_raphson_se::NRSEUnknown<symmetric_t>;

MathModelTopology branch_topology() {
    MathModelTopology topo;
    topo.slack_bus = 2;
    topo.phase_shift.resize(3, 0.0);
    topo.branch_bus_idx = {{0, 1}, {0, 2}, {1, 2}};
    topo.shunts_per_bus = {from_sparse, {0, 0, 0, 0}};
    return topo;
}

std::array<double, 4> evaluate(ComplexValueVector<symmetric_t> const& u, BranchCalcParam<symmetric_t> const& p,
                               int side, bool relative) {
    DoubleComplex const current = side == 0 ? p.yff() * u[0] + p.yft() * u[1] : p.ytf() * u[0] + p.ytt() * u[1];
    auto const power = u[side] * std::conj(current);
    return {power.real(), power.imag(), std::abs(current), std::arg(current) - (relative ? std::arg(u[2]) : 0)};
}

template <class Gain, class Unknown, int N> void reference(bool relative, bool virtual_angle = false) {
    using Scalar = typename Gain::Scalar;
    constexpr int size = Gain::RowsAtCompileTime;
    auto topo = branch_topology();
    MathModelParam<symmetric_t> parameters;
    // Deliberately nonsymmetric transformer/shunt row, both terminal currents.
    parameters.branch_param.resize(
        3, {DoubleComplex{2, -3}, DoubleComplex{-1, 2}, DoubleComplex{-1.2, 1.8}, DoubleComplex{1.7, -2.6}});
    YBus<symmetric_t> const y_bus{topo, parameters};
    Eigen::Matrix<Scalar, 3 * N, 3 * N> c;
    for (int i = 0; i != 3 * N; ++i) {
        for (int j = 0; j != 3 * N; ++j) {
            c(i, j) = i == j ? 1.0 + 0.1 * i : 0.03 * (i + j + 1);
            if constexpr (N == 1) {
                if (i != j) {
                    c(i, j) += Scalar{0, 0.02 * (i - j)};
                }
            }
        }
    }
    auto const precision = c.inverse().eval();
    std::vector<Gain> gain(y_bus.nnz_lu());
    for (Idx i = 0; i != 3; ++i) {
        for (Idx k = y_bus.row_indptr_lu()[i]; k != y_bus.row_indptr_lu()[i + 1]; ++k) {
            Idx const j = y_bus.col_indices_lu()[k];
            gain[k].matrix().template topLeftCorner<N, N>() = precision.template block<N, N>(i * N, j * N);
            if (i == j) {
                for (int a = N; a != size; ++a) {
                    gain[k](a, a) = -1;
                }
            }
        }
    }
    ComplexValueVector<symmetric_t> u{std::polar(1.2, 0.2), std::polar(0.8, -0.1), std::polar(1.1, 0.3)};
    SparseCovarianceWorkspace<Gain, Unknown, N> workspace{
        y_bus, gain, 1.0, u, !relative, virtual_angle ? 0 : -1, virtual_angle ? 0.2 : 0.0};
    Eigen::Matrix<double, 6, 6> dense;
    for (int i = 0; i != 3; ++i) {
        for (int j = 0; j != 3; ++j) {
            if constexpr (N == 1) {
                auto v = c(i, j);
                dense.template block<2, 2>(2 * i, 2 * j) << v.real() / 2, -v.imag() / 2, v.imag() / 2, v.real() / 2;
            } else {
                dense.template block<2, 2>(2 * i, 2 * j) = c.template block<2, 2>(2 * i, 2 * j);
                if (virtual_angle) {
                    dense.template block<2, 2>(2 * i, 2 * j) -=
                        0.2 * c.template block<2, 1>(2 * i, 0) * c.template block<1, 2>(0, 2 * j);
                }
            }
        }
    }
    for (int side = 0; side != 2; ++side) {
        StateEstimationUQRequest request;
        request.branch = {{0, side == 0 ? BranchSide::from : BranchSide::to, true, true, true, true}};
        StateEstimationUQResult<symmetric_t> result;
        PhysicalBlocks result_blocks{workspace};
        branch_results(workspace, y_bus, request, result, result_blocks);
        auto const& out = result.branch[0];
        std::array<UQPhaseValue, 4> const values{out.active_power.phase[0], out.reactive_power.phase[0],
                                                 out.current_magnitude.phase[0], out.current_angle.phase[0]};
        Eigen::Matrix<double, 4, 6> jacobian;
        double constexpr step = 1e-6;
        for (int bus = 0; bus != 3; ++bus) {
            for (int axis = 0; axis != 2; ++axis) {
                auto plus = u, minus = u;
                if constexpr (N == 1) {
                    auto delta = axis == 0 ? DoubleComplex{step, 0} : DoubleComplex{0, step};
                    plus[bus] += delta;
                    minus[bus] -= delta;
                } else {
                    plus[bus] = std::polar(std::abs(u[bus]) + (axis == 1 ? step : 0),
                                           std::arg(u[bus]) + (axis == 0 ? step : 0));
                    minus[bus] = std::polar(std::abs(u[bus]) - (axis == 1 ? step : 0),
                                            std::arg(u[bus]) - (axis == 0 ? step : 0));
                }
                auto const fp = evaluate(plus, parameters.branch_param[0], side, relative);
                auto const fm = evaluate(minus, parameters.branch_param[0], side, relative);
                for (int a = 0; a != 4; ++a) {
                    jacobian(a, 2 * bus + axis) = (fp[a] - fm[a]) / (2 * step);
                }
            }
        }
        Eigen::Matrix4d const expected = jacobian * dense * jacobian.transpose();
        for (int a = 0; a != 4; ++a) {
            CHECK(values[a].raw_status == UQVarianceStatus::valid);
            CHECK(values[a].raw_variance == doctest::Approx(expected(a, a)).epsilon(2e-8));
        }
        if constexpr (N == 1) {
            // Independent widely-linear power covariance and pseudo-covariance.
            auto const& p = parameters.branch_param[0];
            std::array<DoubleComplex, 2> const row = side == 0 ? std::array<DoubleComplex, 2>{p.yff(), p.yft()}
                                                               : std::array<DoubleComplex, 2>{p.ytf(), p.ytt()};
            auto const current = row[0] * u[0] + row[1] * u[1];
            std::array<DoubleComplex, 2> a{}, b{};
            a[side] = std::conj(current);
            for (int i = 0; i != 2; ++i) {
                b[i] = u[side] * std::conj(row[i]);
            }
            DoubleComplex direct{}, pseudo{};
            for (int i = 0; i != 2; ++i) {
                for (int j = 0; j != 2; ++j) {
                    direct += a[i] * c(i, j) * std::conj(a[j]) + b[i] * std::conj(c(i, j)) * std::conj(b[j]);
                    pseudo += a[i] * c(i, j) * b[j] + b[i] * c(j, i) * a[j];
                }
            }
            CHECK(values[0].raw_variance == doctest::Approx((direct.real() + pseudo.real()) / 2).epsilon(1e-12));
            CHECK(values[1].raw_variance == doctest::Approx((direct.real() - pseudo.real()) / 2).epsilon(1e-12));
            CHECK(std::abs(pseudo.real()) > 0.01);
        }
        // Fresh result storage leaves unrequested fields at their default state.
        request.branch[0] = {0, BranchSide::from, false, false, true, false};
        StateEstimationUQResult<symmetric_t> selected;
        PhysicalBlocks selected_blocks{workspace};
        branch_results(workspace, y_bus, request, selected, selected_blocks);
        CHECK(selected.branch[0].active_power.phase[0].raw_status == UQVarianceStatus::not_requested);
    }
}
} // namespace

TEST_CASE("Branch UQ agrees with independent dense finite-difference contractions") {
    for (bool relative : {false, true}) {
        reference<ILGain, ILUnknown, 1>(relative);
        // NRSE has a virtual angle exactly when it reports relative angles.
        reference<NRGain, NRUnknown, 2>(relative, relative);
    }
}

TEST_CASE("Branch UQ zero current keeps signed powers valid") {
    auto topo = branch_topology();
    MathModelParam<symmetric_t> params;
    params.branch_param.resize(3, {1., -1., -1., 1.});
    YBus<symmetric_t> const y_bus{topo, params};
    std::vector<ILGain> gain(y_bus.nnz_lu());
    for (auto d : y_bus.lu_diag()) {
        gain[d](0, 0) = 1;
        gain[d](1, 1) = -1;
    }
    SparseCovarianceWorkspace<ILGain, ILUnknown, 1> workspace{y_bus, gain, 1., {1., 1., 1.}, true};
    StateEstimationUQRequest request;
    request.branch = {{0, BranchSide::from, true, true, true, true}};
    StateEstimationUQResult<symmetric_t> result;
    PhysicalBlocks result_blocks{workspace};
    branch_results(workspace, y_bus, request, result, result_blocks);
    CHECK(result.branch[0].active_power.phase[0].raw_status == UQVarianceStatus::valid);
    CHECK(result.branch[0].reactive_power.phase[0].raw_status == UQVarianceStatus::valid);
    CHECK(result.branch[0].current_magnitude.phase[0].raw_status == UQVarianceStatus::invalid);
    CHECK(result.branch[0].current_angle.phase[0].raw_status == UQVarianceStatus::invalid);
    // Failed covariance must never become a successful zero variance.
    workspace.augmented_matrix[y_bus.lu_diag()[0]](0, 0) = nan;
    StateEstimationUQResult<symmetric_t> failed;
    PhysicalBlocks failed_blocks{workspace};
    branch_results(workspace, y_bus, request, failed, failed_blocks);
    CHECK(failed.branch[0].active_power.phase[0].raw_status == UQVarianceStatus::invalid);
    workspace.augmented_matrix[y_bus.lu_diag()[0]](0, 0) = 1.;
    // A terminal pair outside the stored LU pattern: the workspace covers the radial chain 0-1-2,
    // which has no (0, 2) block, while the request names the branch between buses 0 and 2.
    auto radial_topo = branch_topology();
    radial_topo.branch_bus_idx = {{0, 1}, {1, 2}};
    MathModelParam<symmetric_t> radial_params;
    radial_params.branch_param.resize(2, {1., -1., -1., 1.});
    YBus<symmetric_t> const radial_y_bus{radial_topo, radial_params};
    std::vector<ILGain> radial_gain(radial_y_bus.nnz_lu());
    for (auto d : radial_y_bus.lu_diag()) {
        radial_gain[d](0, 0) = 1;
        radial_gain[d](1, 1) = -1;
    }
    SparseCovarianceWorkspace<ILGain, ILUnknown, 1> radial_workspace{radial_y_bus, radial_gain, 1., {1., 1., 1.}, true};
    REQUIRE(!radial_workspace.find_block_index(0, 2).has_value());
    request.branch = {{1, BranchSide::from, true, true, true, true}};
    StateEstimationUQResult<symmetric_t> missing_result;
    PhysicalBlocks missing_result_blocks{radial_workspace};
    branch_results(radial_workspace, y_bus, request, missing_result, missing_result_blocks);
    CHECK(missing_result.branch[0].reactive_power.phase[0].raw_status == UQVarianceStatus::invalid);
}

TEST_CASE("Branch UQ treats a current with subnormal squared modulus as undefined") {
    auto topo = branch_topology();
    MathModelParam<symmetric_t> params;
    // With unit voltages, I = j - (1e-160 + j) = -1e-160: |I|^2 is subnormal.
    params.branch_param.resize(3, {DoubleComplex{0.0, 1.0}, -DoubleComplex{1e-160, 1.0}, 1., 1.});
    YBus<symmetric_t> const y_bus{topo, params};
    std::vector<ILGain> gain(y_bus.nnz_lu());
    for (auto d : y_bus.lu_diag()) {
        gain[d](0, 0) = 1;
        gain[d](1, 1) = -1;
    }
    SparseCovarianceWorkspace<ILGain, ILUnknown, 1> const workspace{y_bus, gain, 1., {1., 1., 1.}, true};
    StateEstimationUQRequest request;
    request.branch = {{0, BranchSide::from, true, true, true, true}};
    StateEstimationUQResult<symmetric_t> result;
    PhysicalBlocks result_blocks{workspace};
    branch_results(workspace, y_bus, request, result, result_blocks);
    auto const& output = result.branch[0];
    CHECK(output.active_power.phase[0].raw_status == UQVarianceStatus::valid);
    CHECK(output.reactive_power.phase[0].raw_status == UQVarianceStatus::valid);
    for (auto const* quantity : {&output.current_magnitude, &output.current_angle}) {
        CHECK(quantity->phase[0].raw_status == UQVarianceStatus::invalid);
        CHECK(std::isnan(quantity->phase[0].sigma));
    }
}

TEST_CASE("Branch UQ grouped complex products preserve zero current") {
    auto check = []<class Gain, class Unknown, int N>() {
        auto topo = branch_topology();
        MathModelParam<symmetric_t> params;
        DoubleComplex const coefficient{20.0, -40.0};
        params.branch_param.resize(3, {coefficient, -coefficient, -coefficient, coefficient});
        YBus<symmetric_t> const y_bus{topo, params};
        std::vector<Gain> gain(y_bus.nnz_lu());
        for (auto const diagonal : y_bus.lu_diag()) {
            for (int axis = 0; axis != N; ++axis) {
                gain[diagonal](axis, axis) = 1.0;
                gain[diagonal](N + axis, N + axis) = -1.0;
            }
        }
        ComplexValueVector<symmetric_t> const u(3, std::polar(1.0, 0.13));
        auto const ordinary = y_bus.calculate_branch_flow<BranchSolverOutput<symmetric_t>>(u);
        REQUIRE(ordinary[0].i_f == DoubleComplex{});
        REQUIRE(ordinary[0].i_t == DoubleComplex{});
        SparseCovarianceWorkspace<Gain, Unknown, N> const workspace{y_bus, gain, 1.0, u, true};
        StateEstimationUQRequest request;
        request.branch = {{0, BranchSide::from, true, true, true, true}, {0, BranchSide::to, true, true, true, true}};
        StateEstimationUQResult<symmetric_t> result;
        PhysicalBlocks result_blocks{workspace};
        branch_results(workspace, y_bus, request, result, result_blocks);
        REQUIRE(result.branch.size() == 2);
        for (auto const& output : result.branch) {
            for (auto const* quantity : {&output.active_power, &output.reactive_power}) {
                CHECK(quantity->phase[0].raw_status == UQVarianceStatus::valid);
                CHECK(std::isfinite(quantity->phase[0].raw_variance));
                CHECK(std::isfinite(quantity->phase[0].sigma));
            }
            for (auto const* quantity : {&output.current_magnitude, &output.current_angle}) {
                CHECK(quantity->phase[0].raw_status == UQVarianceStatus::invalid);
                CHECK(std::isnan(quantity->phase[0].sigma));
            }
        }
    };
    check.template operator()<ILGain, ILUnknown, 1>();
    check.template operator()<NRGain, NRUnknown, 2>();
}
} // namespace power_grid_model::math_solver

namespace power_grid_model::math_solver {
TEST_CASE("Branch UQ handles disconnected terminals and zero voltage") {
    using namespace state_estimation_uq;
    auto topo = branch_topology();
    topo.branch_bus_idx[0] = {-1, 1};
    MathModelParam<symmetric_t> params;
    params.branch_param.resize(3, {1., -1., -1., 2.});
    YBus<symmetric_t> const y_bus{topo, params};
    std::vector<ILGain> gain(y_bus.nnz_lu());
    for (auto d : y_bus.lu_diag()) {
        gain[d](0, 0) = 1;
        gain[d](1, 1) = -1;
    }
    SparseCovarianceWorkspace<ILGain, ILUnknown, 1> workspace{y_bus, gain, 1., {1., 1., 1.}, true};
    StateEstimationUQRequest request;
    request.branch = {{0, BranchSide::from, true, true, true, true}, {0, BranchSide::to, true, true, true, true}};
    StateEstimationUQResult<symmetric_t> result;
    PhysicalBlocks result_blocks{workspace};
    branch_results(workspace, y_bus, request, result, result_blocks);
    CHECK(result.branch[0].active_power.phase[0].raw_status == UQVarianceStatus::invalid);
    CHECK(result.branch[0].current_angle.phase[0].raw_status == UQVarianceStatus::invalid);
    CHECK(result.branch[1].active_power.phase[0].raw_status == UQVarianceStatus::valid);
    CHECK(result.branch[1].current_angle.phase[0].raw_variance == doctest::Approx(0.5));
    SparseCovarianceWorkspace<ILGain, ILUnknown, 1> zero_workspace{y_bus, gain, 1., {1., 0., 1.}, true};
    StateEstimationUQResult<symmetric_t> zero_voltage;
    PhysicalBlocks zero_voltage_blocks{zero_workspace};
    branch_results(zero_workspace, y_bus, request, zero_voltage, zero_voltage_blocks);
    CHECK(zero_voltage.branch[1].active_power.phase[0].raw_status == UQVarianceStatus::valid);
    CHECK(zero_voltage.branch[1].active_power.phase[0].raw_variance == 0);
    CHECK(zero_voltage.branch[1].current_magnitude.phase[0].raw_status == UQVarianceStatus::invalid);
}
} // namespace power_grid_model::math_solver

namespace power_grid_model::math_solver {
TEST_CASE("Branch UQ combines repeated endpoints") {
    using namespace state_estimation_uq;
    auto topo = branch_topology();
    for (bool const self_loop : {false, true}) {
        if (self_loop) {
            topo.branch_bus_idx[0] = {0, 0};
        }
        MathModelParam<symmetric_t> params;
        params.branch_param.resize(3, {1., 1., 1., 1.});
        YBus<symmetric_t> const y_bus{topo, params};
        std::vector<ILGain> gain(y_bus.nnz_lu());
        for (auto d : y_bus.lu_diag()) {
            gain[d](0, 0) = 1;
            gain[d](1, 1) = -1;
        }
        SparseCovarianceWorkspace<ILGain, ILUnknown, 1> workspace{y_bus, gain, 1., {1., 1., 1.}, true};
        StateEstimationUQRequest request;
        request.branch = {{0, BranchSide::from, true, true, true, true}};
        StateEstimationUQResult<symmetric_t> result;
        PhysicalBlocks result_blocks{workspace};
        branch_results(workspace, y_bus, request, result, result_blocks);
        auto const& output = result.branch[0];
        double const expected_magnitude = self_loop ? 2.0 : 1.0;
        CHECK(output.current_magnitude.phase[0].raw_status == UQVarianceStatus::valid);
        CHECK(output.current_magnitude.phase[0].raw_variance == doctest::Approx(expected_magnitude));
        CHECK(output.current_angle.phase[0].raw_status == UQVarianceStatus::valid);
        double const expected_angle = expected_magnitude / 4.0;
        CHECK(std::abs(output.current_angle.phase[0].raw_variance / expected_angle - 1.0) < 1e-12);
    }
}
} // namespace power_grid_model::math_solver

namespace power_grid_model::math_solver {
TEST_CASE("Branch UQ rejects inconsistent original pair orientations before contraction") {
    using namespace state_estimation_uq;
    auto topo = branch_topology();
    MathModelParam<symmetric_t> params;
    params.branch_param.resize(3, {1., 1., 1., 1.});
    YBus<symmetric_t> const y_bus{topo, params};
    std::vector<ILGain> gain(y_bus.nnz_lu());
    for (auto d : y_bus.lu_diag()) {
        gain[d](0, 0) = 1;
        gain[d](1, 1) = -1;
    }
    SparseCovarianceWorkspace<ILGain, ILUnknown, 1> workspace{y_bus, gain, 1., {1., 1., 1.}, true};
    auto const forward = workspace.find_block_index(0, 1), reverse = workspace.find_block_index(1, 0);
    REQUIRE(forward.has_value());
    REQUIRE(reverse.has_value());
    // Opposite defects cancel in h C h^T, so scalar validity alone misses them.
    workspace.augmented_matrix[*forward](0, 0) = 0.2;
    workspace.augmented_matrix[*reverse](0, 0) = -0.2;
    StateEstimationUQRequest request;
    request.branch = {{0, BranchSide::from, true, true, true, true}};
    StateEstimationUQResult<symmetric_t> result;
    PhysicalBlocks result_blocks{workspace};
    branch_results(workspace, y_bus, request, result, result_blocks);
    auto const& output = result.branch[0];
    CHECK(output.active_power.phase[0].raw_status == UQVarianceStatus::invalid);
    CHECK(output.reactive_power.phase[0].raw_status == UQVarianceStatus::invalid);
    CHECK(output.current_magnitude.phase[0].raw_status == UQVarianceStatus::invalid);
    CHECK(output.current_angle.phase[0].raw_status == UQVarianceStatus::invalid);
}
} // namespace power_grid_model::math_solver

namespace power_grid_model::math_solver {
TEST_CASE("NRSE branch UQ zero voltage keeps signed powers and rejects current") {
    using namespace state_estimation_uq;
    auto topo = branch_topology();
    MathModelParam<symmetric_t> params;
    params.branch_param.resize(3, {1., 1., 1., 1.});
    YBus<symmetric_t> const y_bus{topo, params};
    std::vector<NRGain> gain(y_bus.nnz_lu());
    for (auto d : y_bus.lu_diag()) {
        gain[d](0, 0) = gain[d](1, 1) = 1;
        gain[d](2, 2) = gain[d](3, 3) = -1;
    }
    SparseCovarianceWorkspace<NRGain, NRUnknown, 2> workspace{y_bus, gain, 1., {0.0, 0.0, 1.}, true};
    StateEstimationUQRequest request;
    request.branch = {{0, BranchSide::from, true, true, true, true}};
    StateEstimationUQResult<symmetric_t> result;
    PhysicalBlocks result_blocks{workspace};
    branch_results(workspace, y_bus, request, result, result_blocks);
    auto const& output = result.branch[0];
    // The exact zero-voltage power derivatives stay valid.
    CHECK(output.active_power.phase[0].raw_status == UQVarianceStatus::valid);
    CHECK(output.reactive_power.phase[0].raw_status == UQVarianceStatus::valid);
    CHECK(output.current_magnitude.phase[0].raw_status == UQVarianceStatus::invalid);
    CHECK(output.current_angle.phase[0].raw_status == UQVarianceStatus::invalid);
}
} // namespace power_grid_model::math_solver

namespace power_grid_model::math_solver {
namespace {
template <class Gain, class Unknown, int N> void shared_terminal_blocks() {
    using Scalar = typename Gain::Scalar;
    constexpr int size = Gain::RowsAtCompileTime;
    auto topo = branch_topology();
    MathModelParam<symmetric_t> parameters;
    parameters.branch_param.resize(
        3, {DoubleComplex{2, -3}, DoubleComplex{-1, 2}, DoubleComplex{-1.2, 1.8}, DoubleComplex{1.7, -2.6}});
    YBus<symmetric_t> const y_bus{topo, parameters};
    Eigen::Matrix<Scalar, 3 * N, 3 * N> c;
    for (int i = 0; i != 3 * N; ++i) {
        for (int j = 0; j != 3 * N; ++j) {
            c(i, j) = i == j ? 1.0 + 0.1 * i : 0.03 * (i + j + 1);
        }
    }
    auto const precision = c.inverse().eval();
    std::vector<Gain> gain(y_bus.nnz_lu());
    for (Idx i = 0; i != 3; ++i) {
        for (Idx k = y_bus.row_indptr_lu()[i]; k != y_bus.row_indptr_lu()[i + 1]; ++k) {
            Idx const j = y_bus.col_indices_lu()[k];
            gain[k].matrix().template topLeftCorner<N, N>() = precision.template block<N, N>(i * N, j * N);
            for (int a = N; i == j && a != size; ++a) {
                gain[k](a, a) = -1;
            }
        }
    }
    ComplexValueVector<symmetric_t> const u{std::polar(1.2, 0.2), std::polar(0.8, -0.1), std::polar(1.1, 0.3)};
    for (bool const relative : {false, true}) {
        // NRSE has a virtual angle exactly when it reports relative angles.
        bool const virtual_angle = N == 2 && relative;
        SparseCovarianceWorkspace<Gain, Unknown, N> const workspace{
            y_bus, gain, 1.0, u, !relative, virtual_angle ? 0 : -1, virtual_angle ? 0.2 : 0.0};
        // Fill reverse orientations first; every access must equal a fresh derivation.
        PhysicalBlocks cache{workspace};
        branch_detail::TerminalBlocks blocks{cache};
        for (Idx first : {2, 1, 0}) {
            for (Idx second : {0, 1, 2}) {
                auto const cached = blocks(first, second);
                auto const fresh = branch_detail::physical_pair(workspace, first, second);
                CHECK(cached.valid == fresh.valid);
                CHECK(cached.construction_scale == fresh.construction_scale);
                CHECK((cached.value.array() == fresh.value.array()).all());
            }
        }
        // One call sharing blocks across branches and sides equals one call per request.
        StateEstimationUQRequest all;
        for (Idx branch = 0; branch != 3; ++branch) {
            for (auto const side : {BranchSide::from, BranchSide::to}) {
                all.branch.push_back({branch, side, true, true, true, true});
            }
        }
        StateEstimationUQResult<symmetric_t> shared;
        PhysicalBlocks shared_blocks{workspace};
        branch_results(workspace, y_bus, all, shared, shared_blocks);
        // A caller-owned cache, also when already filled, gives the same results.
        PhysicalBlocks stored{workspace};
        for (int repeat = 0; repeat != 2; ++repeat) {
            StateEstimationUQResult<symmetric_t> external;
            branch_results(workspace, y_bus, all, external, stored);
            REQUIRE(external.branch.size() == shared.branch.size());
            for (std::size_t index = 0; index != all.branch.size(); ++index) {
                auto const& a = shared.branch[index];
                auto const& b = external.branch[index];
                for (auto const& [x, y] : {std::pair{a.active_power, b.active_power},
                                           {a.reactive_power, b.reactive_power},
                                           {a.current_magnitude, b.current_magnitude},
                                           {a.current_angle, b.current_angle}}) {
                    CHECK(x.phase[0].raw_status == y.phase[0].raw_status);
                    CHECK(x.phase[0].raw_variance == y.phase[0].raw_variance);
                    CHECK(x.phase[0].absolute_scale == y.phase[0].absolute_scale);
                }
            }
        }
        for (std::size_t index = 0; index != all.branch.size(); ++index) {
            StateEstimationUQRequest single;
            single.branch = {all.branch[index]};
            StateEstimationUQResult<symmetric_t> separate;
            PhysicalBlocks separate_blocks{workspace};
            branch_results(workspace, y_bus, single, separate, separate_blocks);
            auto const& a = shared.branch[index];
            auto const& b = separate.branch[0];
            for (auto const& [x, y] : {std::pair{a.active_power, b.active_power},
                                       {a.reactive_power, b.reactive_power},
                                       {a.current_magnitude, b.current_magnitude},
                                       {a.current_angle, b.current_angle}}) {
                CHECK(x.phase[0].raw_status == UQVarianceStatus::valid);
                CHECK(x.phase[0].raw_status == y.phase[0].raw_status);
                CHECK(x.phase[0].raw_variance == y.phase[0].raw_variance);
                CHECK(x.phase[0].absolute_scale == y.phase[0].absolute_scale);
            }
        }
    }
}
} // namespace

TEST_CASE("Branch UQ derives each terminal block once and shares it exactly") {
    shared_terminal_blocks<ILGain, ILUnknown, 1>();
    shared_terminal_blocks<NRGain, NRUnknown, 2>();
}
} // namespace power_grid_model::math_solver
