// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
// SPDX-License-Identifier: MPL-2.0

// Integrated checks of the symmetric UQ core against independent references:
// sparse Full and separator completion, a constrained NRSE reference for every
// output group, and seeded noisy samples for both solvers.

#include <power_grid_model/common/dummy_logging.hpp>
#include <power_grid_model/math_solver/iterative_linear_se_solver.hpp>
#include <power_grid_model/math_solver/newton_raphson_se_solver.hpp>
#include <power_grid_model/math_solver/state_estimation_uq_completion.hpp>
#include <power_grid_model/math_solver/state_estimation_uq_injection.hpp>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <doctest/doctest.h>

#include <array>
#include <random>
#include <vector>

namespace full_reference {

using namespace power_grid_model;
using namespace power_grid_model::math_solver;
using namespace power_grid_model::math_solver::state_estimation_uq;
using ILGain = iterative_linear_se::ILSEGainBlock<symmetric_t>;
using ILUnknown = iterative_linear_se::ILSEUnknown<symmetric_t>;
using NRGain = newton_raphson_se::NRSEGainBlock<symmetric_t>;
using NRUnknown = newton_raphson_se::NRSEUnknown<symmetric_t>;

MathModelTopology topology(bool cycle, bool full) {
    MathModelTopology t;
    t.slack_bus = 3;
    t.phase_shift = {0, 0, 0, 0};
    t.branch_bus_idx =
        cycle ? std::vector<BranchIdx>{{0, 1}, {1, 2}, {2, 3}, {3, 0}} : std::vector<BranchIdx>{{0, 3}, {1, 3}, {2, 3}};
    t.sources_per_bus = {from_sparse, {0, 0, 0, 0, 0}};
    t.shunts_per_bus = {from_sparse, {0, 0, 0, 0, 0}};
    t.load_gens_per_bus = {from_sparse, {0, 0, 0, 0, 0}};
    std::array<std::array<bool, 4>, 4> graph{};
    for (auto edge : t.branch_bus_idx)
        graph[edge[0]][edge[1]] = graph[edge[1]][edge[0]] = true;
    // Full adds requested distinct-neighbor pairs to the augmented structural
    // graph BEFORE symbolic elimination. Added numerical entries stay zero.
    if (full) {
        std::vector<BranchIdx> const required =
            cycle ? std::vector<BranchIdx>{{0, 2}} : std::vector<BranchIdx>{{0, 1}, {0, 2}, {1, 2}};
        for (auto pair : required)
            if (!graph[pair[0]][pair[1]]) {
                graph[pair[0]][pair[1]] = graph[pair[1]][pair[0]] = true;
                t.fill_in.push_back(pair);
            }
    }
    for (int k = 0; k < 4; ++k)
        for (int a = k + 1; a < 4; ++a)
            for (int b = a + 1; b < 4; ++b) {
                if (graph[k][a] && graph[k][b] && !graph[a][b]) {
                    graph[a][b] = graph[b][a] = true;
                    t.fill_in.push_back({a, b});
                }
            }
    return t;
}

template <class Gain, class Unknown, int D> void reference(bool cycle) {
    using Scalar = typename Gain::Scalar;
    using Dense = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    constexpr int blocksize = 2 * D;
    auto ordinary_topo = topology(cycle, false), full_topo = topology(cycle, true);
    MathModelParam<symmetric_t> param;
    DoubleComplex const y{2, -4};
    param.branch_param.resize(ordinary_topo.branch_bus_idx.size(), {y, -y, -y, y});
    YBus<symmetric_t> const original{ordinary_topo, param}, full{full_topo, param};
    ComplexValueVector<symmetric_t> u{std::polar(1.04, .13), std::polar(.96, -.08), std::polar(1.01, .03), 0};
    u[3] = cycle ? (u[0] + u[2]) / 2.0 : (u[0] + u[1] + u[2]) / 3.0;
    Eigen::Matrix4cd nodal = Eigen::Matrix4cd::Zero();
    for (auto edge : ordinary_topo.branch_bus_idx) {
        auto a = edge[0], b = edge[1];
        nodal(a, a) += y;
        nodal(b, b) += y;
        nodal(a, b) -= y;
        nodal(b, a) -= y;
    }
    Dense matrix = Dense::Zero(4 * blocksize, 4 * blocksize);
    constexpr double alpha = .2;
    if constexpr (D == 1) {
        Eigen::Matrix4cd g = Eigen::Matrix4cd::Zero();
        for (int b = 0; b < 4; ++b)
            g(b, b) = 1.0 / (.7 + .2 * b);
        for (auto edge : ordinary_topo.branch_bus_idx) {
            Eigen::RowVector4cd row = Eigen::RowVector4cd::Zero();
            row(edge[0]) = y;
            row(edge[1]) = -y;
            g += row.adjoint() * row / .8;
        }
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
                matrix(2 * a, 2 * b) = alpha * g(a, b);
        for (int b = 0; b < 4; ++b) {
            matrix(2 * 3 + 1, 2 * b) = nodal(3, b);
            matrix(2 * b, 2 * 3 + 1) = std::conj(nodal(3, b));
        }
        for (int b = 0; b < 3; ++b)
            matrix(2 * b + 1, 2 * b + 1) = -1 / alpha;
    } else {
        Eigen::Vector<double, 8> q;
        for (int b = 0; b < 4; ++b) {
            q(2 * b) = std::arg(u[b]);
            q(2 * b + 1) = std::abs(u[b]);
        }
        auto phasors = [](Eigen::Vector<double, 8> const& x) {
            Eigen::Vector4cd v;
            for (int b = 0; b < 4; ++b)
                v(b) = std::polar(x(2 * b + 1), x(2 * b));
            return v;
        };
        Eigen::Matrix<double, 8, 8> g = Eigen::Matrix<double, 8, 8>::Zero();
        for (int b = 0; b < 4; ++b)
            g(2 * b, 2 * b) = g(2 * b + 1, 2 * b + 1) = 1 / (.7 + .2 * b);
        for (auto edge : ordinary_topo.branch_bus_idx) {
            Eigen::Matrix<double, 2, 8> jac;
            for (int k = 0; k < 8; ++k) {
                auto l = q, r = q;
                l(k) -= 1e-5;
                r(k) += 1e-5;
                auto vl = phasors(l), vr = phasors(r);
                auto diff = y * ((vr(edge[0]) - vr(edge[1])) - (vl(edge[0]) - vl(edge[1]))) / 2e-5;
                jac(0, k) = diff.real();
                jac(1, k) = diff.imag();
            }
            g += jac.transpose() * jac / .8;
        }
        Eigen::Matrix<double, 2, 8> constraint;
        for (int k = 0; k < 8; ++k) {
            auto l = q, r = q;
            l(k) -= 1e-5;
            r(k) += 1e-5;
            auto vl = phasors(l), vr = phasors(r);
            auto sl = vl(3) * std::conj((nodal * vl)(3)), sr = vr(3) * std::conj((nodal * vr)(3));
            auto diff = (sr - sl) / 2e-5;
            constraint(0, k) = diff.real();
            constraint(1, k) = diff.imag();
        }
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
                matrix.template block<2, 2>(4 * a, 4 * b) = alpha * g.template block<2, 2>(2 * a, 2 * b);
        for (int b = 0; b < 4; ++b) {
            matrix.template block<2, 2>(4 * 3 + 2, 4 * b) = constraint.template block<2, 2>(0, 2 * b);
            matrix.template block<2, 2>(4 * b, 4 * 3 + 2) = constraint.template block<2, 2>(0, 2 * b).transpose();
        }
        for (int b = 0; b < 3; ++b)
            matrix.template block<2, 2>(4 * b + 2, 4 * b + 2) = -Eigen::Matrix2d::Identity() / alpha;
    }
    auto blocks = [&](YBus<symmetric_t> const& bus) {
        std::vector<Gain> result(bus.nnz_lu());
        for (Idx a = 0; a < 4; ++a)
            for (Idx k = bus.row_indptr_lu()[a]; k < bus.row_indptr_lu()[a + 1]; ++k) {
                Idx b = bus.col_indices_lu()[k];
                result[k].matrix() = matrix.block(blocksize * a, blocksize * b, blocksize, blocksize);
            }
        return result;
    };
    auto original_blocks = blocks(original), full_blocks = blocks(full);
    SparseCovarianceWorkspace<Gain, Unknown, D> const selected{original, original_blocks, alpha, u, true};
    SparseCovarianceWorkspace<Gain, Unknown, D> const complete{full, full_blocks, alpha, u, true};
    Dense const dense = matrix.fullPivLu().inverse();
    double dense_residual = (matrix * dense - Dense::Identity(matrix.rows(), matrix.cols())).norm();
    double max_full = 0, max_original = 0;
    for (Idx a = 0; a < 4; ++a)
        for (Idx b = 0; b < 4; ++b) {
            if (auto const index = complete.find_block_index(a, b))
                max_full = std::max(max_full, (complete.augmented_matrix[*index].matrix() -
                                               dense.block(a * blocksize, b * blocksize, blocksize, blocksize))
                                                  .norm());
            if (auto const index = selected.find_block_index(a, b))
                max_original = std::max(max_original, (selected.augmented_matrix[*index].matrix() -
                                                       dense.block(a * blocksize, b * blocksize, blocksize, blocksize))
                                                          .norm());
        }
    CHECK(dense_residual <= 1e-10);
    CHECK(max_full <= 1e-10);
    CHECK(max_original <= 1e-10);
    // Entries added for Full stay numerically zero.
    for (Idx a = 0; a < 4; ++a)
        for (Idx k = full.row_indptr_lu()[a]; k < full.row_indptr_lu()[a + 1]; ++k) {
            auto b = full.col_indices_lu()[k];
            if (!selected.find_block_index(a, b))
                CHECK(full_blocks[k].matrix().norm() == 0);
        }
    // Exact separator candidate for the star; the cycle has a bypass and
    // retains the first conditioned raw augmented product as an approximation.
    if (!cycle) {
        auto a = 0, b = 1, k = 3;
        auto const raw_block = [&](Idx row, Idx col) {
            return selected.augmented_matrix[*selected.find_block_index(row, col)].matrix();
        };
        auto raw = (raw_block(a, k) * raw_block(k, k).fullPivLu().solve(raw_block(k, b))).eval();
        CHECK((raw - dense.block(a * blocksize, b * blocksize, blocksize, blocksize)).norm() <= 1e-10);
    }
    std::vector<std::vector<Idx>> graph(4);
    for (auto edge : ordinary_topo.branch_bus_idx) {
        graph[edge[0]].push_back(edge[1]);
        graph[edge[1]].push_back(edge[0]);
    }
    PhysicalBlocks completion_stored{selected};
    CovarianceCompletion completion{selected, graph, graph, completion_stored};
    Idx const first = 0, second = cycle ? 2 : 1;
    auto const& completed = completion.pair(first, second);
    auto const exact = complete.physical_block(first, second);
    REQUIRE(completed.block.valid);
    if (!cycle) {
        CHECK(completed.diagnostic.route == UQPairRoute::separator);
        CHECK((completed.block.value - exact.value).norm() <= 1e-10);
    }
    if (cycle) {
        constexpr Idx middle = 1; // Ascending augmented common middles are {1, 3}.
        Eigen::Matrix<Scalar, blocksize, blocksize> const raw =
            dense.template block<blocksize, blocksize>(first * blocksize, middle * blocksize) *
            dense.template block<blocksize, blocksize>(middle * blocksize, middle * blocksize).inverse() *
            dense.template block<blocksize, blocksize>(middle * blocksize, second * blocksize);
        // This fixture has an absolute angle and no virtual row: independently
        // apply its one normalization and physical coordinate map to the product.
        typename SparseCovarianceWorkspace<Gain, Unknown, D>::PhysicalMatrix predicted;
        if constexpr (D == 1) {
            predicted(0, 0) = alpha * raw(0, 0);
        } else {
            auto jacobian = [&](Idx bus) {
                auto const phase = std::polar(1., std::arg(u[bus]));
                Eigen::Matrix2d result;
                result << -u[bus].imag(), phase.real(), u[bus].real(), phase.imag();
                return result;
            };
            predicted = alpha * jacobian(first) * raw.template topLeftCorner<2, 2>() * jacobian(second).transpose();
        }
        CHECK(completed.diagnostic.route == UQPairRoute::augmented_path);
        CHECK(completed.diagnostic.middle_bus == middle);
        CHECK((completed.block.value - predicted).norm() <= 1e-10);
        // The fixture must expose the nonseparator approximation error.
        CHECK((predicted - exact.value).norm() > 1e-10);
    }
}

} // namespace full_reference

namespace constrained_reference {

using namespace power_grid_model;
using namespace power_grid_model::math_solver;
void check() {
    MathModelTopology t;
    t.slack_bus = 0;
    t.phase_shift = {0, 0, 0};
    t.branch_bus_idx = {{0, 1}, {1, 2}};
    t.sources_per_bus = {from_sparse, {0, 1, 1, 2}};
    t.shunts_per_bus = {from_sparse, {0, 0, 0, 0}};
    t.load_gens_per_bus = {from_sparse, {0, 0, 0, 0}};
    t.voltage_sensors_per_bus = {from_sparse, {0, 1, 2, 3}};
    t.power_sensors_per_bus = {from_sparse, {0, 0, 0, 0}};
    t.power_sensors_per_source = {from_sparse, {0, 0, 0}};
    t.power_sensors_per_load_gen = {from_sparse, {0}};
    t.power_sensors_per_shunt = {from_sparse, {0}};
    t.power_sensors_per_branch_from = {from_sparse, {0, 1, 2}};
    t.power_sensors_per_branch_to = {from_sparse, {0, 0, 0}};
    MathModelParam<symmetric_t> param;
    DoubleComplex const y{2, -4};
    param.branch_param = {{y, -y, -y, y}, {y, -y, -y, y}};
    YBus<symmetric_t> const y_bus{t, param};
    ComplexValueVector<symmetric_t> truth{std::polar(1.02, .04), 0, std::polar(.98, -.05)};
    truth[1] = (truth[0] + truth[2]) / 2.0;
    StateEstimationInput<symmetric_t> input;
    input.source_status = {1, 1};
    input.measured_voltage = {{{std::abs(truth[0]), power_grid_model::nan}, 1e-4},
                              {{std::abs(truth[1]), power_grid_model::nan}, .8e-4},
                              {{std::abs(truth[2]), power_grid_model::nan}, 1.2e-4}};
    auto s0 = truth[0] * std::conj(y * (truth[0] - truth[1])), s1 = truth[1] * std::conj(y * (truth[1] - truth[2]));
    input.measured_branch_from_power = {{.real_component = {.value = s0.real() + .002, .variance = .7e-4},
                                         .imag_component = {.value = s0.imag() - .001, .variance = .4e-4}},
                                        {.real_component = {.value = s1.real() - .001, .variance = .9e-4},
                                         .imag_component = {.value = s1.imag() + .003, .variance = .6e-4}}};
    NewtonRaphsonSESolver<symmetric_t> solver{y_bus, t};
    common::logging::NoLogger log;
    auto ordinary = solver.run_state_estimation(y_bus, input, 1e-12, 100, log);
    StateEstimationUQRequest request;
    request.voltage = {{0, true, true}, {1, true, true}, {2, true, true}};
    request.branch = {{0, BranchSide::from, true, true, true, true},
                      {0, BranchSide::to, true, true, true, true},
                      {1, BranchSide::from, true, true, true, true},
                      {1, BranchSide::to, true, true, true, true}};
    request.injection = {{0, true, true}, {1, true, true}, {2, true, true}};
    request.pair_diagnostics = true;
    StateEstimationUQResult<symmetric_t> result;
    StateEstimationUQContext<symmetric_t> context{request, result};
    auto constrained = solver.run_state_estimation(y_bus, input, 1e-12, 100, log, &context);
    CHECK((constrained.u == ordinary.u));
    Eigen::Vector<double, 6> q;
    for (int b = 0; b < 3; ++b) {
        q(2 * b) = std::arg(constrained.u[b]);
        q(2 * b + 1) = std::abs(constrained.u[b]);
    }
    auto rows = [&](Eigen::Vector<double, 6> const& state) {
        std::array<DoubleComplex, 3> u;
        for (int b = 0; b < 3; ++b)
            u[b] = std::polar(state(2 * b + 1), state(2 * b));
        auto a = u[0] * std::conj(y * (u[0] - u[1])), b = u[1] * std::conj(y * (u[1] - u[2]));
        Eigen::Vector<double, 7> r;
        r << state(1), state(3), state(5), a.real(), a.imag(), b.real(), b.imag();
        return r;
    };
    Eigen::Matrix<double, 7, 5> h;
    for (int k = 1; k < 6; ++k) {
        auto l = q, r = q;
        l(k) -= 1e-5;
        r(k) += 1e-5;
        h.col(k - 1) = (rows(r) - rows(l)) / 2e-5;
    }
    Eigen::Vector<double, 7> w;
    w << 1e4, 1 / .8e-4, 1 / 1.2e-4, 1 / .7e-4, 1 / .4e-4, 1 / .9e-4, 1 / .6e-4;
    Eigen::Matrix<double, 5, 5> const gain = h.transpose() * w.asDiagonal() * h;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 5, 5>> eig{gain};
    REQUIRE(eig.eigenvalues().minCoeff() > 0); // finite sensor rows are observable
    // Independent constrained reference uses the same finite sensor rows but
    // keeps the exact middle-bus P/Q constraints in a reduced augmented system.
    // Reference reduction removes the virtual row without any noise subtraction.
    auto injection = [&](Eigen::Vector<double, 6> const& state) {
        std::array<DoubleComplex, 3> u;
        for (int b = 0; b < 3; ++b)
            u[b] = std::polar(state(2 * b + 1), state(2 * b));
        return u[1] * std::conj(y * (2.0 * u[1] - u[0] - u[2]));
    };
    Eigen::Matrix<double, 2, 5> exact_rows;
    for (int k = 1; k < 6; ++k) {
        auto l = q, r = q;
        l(k) -= 1e-5;
        r(k) += 1e-5;
        auto diff = (injection(r) - injection(l)) / 2e-5;
        exact_rows(0, k - 1) = diff.real();
        exact_rows(1, k - 1) = diff.imag();
    }
    Eigen::Matrix<double, 7, 7> augmented = Eigen::Matrix<double, 7, 7>::Zero();
    augmented.block<5, 5>(0, 0) = gain;
    augmented.block<2, 5>(5, 0) = exact_rows;
    augmented.block<5, 2>(0, 5) = exact_rows.transpose();
    Eigen::Matrix<double, 7, 7> const augmented_inverse = augmented.fullPivLu().inverse();
    REQUIRE((augmented * augmented_inverse - Eigen::Matrix<double, 7, 7>::Identity()).norm() <= 1e-8);
    Eigen::Matrix<double, 6, 6> constrained_covariance = Eigen::Matrix<double, 6, 6>::Zero();
    constrained_covariance.block<5, 5>(1, 1) = augmented_inverse.block<5, 5>(0, 0);
    double max_variance_error = 0;
    for (int b = 0; b < 3; ++b) {
        max_variance_error =
            std::max(max_variance_error, std::abs(static_cast<double>(result.voltage[b].angle.phase[0].raw_variance) -
                                                  constrained_covariance(2 * b, 2 * b)));
        max_variance_error = std::max(max_variance_error,
                                      std::abs(static_cast<double>(result.voltage[b].magnitude.phase[0].raw_variance) -
                                               constrained_covariance(2 * b + 1, 2 * b + 1)));
    }
    CHECK(max_variance_error <= 1e-12);
    // All physical output maps use this same constrained returned estimate.
    auto observable_rows = [&](Eigen::Vector<double, 6> const& state) {
        std::array<DoubleComplex, 3> u;
        for (int node = 0; node < 3; ++node)
            u[node] = std::polar(state(2 * node + 1), state(2 * node));
        Eigen::Matrix<double, 28, 1> out;
        for (int node = 0; node < 3; ++node) {
            out(2 * node) = std::abs(u[node]);
            out(2 * node + 1) = std::arg(u[node]);
        }
        for (int edge = 0; edge < 2; ++edge) {
            auto const current = y * (u[edge] - u[edge + 1]);
            for (int side = 0; side < 2; ++side) {
                auto const terminal_i = side == 0 ? current : -current;
                auto const power = u[edge + side] * std::conj(terminal_i);
                int const offset = 6 + 8 * edge + 4 * side;
                out(offset) = std::abs(terminal_i);
                out(offset + 1) = std::arg(terminal_i);
                out(offset + 2) = power.real();
                out(offset + 3) = power.imag();
            }
        }
        std::array<DoubleComplex, 3> const currents{y * (u[0] - u[1]), y * (2.0 * u[1] - u[0] - u[2]),
                                                    y * (u[2] - u[1])};
        for (int node = 0; node < 3; ++node) {
            auto const power = u[node] * std::conj(currents[node]);
            out(22 + 2 * node) = power.real();
            out(23 + 2 * node) = power.imag();
        }
        return out;
    };
    Eigen::Matrix<double, 28, 6> coarse_output_jacobian, half_output_jacobian, output_jacobian;
    for (int coordinate = 0; coordinate < 6; ++coordinate) {
        auto coarse_left = q, coarse_right = q, half_left = q, half_right = q;
        coarse_left(coordinate) -= 1e-5;
        coarse_right(coordinate) += 1e-5;
        half_left(coordinate) -= 5e-6;
        half_right(coordinate) += 5e-6;
        coarse_output_jacobian.col(coordinate) = (observable_rows(coarse_right) - observable_rows(coarse_left)) / 2e-5;
        half_output_jacobian.col(coordinate) = (observable_rows(half_right) - observable_rows(half_left)) / 1e-5;
    }
    // Richardson cancellation removes the O(h^2) error of current-angle
    // differences at small nominal currents, without loosening the variance gate.
    output_jacobian = (4 * half_output_jacobian - coarse_output_jacobian) / 3;
    Eigen::Matrix<double, 28, 28> const constrained_output =
        output_jacobian * constrained_covariance * output_jacobian.transpose();
    Eigen::Matrix<double, 28, 1> raw_variances;
    for (int node = 0; node < 3; ++node) {
        raw_variances(2 * node) = result.voltage[node].magnitude.phase[0].raw_variance;
        raw_variances(2 * node + 1) = result.voltage[node].angle.phase[0].raw_variance;
    }
    for (int terminal = 0; terminal < 4; ++terminal) {
        auto const& branch = result.branch[terminal];
        int const offset = 6 + 4 * terminal;
        raw_variances(offset) = branch.current_magnitude.phase[0].raw_variance;
        raw_variances(offset + 1) = branch.current_angle.phase[0].raw_variance;
        raw_variances(offset + 2) = branch.active_power.phase[0].raw_variance;
        raw_variances(offset + 3) = branch.reactive_power.phase[0].raw_variance;
    }
    for (int node = 0; node < 3; ++node) {
        raw_variances(22 + 2 * node) = result.injection[node].active_power.phase[0].raw_variance;
        raw_variances(23 + 2 * node) = result.injection[node].reactive_power.phase[0].raw_variance;
    }
    bool approximate_injection = false;
    for (auto const& pair : result.pairs)
        approximate_injection =
            approximate_injection || (pair.route != UQPairRoute::stored && pair.route != UQPairRoute::separator);
    REQUIRE(constrained_output.allFinite());
    double max_output_error = 0;
    for (int coordinate = 0; coordinate < 28; ++coordinate) {
        // Only middle-bus injection depends on the missing 0--2 covariance.
        // Approximate/raw invalid injection is a reported discrepancy, never
        // relabeled an exact inverse reference or hidden by exact-zero output.
        if (approximate_injection && (coordinate == 24 || coordinate == 25))
            continue;
        REQUIRE(std::isfinite(raw_variances(coordinate)));
        max_output_error = std::max(max_output_error,
                                    std::abs(raw_variances(coordinate) - constrained_output(coordinate, coordinate)));
    }
    CHECK(max_output_error <= 1e-12);
}

} // namespace constrained_reference

namespace nonlinear_validation {

using namespace power_grid_model;
using namespace power_grid_model::math_solver;

MathModelTopology fixture_topology() {
    MathModelTopology t;
    t.slack_bus = 0;
    t.phase_shift = {0.0, 0.0};
    t.branch_bus_idx = {{0, 1}};
    t.sources_per_bus = {from_sparse, {0, 1, 2}};
    t.shunts_per_bus = {from_sparse, {0, 0, 1}};
    t.load_gens_per_bus = {from_sparse, {0, 0, 0}};
    t.voltage_sensors_per_bus = {from_sparse, {0, 1, 2}};
    t.power_sensors_per_bus = {from_sparse, {0, 0, 0}};
    t.power_sensors_per_source = {from_sparse, {0, 0, 0}};
    t.power_sensors_per_load_gen = {from_sparse, {0}};
    t.power_sensors_per_shunt = {from_sparse, {0, 0}};
    t.power_sensors_per_branch_from = {from_sparse, {0, 0}};
    t.power_sensors_per_branch_to = {from_sparse, {0, 0}};
    t.current_sensors_per_branch_from = {from_sparse, {0, 1}};
    t.current_sensors_per_branch_to = {from_sparse, {0, 0}};
    return t;
}

using Observables = Eigen::Matrix<double, 16, 1>;
Observables outputs(ComplexValueVector<symmetric_t> const& u) {
    DoubleComplex const y{2, -4}, shunt{.03, -.08};
    auto const from = y * (u[0] - u[1]), to = -from;
    auto const sf = u[0] * std::conj(from), st = u[1] * std::conj(to);
    auto const injection0 = sf, injection1 = u[1] * std::conj(to + shunt * u[1]);
    Observables values;
    values << std::abs(u[0]), std::arg(u[0]), std::abs(u[1]), std::arg(u[1]), std::abs(from), std::arg(from), sf.real(),
        sf.imag(), std::abs(to), std::arg(to), st.real(), st.imag(), injection0.real(), injection0.imag(),
        injection1.real(), injection1.imag();
    return values;
}
Observables sigmas(StateEstimationUQResult<symmetric_t> const& result, bool raw = false) {
    Observables values;
    auto value = [raw](UQPhaseValue const& v) { return static_cast<double>(raw ? v.raw_variance : v.sigma); };
    for (int b = 0; b < 2; ++b) {
        values(2 * b) = value(result.voltage[b].magnitude.phase[0]);
        values(2 * b + 1) = value(result.voltage[b].angle.phase[0]);
    }
    for (int b = 0; b < 2; ++b) {
        auto const& branch = result.branch[b];
        values(4 + 4 * b) = value(branch.current_magnitude.phase[0]);
        values(5 + 4 * b) = value(branch.current_angle.phase[0]);
        values(6 + 4 * b) = value(branch.active_power.phase[0]);
        values(7 + 4 * b) = value(branch.reactive_power.phase[0]);
    }
    for (int b = 0; b < 2; ++b) {
        values(12 + 2 * b) = value(result.injection[b].active_power.phase[0]);
        values(13 + 2 * b) = value(result.injection[b].reactive_power.phase[0]);
    }
    return values;
}

template <class Solver> void sample_validation() {
    constexpr std::size_t attempts = 256;
    constexpr double variance = 1e-6;
    DoubleComplex const y{2, -4};
    ComplexValueVector<symmetric_t> const truth{std::polar(1.03, 0.12), std::polar(0.97, -0.03)};
    auto topo = fixture_topology();
    MathModelParam<symmetric_t> param;
    param.branch_param = {{y, -y, -y, y}};
    param.shunt_param = {{0.03, -0.08}};
    YBus<symmetric_t> const y_bus{topo, param};
    StateEstimationInput<symmetric_t> input;
    input.source_status = {1, 1};
    input.shunt_status = {1};
    input.measured_voltage = {{truth[0], variance}, {truth[1], 1.7 * variance}};
    auto const truth_i = y * (truth[0] - truth[1]);
    double const real_var = Solver::is_NRSE_solver ? 0.8 * variance : 0.4 * variance;
    double const imag_var = real_var;
    input.measured_branch_from_current = {
        {.angle_measurement_type = AngleMeasurementType::global_angle,
         .measurement = {.real_component = {.value = truth_i.real(), .variance = real_var},
                         .imag_component = {.value = truth_i.imag(), .variance = imag_var}}}};
    StateEstimationUQRequest request;
    request.voltage = {{0, true, true}, {1, true, true}};
    request.branch = {{0, BranchSide::from, true, true, true, true}, {0, BranchSide::to, true, true, true, true}};
    request.injection = {{0, true, true}, {1, true, true}};
    StateEstimationUQResult<symmetric_t> result;
    StateEstimationUQContext<symmetric_t> context{request, result};
    Solver solver{y_bus, topo};
    common::logging::NoLogger log;
    auto const nominal = solver.run_state_estimation(y_bus, input, 1e-12, 100, log, &context);
    CHECK((outputs(nominal.u) - outputs(truth)).norm() <= 1e-10);
    Eigen::Matrix<double, 16, 16> expected;
    if constexpr (Solver::is_NRSE_solver) {
        Eigen::Vector4d const state{std::arg(truth[0]), std::abs(truth[0]), std::arg(truth[1]), std::abs(truth[1])};
        auto rows = [&](Eigen::Vector4d const& q) {
            auto const current = y * (std::polar(q(1), q(0)) - std::polar(q(3), q(2)));
            Eigen::Matrix<double, 6, 1> value;
            value << q(0), q(1), q(2), q(3), current.real(), current.imag();
            return value;
        };
        Eigen::Matrix<double, 6, 4> jac;
        for (int k = 0; k < 4; ++k) {
            auto l = state, r = state;
            l(k) -= 1e-5;
            r(k) += 1e-5;
            jac.col(k) = (rows(r) - rows(l)) / 2e-5;
        }
        Eigen::Matrix<double, 6, 1> weights;
        weights << 1, 1, 1 / 1.7, 1 / 1.7, 1 / 0.8, 1 / 0.8;
        Eigen::Matrix4d const covariance = (jac.transpose() * weights.asDiagonal() * jac).inverse() * variance;
        Eigen::Matrix<double, 16, 4> output_jacobian;
        for (int k = 0; k < 4; ++k) {
            auto l = state, r = state;
            l(k) -= 1e-5;
            r(k) += 1e-5;
            ComplexValueVector<symmetric_t> const ul{std::polar(l(1), l(0)), std::polar(l(3), l(2))},
                ur{std::polar(r(1), r(0)), std::polar(r(3), r(2))};
            output_jacobian.col(k) = (outputs(ur) - outputs(ul)) / 2e-5;
        }
        expected = output_jacobian * covariance * output_jacobian.transpose();
    } else {
        Eigen::Matrix<DoubleComplex, 3, 2> rows;
        rows << 1, 0, 0, 1, y, -y;
        Eigen::Vector3d weights;
        weights << 1, 1 / 1.7, 1 / 0.8;
        Eigen::Matrix2cd const covariance = (rows.adjoint() * weights.asDiagonal() * rows).inverse() * variance;
        Eigen::Matrix4d rectangular;
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) {
                rectangular(2 * a, 2 * b) = rectangular(2 * a + 1, 2 * b + 1) = covariance(a, b).real() / 2;
                rectangular(2 * a, 2 * b + 1) = -covariance(a, b).imag() / 2;
                rectangular(2 * a + 1, 2 * b) = covariance(a, b).imag() / 2;
            }
        Eigen::Matrix<double, 16, 4> jac;
        for (int k = 0; k < 4; ++k) {
            auto l = truth, r = truth;
            DoubleComplex delta = k % 2 == 0 ? DoubleComplex{1e-5, 0} : DoubleComplex{0, 1e-5};
            l[k / 2] -= delta;
            r[k / 2] += delta;
            jac.col(k) = (outputs(r) - outputs(l)) / 2e-5;
        }
        expected = jac * rectangular * jac.transpose();
    }
    auto const predicted = sigmas(result, true);
    double max_relative_error = 0;
    for (int k = 0; k < 16; ++k)
        max_relative_error = std::max(max_relative_error, std::abs(predicted(k) - expected(k, k)) / expected(k, k));
    REQUIRE(predicted.allFinite());
    REQUIRE(expected.allFinite());
    CHECK(max_relative_error <= 1e-7);
    // Every seeded noisy realization converges with finite sigmas and errors.
    std::mt19937_64 rng{20261030};
    std::normal_distribution<double> noise;
    std::size_t valid = 0;
    for (std::size_t k = 0; k < attempts; ++k) {
        for (int b = 0; b < 2; ++b) {
            double const v = variance * (b == 0 ? 1 : 1.7);
            if constexpr (Solver::is_NRSE_solver)
                input.measured_voltage[b].value = std::polar(std::abs(truth[b]) + std::sqrt(v) * noise(rng),
                                                             std::arg(truth[b]) + std::sqrt(v) * noise(rng));
            else
                input.measured_voltage[b].value =
                    truth[b] + DoubleComplex{std::sqrt(v / 2) * noise(rng), std::sqrt(v / 2) * noise(rng)};
        }
        input.measured_branch_from_current[0].measurement.real_component.value =
            truth_i.real() + std::sqrt(real_var) * noise(rng);
        input.measured_branch_from_current[0].measurement.imag_component.value =
            truth_i.imag() + std::sqrt(imag_var) * noise(rng);
        auto const sample = solver.run_state_estimation(y_bus, input, 1e-12, 100, log, &context);
        if (sigmas(result).allFinite() && outputs(sample.u).allFinite())
            ++valid;
    }
    CHECK(valid == attempts);
}

} // namespace nonlinear_validation

TEST_CASE("UQ integrated sparse Full and separator references") {
    full_reference::reference<full_reference::ILGain, full_reference::ILUnknown, 1>(false);
    full_reference::reference<full_reference::ILGain, full_reference::ILUnknown, 1>(true);
    full_reference::reference<full_reference::NRGain, full_reference::NRUnknown, 2>(false);
    full_reference::reference<full_reference::NRGain, full_reference::NRUnknown, 2>(true);
}
TEST_CASE("UQ independent constrained NRSE reference for all output groups") { constrained_reference::check(); }
TEST_CASE("UQ seeded noisy samples converge with finite sigmas for all output groups") {
    nonlinear_validation::sample_validation<
        power_grid_model::math_solver::IterativeLinearSESolver<power_grid_model::symmetric_t>>();
    nonlinear_validation::sample_validation<
        power_grid_model::math_solver::NewtonRaphsonSESolver<power_grid_model::symmetric_t>>();
}
