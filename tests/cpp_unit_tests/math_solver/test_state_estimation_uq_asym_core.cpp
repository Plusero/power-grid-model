// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#include <power_grid_model/common/dummy_logging.hpp>
#include <power_grid_model/math_solver/iterative_linear_se_solver.hpp>
#include <power_grid_model/math_solver/newton_raphson_se_solver.hpp>

#include <Eigen/LU>
#include <doctest/doctest.h>

#include <complex>
#include <vector>

namespace power_grid_model::math_solver {
namespace {
using namespace state_estimation_uq;
using Asym = asymmetric_t;
using Voltage = ComplexValue<Asym>;

MathModelTopology asym_topology(Idx n, std::vector<BranchIdx> branches = {}) {
    MathModelTopology result;
    result.slack_bus = n - 1;
    result.phase_shift.resize(n, 0.0);
    result.branch_bus_idx = std::move(branches);
    result.is_radial = result.n_branch() == n - 1;
    IdxVector empty(n + 1, 0), each(n + 1);
    for (Idx bus = 0; bus <= n; ++bus) {
        each[bus] = bus;
    }
    IdxVector source(n + 1, 0), loads = each;
    source[n] = 1;
    loads[n] = n - 1;
    result.sources_per_bus = {from_sparse, source};
    result.voltage_sensors_per_bus = {from_sparse, each};
    result.power_sensors_per_bus = {from_sparse, each};
    result.load_gens_per_bus = {from_sparse, loads};
    result.load_gen_type.assign(n - 1, LoadGenType::const_pq);
    result.shunts_per_bus = {from_sparse, empty};
    result.power_sensors_per_source = {from_sparse, {0, 0}};
    result.power_sensors_per_load_gen = {from_sparse, IdxVector(n, 0)};
    result.power_sensors_per_shunt = {from_sparse, {0}};
    IdxVector no_branch(result.branch_bus_idx.size() + 1, 0);
    result.power_sensors_per_branch_from = {from_sparse, no_branch};
    result.power_sensors_per_branch_to = {from_sparse, no_branch};
    return result;
}
MathModelParam<Asym> asym_parameters(Idx branches) {
    MathModelParam<Asym> result;
    result.branch_param.resize(branches, {ComplexTensor<Asym>{1.0}, ComplexTensor<Asym>{-1.0},
                                          ComplexTensor<Asym>{-1.0}, ComplexTensor<Asym>{1.0}});
    return result;
}
Voltage phasors(double offset = 0.0) {
    double const angle = offset - 0.03;
    return {std::polar(1.01 + offset, angle), std::polar(0.99 + offset, angle - 2.0 * pi / 3.0),
            std::polar(1.0 + offset, angle + 2.0 * pi / 3.0)};
}
ComplexTensor<Asym> coupled_admittance() {
    return ComplexTensor<Asym>{DoubleComplex{4.0, -12.0}, DoubleComplex{0.3, -0.8}};
}
PowerSensorCalcParam<Asym> sensor(Voltage const& power, double variance_scale = 1.0) {
    return {.real_component = {.value = real(power), .variance = RealValue<Asym>{0.3, 0.7, 1.4} * variance_scale},
            .imag_component = {.value = imag(power), .variance = RealValue<Asym>{0.6, 1.2, 0.4} * variance_scale}};
}

// Voltage weights and noisy bus injections produce the same augmented block
// structure as SE: [G, Q^H; Q, -R], with bus-local measurement variances.
template <class Gain, int N>
std::vector<Gain> augmented_blocks(YBus<Asym> const& y_bus, MeasuredValues<Asym> const& measured,
                                   ComplexValueVector<Asym> const& u, ComplexValueVector<Asym> const& injection,
                                   bool relative, Idx virtual_bus) {
    constexpr bool uses_complex_state = Eigen::NumTraits<typename Gain::Scalar>::IsComplex;
    std::vector<Gain> blocks(y_bus.nnz_lu());
    for (Idx row = 0; row != y_bus.size(); ++row) {
        double const voltage_weight = 1.0 / measured.voltage_var(row);
        auto const& p_variance = measured.bus_injection(row).real_component.variance;
        auto const& q_variance = measured.bus_injection(row).imag_component.variance;
        for (Idx k = y_bus.row_indptr_lu()[row]; k != y_bus.row_indptr_lu()[row + 1]; ++k) {
            Idx const col = y_bus.col_indices_lu()[k];
            Idx const admittance_index = y_bus.map_lu_y_bus()[k];
            if (admittance_index < 0) {
                continue;
            }
            auto& block = blocks[k];
            auto const& y = y_bus.admittance()[admittance_index];
            for (int phase = 0; phase != 3; ++phase) {
                if (row == col) {
                    if constexpr (uses_complex_state) {
                        block(phase, phase) = voltage_weight;
                        block(N + phase, N + phase) = -(p_variance[phase] + q_variance[phase]);
                    } else {
                        block(phase, phase) = (!relative || row == virtual_bus) ? voltage_weight : 0.0;
                        block(3 + phase, 3 + phase) = voltage_weight;
                        block(N + phase, N + phase) = -p_variance[phase];
                        block(N + 3 + phase, N + 3 + phase) = -q_variance[phase];
                    }
                }
                for (int other = 0; other != 3; ++other) {
                    if constexpr (uses_complex_state) {
                        // ILSE injection-current constraints: Q = Y, R = -(var P + var Q).
                        block(N + phase, other) = y(phase, other);
                    } else {
                        // dS_i/dtheta_j = i*(delta_ij*S_i - u_i*conj(Y_ij*u_j));
                        // dS_i/d|u_j| = (delta_ij*S_i + u_i*conj(Y_ij*u_j))/|u_j|.
                        DoubleComplex const product = u[row](phase) * std::conj(y(phase, other) * u[col](other));
                        DoubleComplex const diagonal = row == col && phase == other ? injection[row](phase) : 0.0;
                        DoubleComplex const theta = 1.0i * (diagonal - product);
                        DoubleComplex const magnitude = (diagonal + product) / std::abs(u[col](other));
                        block(N + phase, other) = theta.real();
                        block(N + phase, 3 + other) = magnitude.real();
                        block(N + 3 + phase, other) = theta.imag();
                        block(N + 3 + phase, 3 + other) = magnitude.imag();
                    }
                }
            }
        }
    }
    for (Idx k = 0; k != y_bus.nnz_lu(); ++k) {
        blocks[k].matrix().template block<N, N>(0, N) =
            blocks[y_bus.lu_transpose_entry()[k]].matrix().template block<N, N>(N, 0).adjoint();
    }
    return blocks;
}

// ILSE angle variance of one bus phase from the dense augmented inverse: an explicit
// rectangular proper covariance and the analytic derivative of the output angle,
// including its common slack phase-a reference when the angle is relative.
template <class Dense>
double ilse_angle_variance(Dense const& inverse, int size, double alpha, ComplexValueVector<Asym> const& u, Idx bus,
                           int phase, bool relative) {
    Eigen::Matrix<double, 24, 24> rectangular;
    for (int i = 0; i != 12; ++i) {
        for (int j = 0; j != 12; ++j) {
            DoubleComplex const value = alpha * inverse((i / 3) * size + i % 3, (j / 3) * size + j % 3);
            rectangular(i, j) = .5 * value.real();
            rectangular(i, j + 12) = -.5 * value.imag();
            rectangular(i + 12, j) = .5 * value.imag();
            rectangular(i + 12, j + 12) = .5 * value.real();
        }
    }
    Eigen::Matrix<double, 24, 1> derivative = Eigen::Matrix<double, 24, 1>::Zero();
    auto add = [&](Idx at_bus, int at_phase, double sign) {
        DoubleComplex const voltage = u[at_bus](at_phase);
        derivative(3 * at_bus + at_phase) += sign * -voltage.imag() / std::norm(voltage);
        derivative(12 + 3 * at_bus + at_phase) += sign * voltage.real() / std::norm(voltage);
    };
    add(bus, phase, 1.0);
    if (relative) {
        add(3, 0, -1.0);
    }
    return (derivative.transpose() * rectangular * derivative)(0, 0);
}

template <class Gain, class Unknown, int N>
void workspace_dense_reference(bool relative, Idx virtual_bus, double alpha) {
    constexpr int size = Gain::RowsAtCompileTime;
    using Scalar = typename Gain::Scalar;
    constexpr bool uses_complex_state = Eigen::NumTraits<Scalar>::IsComplex;
    using Dense = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    auto topo = asym_topology(4, {{0, 1}, {1, 2}, {2, 3}, {3, 0}});
    topo.fill_in = {{1, 3}};
    auto param = asym_parameters(4);
    auto const admittance = coupled_admittance();
    for (auto& branch : param.branch_param) {
        branch = {admittance, ComplexTensor<Asym>{-admittance}, ComplexTensor<Asym>{-admittance}, admittance};
    }
    YBus<Asym> const y_bus{topo, param};
    ComplexValueVector<Asym> const u{phasors(), phasors(.01), phasors(.02), phasors(.03)};
    auto const injection = y_bus.calculate_injection(u);
    StateEstimationInput<Asym> input;
    input.source_status = {1};
    input.load_gen_status = {1, 1, 1};
    for (Idx bus = 0; bus != 4; ++bus) {
        Voltage measured_voltage = u[bus];
        if (relative) {
            for (int phase = 0; phase != 3; ++phase) {
                measured_voltage(phase) = {std::abs(measured_voltage(phase)), nan};
            }
        }
        input.measured_voltage.push_back({measured_voltage, alpha});
        input.measured_bus_injection.push_back(sensor(injection[bus], 5.0 * alpha));
    }
    MeasuredValues<Asym> const measured{topo, input};
    CHECK(measured.variance_normalization() == doctest::Approx(alpha));
    double const weight = virtual_bus >= 0 ? 1.0 / measured.voltage_var(virtual_bus) : 0.0;
    auto const blocks = augmented_blocks<Gain, N>(y_bus, measured, u, injection, relative, virtual_bus);
    Dense dense = Dense::Zero(4 * size, 4 * size);
    for (Idx row = 0; row != 4; ++row) {
        for (Idx k = y_bus.row_indptr_lu()[row]; k != y_bus.row_indptr_lu()[row + 1]; ++k) {
            dense.block(row * size, y_bus.col_indices_lu()[k] * size, size, size) = blocks[k].matrix();
        }
    }
    SparseCovarianceWorkspace<Gain, Unknown, N> const covariance{y_bus,     blocks,      alpha, u,
                                                                 !relative, virtual_bus, weight};
    Dense const inverse = dense.fullPivLu().inverse();
    CHECK((dense * inverse - Dense::Identity(dense.rows(), dense.cols())).norm() < 2e-12);
    auto state = [&](Idx row, Idx col) -> Eigen::Matrix<Scalar, N, N> {
        Eigen::Matrix<Scalar, N, N> value = alpha * inverse.block(row * size, col * size, N, N);
        if constexpr (!uses_complex_state) {
            if (virtual_bus >= 0) {
                value -= alpha * weight * inverse.block(row * size, virtual_bus * size, N, 3) *
                         inverse.block(virtual_bus * size, col * size, 3, N);
            }
        }
        return value;
    };
    for (Idx row = 0; row != 4; ++row) {
        for (Idx k = y_bus.row_indptr_lu()[row]; k != y_bus.row_indptr_lu()[row + 1]; ++k) {
            Idx const col = y_bus.col_indices_lu()[k];
            CHECK((covariance.augmented_matrix[k].matrix() - inverse.block(row * size, col * size, size, size)).norm() <
                  2e-12);
            auto expected = state(row, col);
            if constexpr (!uses_complex_state) {
                if (relative) {
                    Eigen::Matrix<double, N, 1> d = Eigen::Matrix<double, N, 1>::Zero();
                    d.template head<3>().setOnes();
                    expected -= state(row, 3).col(0) * d.transpose();
                    expected -= d * state(3, col).row(0);
                    expected += state(3, 3)(0, 0) * d * d.transpose();
                }
            }
            auto const actual = covariance.calculate_state_covariance_block(row, col);
            CHECK(actual.valid);
            CHECK((actual.value - expected).norm() < 3e-12 * alpha);
            if constexpr (!uses_complex_state) {
                auto const physical = covariance.physical_block(row, col);
                auto const expected_physical =
                    (covariance.rectangular_jacobian(row) * expected * covariance.rectangular_jacobian(col).transpose())
                        .eval();
                CHECK(physical.valid);
                CHECK((physical.value - expected_physical).norm() < 4e-12 * alpha);
            }
        }
    }
    StateEstimationUQRequest request;
    request.voltage = {{0, true, true}, {3, true, true}};
    auto const outputs = calculate_voltage_uq_results(covariance, request);
    for (std::size_t index = 0; index != request.voltage.size(); ++index) {
        Idx const bus = request.voltage[index].bus;
        auto const& output = outputs.voltage[index];
        for (int phase = 0; phase != 3; ++phase) {
            CHECK(output.angle.phase[phase].raw_status == UQVarianceStatus::valid);
            CHECK(output.magnitude.phase[phase].raw_status == UQVarianceStatus::valid);
            if constexpr (uses_complex_state) {
                CHECK(
                    output.angle.phase[phase].raw_variance ==
                    doctest::Approx(ilse_angle_variance(inverse, size, alpha, u, bus, phase, relative)).epsilon(2e-10));
                CHECK(output.magnitude.phase[phase].raw_variance ==
                      doctest::Approx(state(bus, bus)(phase, phase).real() / 2).epsilon(2e-10));
            }
        }
    }
    if constexpr (!uses_complex_state) {
        if (relative) {
            CHECK(outputs.voltage[1].angle.phase[0].raw_variance == 0.0);
            CHECK(outputs.voltage[1].angle.phase[1].raw_variance > 0.0);
            CHECK(outputs.voltage[1].angle.phase[2].raw_variance > 0.0);
        }
    }
}
} // namespace

TEST_CASE("asymmetric full selected blocks and voltage maps match dense augmented references") {
    for (double const alpha : {1e-8, 1.0, 1e8}) {
        for (bool const relative : {false, true}) {
            workspace_dense_reference<iterative_linear_se::ILSEGainBlock<Asym>, iterative_linear_se::ILSEUnknown<Asym>,
                                      3>(relative, -1, alpha);
            workspace_dense_reference<newton_raphson_se::NRSEGainBlock<Asym>, newton_raphson_se::NRSEUnknown<Asym>, 6>(
                relative, relative ? 3 : -1, alpha);
        }
    }
}

TEST_CASE_TEMPLATE("asymmetric returned-state voltage UQ preserves coupled estimates", Solver,
                   IterativeLinearSESolver<Asym>, NewtonRaphsonSESolver<Asym>) {
    auto const topo = asym_topology(2, {{0, 1}});
    auto param = asym_parameters(1);
    auto const admittance = coupled_admittance();
    param.branch_param[0] = {admittance, ComplexTensor<Asym>{-admittance}, ComplexTensor<Asym>{-admittance},
                             admittance};
    YBus<Asym> const y_bus{topo, param};
    // A small balanced voltage drop gives nonzero coupled three-phase flow.
    // The common phase-a reference is the only unobservable physical angle.
    Voltage const balanced = exp(1.0i * RealValue<Asym>{0.0, -2.0 * pi / 3.0, 2.0 * pi / 3.0});
    ComplexValueVector<Asym> const voltage{Voltage{1.01 * std::polar(1.0, .02) * balanced}, Voltage{.99 * balanced}};
    for (bool const absolute : {false, true}) {
        CAPTURE(absolute);
        StateEstimationInput<Asym> input;
        input.source_status = {1};
        input.load_gen_status = {1};
        for (int bus = 0; bus != 2; ++bus) {
            Voltage measured_voltage = voltage[bus];
            if (!absolute) {
                for (int phase = 0; phase != 3; ++phase) {
                    measured_voltage(phase) = {std::abs(measured_voltage(phase)), nan};
                }
            }
            input.measured_voltage.push_back({measured_voltage, .2});
            Voltage const current = dot(admittance, Voltage{voltage[bus] - voltage[1 - bus]});
            Voltage const power = voltage[bus] * conj(current);
            input.measured_bus_injection.push_back(sensor(power));
        }
        Solver solver{y_bus, topo};
        common::logging::NoLogger log;
        auto const ordinary = solver.run_state_estimation(y_bus, input, 1e-10, 100, log);
        StateEstimationUQRequest request;
        request.voltage = {{0, true, true}, {1, true, true}};
        StateEstimationUQResult<Asym> result;
        StateEstimationUQContext<Asym> context{request, result};
        auto const enabled = solver.run_state_estimation(y_bus, input, 1e-10, 100, log, &context);
        REQUIRE(result.voltage.size() == 2);
        for (int bus = 0; bus != 2; ++bus) {
            CHECK((ordinary.u[bus] == enabled.u[bus]).all());
            for (int phase = 0; phase != 3; ++phase) {
                CHECK(result.voltage[bus].magnitude.phase[phase].raw_status == UQVarianceStatus::valid);
                CHECK(result.voltage[bus].angle.phase[phase].raw_status == UQVarianceStatus::valid);
            }
        }
        if (!absolute) {
            CHECK(result.voltage[1].angle.phase[0].sigma == 0.0);
            CHECK(result.voltage[1].angle.phase[1].sigma > 0.0);
            CHECK(result.voltage[1].angle.phase[2].sigma > 0.0);
        }
    }
}

} // namespace power_grid_model::math_solver
