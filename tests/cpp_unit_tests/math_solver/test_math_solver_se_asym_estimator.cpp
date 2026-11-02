// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

// Ordinary asymmetric NRSE estimator corrections, checked against finite
// differences of the physical measurement functions.

#include <power_grid_model/math_solver/measured_values.hpp>
#include <power_grid_model/math_solver/newton_raphson_se_solver.hpp>

#include <Eigen/Dense>
#include <doctest/doctest.h>

#include <complex>
#include <utility>
#include <vector>

namespace power_grid_model::math_solver::newton_raphson_se {

struct NRSEAsymTestAccess {
    using Solver = NewtonRaphsonSESolver<asymmetric_t>;
    using Gain = NRSEGainBlock<asymmetric_t>;
    using Rhs = NRSERhs<asymmetric_t>;

    static auto state(ComplexValue<asymmetric_t> const& left, ComplexValue<asymmetric_t> const& right) {
        Solver::NRSEVoltageState result;
        result.ui = left;
        result.uj = right;
        result.ui_ui_conj = vector_outer_product(left, conj(left));
        result.uj_uj_conj = vector_outer_product(right, conj(right));
        result.ui_uj_conj = vector_outer_product(left, conj(right));
        result.uj_ui_conj = vector_outer_product(right, conj(left));
        result.abs_ui_inv = Solver::diagonal_inverse(cabs(left));
        result.abs_uj_inv = Solver::diagonal_inverse(cabs(right));
        return result;
    }
    static auto shunt(Solver& solver, ComplexTensor<asymmetric_t> const& ys, ComplexValue<asymmetric_t> const& voltage,
                      PowerSensorCalcParam<asymmetric_t> const& sensor) {
        Gain gain;
        Rhs rhs;
        solver.process_shunt_measurement(gain, rhs, ys, state(voltage, voltage), sensor);
        return std::pair{gain, rhs};
    }
    static auto local_current(Solver& solver, ComplexTensor<asymmetric_t> const& self,
                              ComplexTensor<asymmetric_t> const& cross, ComplexValue<asymmetric_t> const& left,
                              ComplexValue<asymmetric_t> const& right,
                              CurrentSensorCalcParam<asymmetric_t> const& sensor, bool local) {
        Gain diagonal, other_diagonal, pair;
        Rhs rhs, other_rhs;
        auto const u = state(left, right);
        auto const swapped = state(right, left);
        Gain reverse_pair;
        if (local) {
            solver.process_branch_local_current_measurement(pair, diagonal, rhs, self, cross, u,
                                                            Solver::Order::row_major, sensor);
            solver.process_branch_local_current_measurement(reverse_pair, other_diagonal, other_rhs, self, cross,
                                                            swapped, Solver::Order::column_major, sensor);
        } else {
            solver.process_branch_power_measurement(pair, diagonal, rhs, self, cross, u, Solver::Order::row_major,
                                                    sensor.measurement);
            solver.process_branch_power_measurement(reverse_pair, other_diagonal, other_rhs, self, cross, swapped,
                                                    Solver::Order::column_major, sensor.measurement);
        }
        Eigen::Matrix<double, 12, 12> gain;
        gain.topLeftCorner<6, 6>() = diagonal.g().matrix();
        gain.topRightCorner<6, 6>() = pair.g().matrix();
        gain.bottomLeftCorner<6, 6>() = reverse_pair.g().matrix();
        gain.bottomRightCorner<6, 6>() = other_diagonal.g().matrix();
        Eigen::Matrix<double, 12, 1> residual;
        residual.head<6>() = rhs.matrix().head<6>();
        residual.tail<6>() = other_rhs.matrix().head<6>();
        return std::pair{gain, residual};
    }
    static auto mixed_multipliers(Solver& solver, MeasuredValues<asymmetric_t> const& measurements) {
        solver.x_[0].theta() = 0.0;
        solver.x_[0].v() = 1.0;
        solver.x_[0].phi_p() = 0.0;
        solver.x_[0].phi_q() = 0.0;
        solver.delta_x_rhs_[0].clear();
        solver.delta_x_rhs_[0].phi_p() = RealValue<asymmetric_t>{0.2, 0.4, 0.6};
        solver.delta_x_rhs_[0].phi_q() = RealValue<asymmetric_t>{-0.3, -0.6, -0.9};
        ComplexValueVector<asymmetric_t> voltage{ComplexValue<asymmetric_t>{1.0}};
        solver.iterate_unknown(voltage, measurements);
        return solver.x_[0];
    }
};
} // namespace power_grid_model::math_solver::newton_raphson_se

namespace power_grid_model::math_solver {
namespace {
using Asym = asymmetric_t;
using Voltage = ComplexValue<Asym>;
using Access = newton_raphson_se::NRSEAsymTestAccess;

MathModelTopology asym_topology(Idx n, std::vector<BranchIdx> branches = {}) {
    MathModelTopology result;
    result.slack_bus = n - 1;
    result.phase_shift.resize(n, 0.0);
    result.branch_bus_idx = std::move(branches);
    IdxVector empty(n + 1, 0), each(n + 1);
    for (Idx bus = 0; bus <= n; ++bus)
        each[bus] = bus;
    result.sources_per_bus = {from_sparse, each};
    result.voltage_sensors_per_bus = {from_sparse, each};
    result.power_sensors_per_bus = {from_sparse, each};
    result.load_gens_per_bus = {from_sparse, empty};
    result.shunts_per_bus = {from_sparse, empty};
    result.power_sensors_per_source = {from_sparse, empty};
    result.power_sensors_per_load_gen = {from_sparse, {0}};
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
    return {std::polar(1.1 + offset, 0.13), std::polar(0.72 + offset, -1.9), std::polar(1.35 + offset, 2.2)};
}
Eigen::Matrix<double, 6, 1> polar_state(Voltage const& u) {
    Eigen::Matrix<double, 6, 1> result;
    result.head<3>() = arg(u).matrix();
    result.tail<3>() = cabs(u).matrix();
    return result;
}
Voltage polar_voltage(Eigen::Matrix<double, 6, 1> const& state) {
    Voltage result;
    for (int phase = 0; phase != 3; ++phase)
        result(phase) = std::polar(state(phase + 3), state(phase));
    return result;
}
Eigen::Matrix<double, 6, 1> real_axes(Voltage const& value) {
    Eigen::Matrix<double, 6, 1> result;
    result.head<3>() = real(value).matrix();
    result.tail<3>() = imag(value).matrix();
    return result;
}
template <int N, class Function>
Eigen::Matrix<double, 6, N> finite_jacobian(Eigen::Matrix<double, N, 1> x, Function const& function) {
    Eigen::Matrix<double, 6, N> result;
    for (int col = 0; col != N; ++col) {
        auto plus = x, minus = x;
        plus(col) += 1e-6;
        minus(col) -= 1e-6;
        result.col(col) = (function(plus) - function(minus)) / 2e-6;
    }
    return result;
}
ComplexTensor<Asym> coupled_admittance() {
    ComplexTensor<Asym> result;
    for (int row = 0; row != 3; ++row)
        for (int col = 0; col != 3; ++col)
            result(row, col) = {row == col ? 2.0 + row : 0.4 + 0.1 * (row + col), -0.2 - 0.15 * (row + col)};
    return result;
}
PowerSensorCalcParam<Asym> sensor() {
    return {.real_component = {.value = RealValue<Asym>{0.5, -0.4, 0.8}, .variance = RealValue<Asym>{0.3, 0.7, 1.4}},
            .imag_component = {.value = RealValue<Asym>{-0.2, 0.6, 0.1}, .variance = RealValue<Asym>{0.6, 1.2, 0.4}}};
}
Eigen::Matrix<double, 6, 1> weights(PowerSensorCalcParam<Asym> const& measurement) {
    Eigen::Matrix<double, 6, 1> result;
    result.head<3>() = measurement.real_component.variance.inverse().matrix();
    result.tail<3>() = measurement.imag_component.variance.inverse().matrix();
    return result;
}
} // namespace

TEST_CASE("asymmetric NRSE shunt Jacobian uses the measured phase radius") {
    auto const topo = asym_topology(1);
    YBus<Asym> const y_bus{topo, asym_parameters(0)};
    Access::Solver solver{y_bus, topo};
    auto const u = phasors();
    auto const ys = coupled_admittance();
    auto const measured = sensor();
    auto const function = [&](Eigen::Matrix<double, 6, 1> const& x) {
        auto const voltage = polar_voltage(x);
        return real_axes(Voltage{-voltage * conj(dot(ys, voltage))});
    };
    auto const x = polar_state(u);
    auto const jacobian = finite_jacobian<6>(x, function);
    auto [gain, rhs] = Access::shunt(solver, ys, u, measured);
    auto const expected = (jacobian.transpose() * weights(measured).asDiagonal() * jacobian).eval();
    auto const expected_rhs =
        (jacobian.transpose() * weights(measured).asDiagonal() * (real_axes(measured.value()) - function(x))).eval();
    CHECK((gain.g().matrix() - expected).norm() < 1e-7 * expected.norm());
    CHECK((rhs.matrix().head<6>() - expected_rhs).norm() < 1e-7 * expected_rhs.norm());
}

TEST_CASE("asymmetric NRSE coupled current and power use correct phase normalization and weighted transposes") {
    auto const topo = asym_topology(2, {{0, 1}});
    YBus<Asym> const y_bus{topo, asym_parameters(1)};
    Access::Solver solver{y_bus, topo};
    auto const self = coupled_admittance();
    ComplexTensor<Asym> const cross = -0.8 * self;
    CurrentSensorCalcParam<Asym> const measured{.angle_measurement_type = AngleMeasurementType::local_angle,
                                                .measurement = sensor()};
    for (bool local_current : {false, true}) {
        for (bool reverse : {false, true}) {
            CAPTURE(local_current);
            CAPTURE(reverse);
            Voltage const left = reverse ? phasors(.2) : phasors();
            Voltage const right = reverse ? phasors() : phasors(.2);
            Eigen::Matrix<double, 12, 1> x;
            x.head<6>() = polar_state(left);
            x.tail<6>() = polar_state(right);
            auto function = [&](Eigen::Matrix<double, 12, 1> const& state) {
                auto const local = polar_voltage(state.head<6>());
                auto const remote = polar_voltage(state.tail<6>());
                Voltage const current = dot(self, local) + dot(cross, remote);
                Voltage const power = local * conj(current);
                return real_axes(Voltage{local_current ? power / cabs(local) : power});
            };
            auto const jacobian = finite_jacobian<12>(x, function);
            auto const [gain, rhs] = Access::local_current(solver, self, cross, left, right, measured, local_current);
            auto const expected = (jacobian.transpose() * weights(measured.measurement).asDiagonal() * jacobian).eval();
            auto const expected_rhs = (jacobian.transpose() * weights(measured.measurement).asDiagonal() *
                                       (real_axes(measured.measurement.value()) - function(x)))
                                          .eval();
            CHECK((gain - expected).norm() < 1e-7 * expected.norm());
            CHECK((rhs - expected_rhs).norm() < 1e-7 * expected_rhs.norm());
        }
    }
}

TEST_CASE("asymmetric NRSE accumulates exact multipliers independently per phase") {
    auto const topo = asym_topology(1);
    YBus<Asym> const y_bus{topo, asym_parameters(0)};
    StateEstimationInput<Asym> input;
    input.source_status = {1};
    input.measured_voltage = {{phasors(), 0.2}};
    input.measured_bus_injection = {
        {.real_component = {.value = RealValue<Asym>{0.0}, .variance = RealValue<Asym>{0.0, 0.4, 0.0}},
         .imag_component = {.value = RealValue<Asym>{0.0}, .variance = RealValue<Asym>{0.3, 0.0, 0.9}}}};
    MeasuredValues<Asym> const measurements{topo, input};
    Access::Solver solver{y_bus, topo};
    auto const result = Access::mixed_multipliers(solver, measurements);
    CHECK(result(6) == doctest::Approx(.2));
    CHECK(result(7) == 0.0);
    CHECK(result(8) == doctest::Approx(.6));
    CHECK(result(9) == 0.0);
    CHECK(result(10) == doctest::Approx(-.6));
    CHECK(result(11) == 0.0);
}

TEST_CASE("asymmetric injection aggregation preserves exact phases and combines noisy phases") {
    auto topo = asym_topology(1);
    topo.power_sensors_per_source = {from_sparse, {0, 1}};
    StateEstimationInput<Asym> input;
    input.source_status = {1};
    input.measured_voltage = {{phasors(), .2}};
    auto appliance = sensor();
    appliance.real_component.variance(0) = 0.0;
    // Appliance aggregation intentionally keeps its ordinary positive sensor
    // policy. Construct a mixed injection from a direct prepared constraint and
    // finite measured appliance, as accepted by the internal solver contract.
    auto direct = appliance;
    appliance.real_component.variance(0) = .6;
    input.measured_source_power = {appliance};
    input.measured_bus_injection = {direct};
    MeasuredValues<Asym> const measurements{topo, input};
    // Prepared variances are scaled by the smallest finite positive input variance.
    double const normalization = .2 / measurements.voltage_var(0);
    REQUIRE(measurements.has_bus_injection(0));
    auto const& combined = measurements.bus_injection(0);
    CHECK(combined.real_component.variance(0) == 0.0);
    CHECK(combined.real_component.value(0) == direct.real_component.value(0));
    for (int phase : {1, 2}) {
        CHECK(combined.real_component.variance(phase) * normalization ==
              doctest::Approx(direct.real_component.variance(phase) / 2));
    }
    for (int phase = 0; phase != 3; ++phase) {
        CHECK(combined.imag_component.variance(phase) * normalization ==
              doctest::Approx(direct.imag_component.variance(phase) / 2));
    }
    // An empty bus is an exact physical injection in every phase. Noisy direct
    // measurements do not turn any of these exact constraints into noise.
    topo.sources_per_bus = {from_sparse, {0, 0}};
    topo.power_sensors_per_source = {from_sparse, {0}};
    input.source_status.clear();
    input.measured_source_power.clear();
    input.measured_bus_injection = {sensor()};
    MeasuredValues<Asym> const empty_bus{topo, input};
    REQUIRE(empty_bus.has_bus_injection(0));
    CHECK(all_zero(empty_bus.bus_injection(0).real_component.variance));
    CHECK(all_zero(empty_bus.bus_injection(0).imag_component.variance));
}
} // namespace power_grid_model::math_solver
