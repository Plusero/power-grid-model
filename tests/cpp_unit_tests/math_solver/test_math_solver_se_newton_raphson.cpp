// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#include "test_math_solver_se.hpp" // NOLINT(misc-include-cleaner)

#include <power_grid_model/math_solver/newton_raphson_se_solver.hpp> // NOLINT(misc-include-cleaner)

#include <power_grid_model/common/common.hpp>

#include <doctest/doctest.h>

#include <array>
#include <cmath>

TYPE_TO_STRING_AS("NewtonRaphsonSESolver<symmetric_t>",
                  power_grid_model::math_solver::NewtonRaphsonSESolver<power_grid_model::symmetric_t>);
TYPE_TO_STRING_AS("NewtonRaphsonSESolver<asymmetric_t>",
                  power_grid_model::math_solver::NewtonRaphsonSESolver<power_grid_model::asymmetric_t>);

namespace power_grid_model::math_solver {
TEST_CASE_TEMPLATE_INVOKE(test_math_solver_se_id, NewtonRaphsonSESolver<symmetric_t>);
TEST_CASE_TEMPLATE_INVOKE(test_math_solver_se_id, NewtonRaphsonSESolver<asymmetric_t>);
TEST_CASE_TEMPLATE_INVOKE(test_math_solver_se_zero_variance_id, NewtonRaphsonSESolver<symmetric_t>);
TEST_CASE_TEMPLATE_INVOKE(test_math_solver_se_measurements_id, NewtonRaphsonSESolver<symmetric_t>);

TEST_CASE("Newton-Raphson SE uncertainty scales with measurement variance and reuses the solver") {
    constexpr double error_tolerance{1e-10};
    constexpr Idx num_iter{20};
    constexpr double variance_scale{4.0};

    SESolverTestGrid<symmetric_t> const grid;
    auto const topo = grid.se_topo_power_sensors();
    YBus<symmetric_t> const y_bus{topo, grid.param()};
    NewtonRaphsonSESolver<symmetric_t> solver{y_bus, topo};
    auto log = get_logger();

    auto const input = grid.se_input_angle();
    auto scaled_input = input;
    auto const scale_finite_variance = [](double& variance) {
        if (std::isfinite(variance)) {
            variance *= variance_scale;
        }
    };
    auto const scale_power_variances = [&scale_finite_variance](auto& measurements) {
        for (auto& measurement : measurements) {
            scale_finite_variance(measurement.real_component.variance);
            scale_finite_variance(measurement.imag_component.variance);
        }
    };

    for (auto& measurement : scaled_input.measured_voltage) {
        scale_finite_variance(measurement.variance);
    }
    scale_power_variances(scaled_input.measured_source_power);
    scale_power_variances(scaled_input.measured_load_gen_power);
    scale_power_variances(scaled_input.measured_shunt_power);
    scale_power_variances(scaled_input.measured_branch_from_power);
    scale_power_variances(scaled_input.measured_branch_to_power);
    scale_power_variances(scaled_input.measured_bus_injection);

    auto const output_without_uncertainty =
        solver.run_state_estimation(y_bus, input, error_tolerance, num_iter, false, log);
    auto const output = solver.run_state_estimation(y_bus, input, error_tolerance, num_iter, true, log);
    auto const scaled_output = solver.run_state_estimation(y_bus, scaled_input, error_tolerance, num_iter, true, log);

    assert_output(output, output_without_uncertainty);
    assert_output(scaled_output, output);
    REQUIRE(output_without_uncertainty.bus_uncertainty.empty());
    REQUIRE(output.bus_uncertainty.size() == output.u.size());
    REQUIRE(scaled_output.bus_uncertainty.size() == scaled_output.u.size());
    REQUIRE(output.branch.size() == scaled_output.branch.size());

    auto const check_scaled_sigma = [](double sigma, double scaled_sigma) {
        CAPTURE(sigma);
        CAPTURE(scaled_sigma);
        REQUIRE(std::isfinite(sigma));
        REQUIRE(std::isfinite(scaled_sigma));
        CHECK(sigma >= 0.0);
        CHECK(scaled_sigma >= 0.0);
        CHECK(scaled_sigma == doctest::Approx(2.0 * sigma).epsilon(1e-10));
    };

    for (Idx bus = 0; bus != std::ssize(output.bus_uncertainty); ++bus) {
        auto const& uncertainty = output.bus_uncertainty[bus];
        auto const& scaled_uncertainty = scaled_output.bus_uncertainty[bus];
        check_scaled_sigma(uncertainty.u_sigma, scaled_uncertainty.u_sigma);
        check_scaled_sigma(uncertainty.u_angle_sigma, scaled_uncertainty.u_angle_sigma);
        check_scaled_sigma(uncertainty.p_sigma, scaled_uncertainty.p_sigma);
        check_scaled_sigma(uncertainty.q_sigma, scaled_uncertainty.q_sigma);
    }

    for (Idx branch = 0; branch != std::ssize(output.branch); ++branch) {
        auto const& branch_output = output.branch[branch];
        auto const& scaled_branch_output = scaled_output.branch[branch];
        check_scaled_sigma(branch_output.p_f_sigma, scaled_branch_output.p_f_sigma);
        check_scaled_sigma(branch_output.q_f_sigma, scaled_branch_output.q_f_sigma);
        check_scaled_sigma(branch_output.i_f_sigma, scaled_branch_output.i_f_sigma);
        check_scaled_sigma(branch_output.p_t_sigma, scaled_branch_output.p_t_sigma);
        check_scaled_sigma(branch_output.q_t_sigma, scaled_branch_output.q_t_sigma);
        check_scaled_sigma(branch_output.i_t_sigma, scaled_branch_output.i_t_sigma);
    }
}

TEST_CASE("Asymmetric Newton-Raphson SE uncertainty supports phase coupling and local currents") {
    constexpr double error_tolerance{1e-10};
    constexpr Idx num_iter{20};

    SESolverTestGrid<asymmetric_t> const grid;
    auto log = get_logger();

    auto const scale_variances = [](StateEstimationInput<asymmetric_t>& input) {
        for (auto& measurement : input.measured_voltage) {
            measurement.variance *= 4.0;
        }
        auto const scale_power = [](auto& measurements) {
            for (auto& measurement : measurements) {
                measurement.real_component.variance *= 4.0;
                measurement.imag_component.variance *= 4.0;
            }
        };
        scale_power(input.measured_source_power);
        scale_power(input.measured_load_gen_power);
        scale_power(input.measured_shunt_power);
        scale_power(input.measured_branch_from_power);
        scale_power(input.measured_branch_to_power);
        scale_power(input.measured_bus_injection);
        for (auto& measurement : input.measured_branch_from_current) {
            measurement.measurement.real_component.variance *= 4.0;
            measurement.measurement.imag_component.variance *= 4.0;
        }
        for (auto& measurement : input.measured_branch_to_current) {
            measurement.measurement.real_component.variance *= 4.0;
            measurement.measurement.imag_component.variance *= 4.0;
        }
    };

    auto const check_scaled = [](RealValue<asymmetric_t> const& sigma, RealValue<asymmetric_t> const& scaled_sigma) {
        for (Idx phase = 0; phase != 3; ++phase) {
            CAPTURE(phase);
            REQUIRE(std::isfinite(sigma(phase)));
            REQUIRE(std::isfinite(scaled_sigma(phase)));
            CHECK(sigma(phase) >= 0.0);
            CHECK(scaled_sigma(phase) == doctest::Approx(2.0 * sigma(phase)).epsilon(1e-9));
        }
    };

    auto const check_outputs = [&check_scaled](SolverOutput<asymmetric_t> const& output,
                                               SolverOutput<asymmetric_t> const& scaled_output) {
        for (Idx bus = 0; bus != std::ssize(output.bus_uncertainty); ++bus) {
            check_scaled(output.bus_uncertainty[bus].u_sigma, scaled_output.bus_uncertainty[bus].u_sigma);
            check_scaled(output.bus_uncertainty[bus].u_angle_sigma, scaled_output.bus_uncertainty[bus].u_angle_sigma);
            check_scaled(output.bus_uncertainty[bus].p_sigma, scaled_output.bus_uncertainty[bus].p_sigma);
            check_scaled(output.bus_uncertainty[bus].q_sigma, scaled_output.bus_uncertainty[bus].q_sigma);
        }
        for (Idx branch = 0; branch != std::ssize(output.branch); ++branch) {
            check_scaled(output.branch[branch].p_f_sigma, scaled_output.branch[branch].p_f_sigma);
            check_scaled(output.branch[branch].q_f_sigma, scaled_output.branch[branch].q_f_sigma);
            check_scaled(output.branch[branch].i_f_sigma, scaled_output.branch[branch].i_f_sigma);
            check_scaled(output.branch[branch].p_t_sigma, scaled_output.branch[branch].p_t_sigma);
            check_scaled(output.branch[branch].q_t_sigma, scaled_output.branch[branch].q_t_sigma);
            check_scaled(output.branch[branch].i_t_sigma, scaled_output.branch[branch].i_t_sigma);
        }
    };

    SUBCASE("Power measurements") {
        auto const topo = grid.se_topo_power_sensors();
        YBus<asymmetric_t> const y_bus{topo, grid.param()};
        NewtonRaphsonSESolver<asymmetric_t> solver{y_bus, topo};
        auto input = grid.se_input_angle();
        input.measured_branch_from_power.front().real_component.variance = RealValue<asymmetric_t>{0.25, 0.5, 0.75};
        auto scaled_input = input;
        scale_variances(scaled_input);

        auto const output = solver.run_state_estimation(y_bus, input, error_tolerance, num_iter, true, log);
        auto const scaled_output =
            solver.run_state_estimation(y_bus, scaled_input, error_tolerance, num_iter, true, log);
        assert_output(scaled_output, output);
        check_outputs(output, scaled_output);
    }

    SUBCASE("Power measurements without a physical angle reference") {
        auto const topo = grid.se_topo_power_sensors();
        YBus<asymmetric_t> const y_bus{topo, grid.param()};
        NewtonRaphsonSESolver<asymmetric_t> solver{y_bus, topo};
        auto input = grid.se_input_no_angle();
        auto scaled_input = input;
        scale_variances(scaled_input);

        auto const output = solver.run_state_estimation(y_bus, input, error_tolerance, num_iter, true, log);
        auto const scaled_output =
            solver.run_state_estimation(y_bus, scaled_input, error_tolerance, num_iter, true, log);
        assert_output(scaled_output, output);
        CHECK(output.bus_uncertainty[topo.slack_bus].u_angle_sigma(0) == 0.0);
        check_outputs(output, scaled_output);
    }

    SUBCASE("Local-angle current measurements") {
        auto const topo = grid.se_topo_current_sensors();
        YBus<asymmetric_t> const y_bus{topo, grid.param()};
        NewtonRaphsonSESolver<asymmetric_t> solver{y_bus, topo};
        auto input = grid.se_input_angle_current_sensors(AngleMeasurementType::local_angle);
        input.measured_branch_from_current.front().measurement.real_component.variance =
            RealValue<asymmetric_t>{0.25, 0.5, 0.75};
        auto scaled_input = input;
        scale_variances(scaled_input);

        auto const output = solver.run_state_estimation(y_bus, input, error_tolerance, num_iter, true, log);
        auto const scaled_output =
            solver.run_state_estimation(y_bus, scaled_input, error_tolerance, num_iter, true, log);
        assert_output(scaled_output, output);
        check_outputs(output, scaled_output);
    }
}

TEST_CASE("Asymmetric Newton-Raphson SE uncertainty matches local estimator sensitivities") {
    constexpr double error_tolerance{1e-12};
    constexpr Idx num_iter{30};
    constexpr double difference_step{1e-6};

    SESolverTestGrid<asymmetric_t> const grid;
    auto topo = grid.se_topo_current_sensors();
    topo.voltage_sensors_per_bus = {from_sparse, {0, 1, 2, 3}};
    topo.power_sensors_per_bus = {from_sparse, {0, 0, 0, 0}};
    topo.power_sensors_per_source = {from_sparse, {0, 0}};
    topo.power_sensors_per_load_gen = {from_sparse, {0, 0, 0, 0, 0, 0, 0, 0}};
    topo.power_sensors_per_shunt = {from_sparse, {0, 1}};
    topo.power_sensors_per_branch_from = {from_sparse, {0, 0, 0}};
    topo.power_sensors_per_branch_to = {from_sparse, {0, 0, 0}};
    topo.current_sensors_per_branch_from = {from_sparse, {0, 1, 1}};
    topo.current_sensors_per_branch_to = {from_sparse, {0, 0, 0}};

    ComplexValueVector<asymmetric_t> target_voltage = grid.output_ref().u;
    target_voltage[0] *=
        ComplexValue<asymmetric_t>{std::polar(1.03, 0.02), std::polar(0.91, -0.03), std::polar(1.08, 0.01)};
    target_voltage[1] *=
        ComplexValue<asymmetric_t>{std::polar(0.94, -0.01), std::polar(1.07, 0.04), std::polar(0.89, -0.02)};
    target_voltage[2] *=
        ComplexValue<asymmetric_t>{std::polar(1.06, 0.03), std::polar(0.93, -0.02), std::polar(1.02, 0.05)};

    YBus<asymmetric_t> const y_bus{topo, grid.param()};
    auto const target_branch = y_bus.calculate_branch_flow<BranchSolverOutput<asymmetric_t>>(target_voltage);
    auto const target_shunt = y_bus.calculate_shunt_flow<ApplianceSolverOutput<asymmetric_t>>(target_voltage);
    ComplexValue<asymmetric_t> const local_current = phase_shift(target_voltage[0]) * conj(target_branch[0].i_f);

    auto input = grid.se_input_angle_current_sensors(AngleMeasurementType::local_angle);
    input.load_gen_status[6] = 1;
    input.measured_voltage = {{target_voltage[0], 0.4}, {target_voltage[1], 0.7}, {target_voltage[2], 1.1}};
    input.measured_source_power.clear();
    input.measured_load_gen_power.clear();
    input.measured_shunt_power = {
        {.real_component = {.value = real(target_shunt[0].s), .variance = RealValue<asymmetric_t>{0.45, 0.7, 0.3}},
         .imag_component = {.value = imag(target_shunt[0].s), .variance = RealValue<asymmetric_t>{0.8, 0.4, 0.65}}}};
    input.measured_branch_from_power.clear();
    input.measured_branch_to_power.clear();
    input.measured_bus_injection.clear();
    input.measured_branch_from_current = {
        {.angle_measurement_type = AngleMeasurementType::local_angle,
         .measurement = {
             .real_component = {.value = real(local_current), .variance = RealValue<asymmetric_t>{0.25, 0.5, 0.8}},
             .imag_component = {.value = imag(local_current), .variance = RealValue<asymmetric_t>{0.9, 0.35, 0.6}}}}};
    input.measured_branch_to_current.clear();

    NewtonRaphsonSESolver<asymmetric_t> solver{y_bus, topo};
    auto log = get_logger();
    auto const output = solver.run_state_estimation(y_bus, input, error_tolerance, num_iter, true, log);
    for (Idx bus = 0; bus != std::ssize(target_voltage); ++bus) {
        double const target_error = max_val(cabs(output.u[bus] - target_voltage[bus]));
        CAPTURE(bus);
        CAPTURE(target_error);
        CHECK(target_error < 1e-10);
    }

    using Observations = std::array<double, 7>;
    auto const observe = [](SolverOutput<asymmetric_t> const& value) -> Observations {
        return {std::abs(value.u[1](1)),         std::arg(value.u[1](1)),          real(value.bus_injection[1])(0),
                imag(value.bus_injection[1])(2), std::abs(value.branch[0].i_f(2)), real(value.branch[0].s_f)(1),
                imag(value.branch[1].s_t)(0)};
    };

    Observations finite_difference_variance{};
    auto const add_noise_channel = [&](double measurement_variance, auto mutate) {
        REQUIRE(std::isfinite(measurement_variance));
        REQUIRE(measurement_variance > 0.0);
        auto plus_input = input;
        auto minus_input = input;
        mutate(plus_input, difference_step);
        mutate(minus_input, -difference_step);
        auto const plus = solver.run_state_estimation(y_bus, plus_input, error_tolerance, num_iter, false, log);
        auto const minus = solver.run_state_estimation(y_bus, minus_input, error_tolerance, num_iter, false, log);
        auto const plus_observations = observe(plus);
        auto const minus_observations = observe(minus);
        for (Idx idx = 0; idx != std::ssize(finite_difference_variance); ++idx) {
            double const derivative = (plus_observations[idx] - minus_observations[idx]) / (2.0 * difference_step);
            finite_difference_variance[idx] += measurement_variance * derivative * derivative;
        }
    };

    for (Idx sensor = 0; sensor != std::ssize(input.measured_voltage); ++sensor) {
        for (Idx phase = 0; phase != 3; ++phase) {
            double const variance = input.measured_voltage[sensor].variance;
            add_noise_channel(variance, [sensor, phase](auto& varied_input, double delta) {
                auto& value = varied_input.measured_voltage[sensor].value(phase);
                value += delta * value / std::abs(value);
            });
            add_noise_channel(variance, [sensor, phase](auto& varied_input, double delta) {
                varied_input.measured_voltage[sensor].value(phase) *= std::polar(1.0, delta);
            });
        }
    }
    for (Idx phase = 0; phase != 3; ++phase) {
        auto const& current = input.measured_branch_from_current.front().measurement;
        add_noise_channel(current.real_component.variance(phase), [phase](auto& varied_input, double delta) {
            varied_input.measured_branch_from_current.front().measurement.real_component.value(phase) += delta;
        });
        add_noise_channel(current.imag_component.variance(phase), [phase](auto& varied_input, double delta) {
            varied_input.measured_branch_from_current.front().measurement.imag_component.value(phase) += delta;
        });

        auto const& shunt = input.measured_shunt_power.front();
        add_noise_channel(shunt.real_component.variance(phase), [phase](auto& varied_input, double delta) {
            varied_input.measured_shunt_power.front().real_component.value(phase) += delta;
        });
        add_noise_channel(shunt.imag_component.variance(phase), [phase](auto& varied_input, double delta) {
            varied_input.measured_shunt_power.front().imag_component.value(phase) += delta;
        });
    }

    Observations const analytical_sigma{
        output.bus_uncertainty[1].u_sigma(1), output.bus_uncertainty[1].u_angle_sigma(1),
        output.bus_uncertainty[1].p_sigma(0), output.bus_uncertainty[1].q_sigma(2),
        output.branch[0].i_f_sigma(2),        output.branch[0].p_f_sigma(1),
        output.branch[1].q_t_sigma(0),
    };
    for (Idx idx = 0; idx != std::ssize(analytical_sigma); ++idx) {
        CAPTURE(idx);
        REQUIRE(std::isfinite(analytical_sigma[idx]));
        CHECK(analytical_sigma[idx] == doctest::Approx(std::sqrt(finite_difference_variance[idx])).epsilon(2e-5));
    }
}
} // namespace power_grid_model::math_solver
