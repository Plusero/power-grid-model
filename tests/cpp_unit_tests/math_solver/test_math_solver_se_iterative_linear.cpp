// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#include "test_math_solver_se.hpp" // NOLINT(misc-include-cleaner)

#include <power_grid_model/math_solver/iterative_linear_se_solver.hpp> // NOLINT(misc-include-cleaner)

#include <power_grid_model/common/common.hpp>

#include <doctest/doctest.h>

#include <cmath>

TYPE_TO_STRING_AS("IterativeLinearSESolver<symmetric_t>",
                  power_grid_model::math_solver::IterativeLinearSESolver<power_grid_model::symmetric_t>);
TYPE_TO_STRING_AS("IterativeLinearSESolver<asymmetric_t>",
                  power_grid_model::math_solver::IterativeLinearSESolver<power_grid_model::asymmetric_t>);

namespace power_grid_model::math_solver {
TEST_CASE_TEMPLATE_INVOKE(test_math_solver_se_id, IterativeLinearSESolver<symmetric_t>);
TEST_CASE_TEMPLATE_INVOKE(test_math_solver_se_id, IterativeLinearSESolver<asymmetric_t>);
TEST_CASE_TEMPLATE_INVOKE(test_math_solver_se_zero_variance_id, IterativeLinearSESolver<symmetric_t>);
TEST_CASE_TEMPLATE_INVOKE(test_math_solver_se_measurements_id, IterativeLinearSESolver<symmetric_t>);

TEST_CASE("Iterative linear SE uncertainty scales with measurement variance and reuses the solver") {
    constexpr double error_tolerance{1e-10};
    constexpr Idx num_iter{20};
    constexpr double variance_scale{4.0};

    SESolverTestGrid<symmetric_t> const grid;
    auto const topo = grid.se_topo_power_sensors();
    YBus<symmetric_t> const y_bus{topo, grid.param()};
    IterativeLinearSESolver<symmetric_t> solver{y_bus, topo};
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

    auto const output = solver.run_state_estimation(y_bus, input, error_tolerance, num_iter, true, log);
    auto const scaled_output = solver.run_state_estimation(y_bus, scaled_input, error_tolerance, num_iter, true, log);

    assert_output(scaled_output, output);
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
} // namespace power_grid_model::math_solver
