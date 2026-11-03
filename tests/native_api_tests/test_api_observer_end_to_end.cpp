// SPDX-FileCopyrightText: Contributors to the Power Grid Model project
// <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#include "load_dataset.hpp"

#include <power_grid_model_c/dataset_definitions.h>
#include <power_grid_model_cpp/buffer.hpp>
#include <power_grid_model_cpp/model.hpp>
#include <power_grid_model_cpp/options.hpp>

#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>

namespace power_grid_model_cpp {
namespace {
using Coordinates = std::array<double, 4>;

Coordinates branch_quantities(Coordinates const& x, int terminal, bool polar) {
    using Complex = std::complex<double>;
    std::array<Complex, 2> const u = polar ? std::array{std::polar(x[1], x[0]), std::polar(x[3], x[2])}
                                           : std::array{Complex{x[0], x[1]}, Complex{x[2], x[3]}};
    auto const current = (u[terminal] - u[1 - terminal]) / (std::sqrt(3.0) * Complex{1.0, 2.0});
    auto const power = std::sqrt(3.0) * u[terminal] * std::conj(current);
    return {power.real(), power.imag(), std::abs(current), std::arg(current)};
}
} // namespace

TEST_CASE("API observer mixed row-column branch outputs match physical Jacobian") {
    auto input = power_grid_model_cpp_test::load_dataset(R"({
      "version":"1.0","type":"input","is_batch":false,"attributes":{},"data":{
      "node":[{"id":1,"u_rated":100},{"id":2,"u_rated":100}],
      "line":[{"id":10,"from_node":1,"to_node":2,"from_status":1,"to_status":1,
               "r1":1,"x1":2,"c1":0,"tan1":0,"r0":1,"x0":2,"c0":0,"tan0":0,"i_n":1000}],
      "source":[{"id":20,"node":1,"status":1,"u_ref":1}],
      "sym_load":[{"id":21,"node":2,"status":1,"type":0,"p_specified":100,"q_specified":20}],
      "sym_voltage_sensor":[
        {"id":31,"measured_object":1,"u_sigma":2,"u_measured":100,"u_angle_measured":0.2},
        {"id":32,"measured_object":2,"u_sigma":3,"u_measured":90,"u_angle_measured":0.1}],
      "power_uncertainty_observer":[
        {"id":111,"observed_object":10,"observed_terminal_type":0,"status":1},
        {"id":112,"observed_object":10,"observed_terminal_type":1,"status":1}],
      "current_uncertainty_observer":[
        {"id":121,"observed_object":10,"observed_terminal_type":0,"status":1},
        {"id":122,"observed_object":10,"observed_terminal_type":1,"status":1}]}})");
    Model model{50.0, input.dataset};
    Options options;
    options.set_calculation_type(PGM_state_estimation);
    options.set_symmetric(1);
    for (auto const method : {PGM_iterative_linear, PGM_newton_raphson}) {
        options.set_calculation_method(method);
        Buffer power{PGM_def_sym_output_power_uncertainty_observer, 2};
        power.set_nan();
        std::array<ID, 2> ids{};
        std::array<std::int8_t, 2> energized{};
        std::array<double, 2> magnitude{}, angle{};
        DatasetMutable output{"sym_output", false, 1};
        output.add_buffer("power_uncertainty_observer", 2, 2, nullptr, power);
        output.add_buffer("current_uncertainty_observer", 2, 2, nullptr, nullptr);
        output.add_attribute_buffer("current_uncertainty_observer", "id", ids.data());
        output.add_attribute_buffer("current_uncertainty_observer", "energized", energized.data());
        output.add_attribute_buffer("current_uncertainty_observer", "i_sigma", magnitude.data());
        output.add_attribute_buffer("current_uncertainty_observer", "i_angle_sigma", angle.data());
        model.calculate(options, output);
        std::array<double, 2> active{}, reactive{};
        power.get_value(PGM_def_sym_output_power_uncertainty_observer_p_sigma, active.data(), -1);
        power.get_value(PGM_def_sym_output_power_uncertainty_observer_q_sigma, reactive.data(), -1);
        CHECK(ids == std::array<ID, 2>{121, 122});
        CHECK(energized == std::array<std::int8_t, 2>{1, 1});
        bool const polar = method == PGM_newton_raphson;
        auto const first = std::polar(100.0, 0.2), second = std::polar(90.0, 0.1);
        Coordinates const state = polar ? Coordinates{0.2, 100.0, 0.1, 90.0}
                                        : Coordinates{first.real(), first.imag(), second.real(), second.imag()};
        Coordinates const covariance = polar ? Coordinates{0.0004, 4.0, 0.0009, 9.0} : Coordinates{2.0, 2.0, 4.5, 4.5};
        for (int terminal = 0; terminal != 2; ++terminal) {
            Coordinates variance{};
            for (int coordinate = 0; coordinate != 4; ++coordinate) {
                double const step = polar && coordinate % 2 == 0 ? 1e-6 : 1e-4;
                auto plus = state, minus = state;
                plus[coordinate] += step;
                minus[coordinate] -= step;
                auto const upper = branch_quantities(plus, terminal, polar),
                           lower = branch_quantities(minus, terminal, polar);
                for (int quantity = 0; quantity != 4; ++quantity) {
                    double const derivative = (upper[quantity] - lower[quantity]) / (2.0 * step);
                    variance[quantity] += derivative * covariance[coordinate] * derivative;
                }
            }
            Coordinates const actual{active[terminal], reactive[terminal], magnitude[terminal], angle[terminal]};
            for (int quantity = 0; quantity != 4; ++quantity) {
                CHECK(actual[quantity] == doctest::Approx(std::sqrt(variance[quantity])).epsilon(2e-8));
            }
        }
    }
}
} // namespace power_grid_model_cpp
