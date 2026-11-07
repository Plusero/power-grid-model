// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#include "load_dataset.hpp"
#include <cmath>
#include <cstdint>
#include <doctest/doctest.h>
#include <power_grid_model_c/dataset_definitions.h>
#include <power_grid_model_cpp/buffer.hpp>
#include <power_grid_model_cpp/model.hpp>
#include <power_grid_model_cpp/options.hpp>

namespace power_grid_model_cpp {
TEST_CASE("API observer voltage values and absent native output storage") {
    auto input = power_grid_model_cpp_test::load_dataset(R"({
        "version":"1.0","type":"input","is_batch":false,"attributes":{},"data":{
        "node":[{"id":1,"u_rated":100}],
        "source":[{"id":2,"node":1,"status":1,"u_ref":1}],
        "sym_load":[{"id":3,"node":1,"status":1,"type":0,"p_specified":100,"q_specified":20}],
        "sym_voltage_sensor":[{"id":4,"measured_object":1,"u_sigma":2,"u_measured":100,"u_angle_measured":0.2}],
        "voltage_uncertainty_observer":[{"id":100,"observed_object":1,"status":1}]}})");
    Model model{50.0, input.dataset};
    Options options;
    options.set_calculation_type(PGM_state_estimation);
    options.set_symmetric(1);
    // Count-only registrations are absent output demand for unsupported modes too.
    for (auto calculation : {PGM_power_flow, PGM_short_circuit}) {
        options.set_calculation_type(calculation);
        DatasetMutable absent{calculation == PGM_short_circuit ? "sc_output" : "sym_output", false, 1};
        absent.add_buffer("voltage_uncertainty_observer", 1, 1, nullptr, nullptr);
        CHECK_NOTHROW(model.calculate(options, absent));
    }
    options.set_calculation_type(PGM_state_estimation);
    for (auto method : {PGM_iterative_linear, PGM_newton_raphson}) {
        options.set_calculation_method(method);
        Buffer buffer{PGM_def_sym_output_voltage_uncertainty_observer, 1};
        buffer.set_nan();
        DatasetMutable output{"sym_output", false, 1};
        output.add_buffer("voltage_uncertainty_observer", 1, 1, nullptr, buffer);
        model.calculate(options, output);
        ID id{};
        std::int8_t energized{};
        double magnitude{}, angle{};
        buffer.get_value(PGM_def_sym_output_voltage_uncertainty_observer_id, &id, -1);
        buffer.get_value(PGM_def_sym_output_voltage_uncertainty_observer_energized, &energized, -1);
        buffer.get_value(PGM_def_sym_output_voltage_uncertainty_observer_u_sigma, &magnitude, -1);
        buffer.get_value(PGM_def_sym_output_voltage_uncertainty_observer_u_angle_sigma, &angle, -1);
        double const expected = method == PGM_iterative_linear ? std::sqrt(2.0) : 2.0;
        CHECK(id == 100);
        CHECK(energized == 1);
        CHECK(magnitude == doctest::Approx(expected));
        CHECK(angle == doctest::Approx(expected / 100.0));

        DatasetMutable empty{"sym_output", false, 1};
        empty.add_buffer("voltage_uncertainty_observer", 1, 1, nullptr, nullptr);
        CHECK_NOTHROW(model.calculate(options, empty));
        ID metadata_id{};
        DatasetMutable metadata{"sym_output", false, 1};
        metadata.add_buffer("voltage_uncertainty_observer", 1, 1, nullptr, nullptr);
        metadata.add_attribute_buffer("voltage_uncertainty_observer", "id", &metadata_id);
        CHECK_NOTHROW(model.calculate(options, metadata));
        CHECK(metadata_id == 100);
    }
}
TEST_CASE("API asymmetric observer phase arrays support row and column buffers with physical units") {
    auto input = power_grid_model_cpp_test::load_dataset(R"({
        "version":"1.0","type":"input","is_batch":false,"attributes":{},"data":{
        "node":[{"id":1,"u_rated":100}],
        "source":[{"id":2,"node":1,"status":1,"u_ref":1}],
        "asym_load":[{"id":3,"node":1,"status":1,"type":0,
            "p_specified":[20,30,40],"q_specified":[5,6,7]}],
        "asym_voltage_sensor":[{"id":4,"measured_object":1,"u_sigma":2,
            "u_measured":[60,45,70],"u_angle_measured":[0.2,-1.8,2.3]}],
        "voltage_uncertainty_observer":[{"id":100,"observed_object":1,"status":1}]}})");
    Model model{50.0, input.dataset};
    Options options;
    options.set_calculation_type(PGM_state_estimation);
    options.set_symmetric(0);
    for (auto method : {PGM_iterative_linear, PGM_newton_raphson}) {
        options.set_calculation_method(method);
        Buffer buffer{PGM_def_asym_output_voltage_uncertainty_observer, 1};
        buffer.set_nan();
        DatasetMutable rows{"asym_output", false, 1};
        rows.add_buffer("voltage_uncertainty_observer", 1, 1, nullptr, buffer);
        model.calculate(options, rows);
        double magnitude[3]{}, angle[3]{};
        buffer.get_value(PGM_def_asym_output_voltage_uncertainty_observer_u_sigma, magnitude, -1);
        buffer.get_value(PGM_def_asym_output_voltage_uncertainty_observer_u_angle_sigma, angle, -1);
        double const voltages[3]{60., 45., 70.};
        double const expected = method == PGM_iterative_linear ? std::sqrt(2.) : 2.;
        for (int p = 0; p != 3; ++p) {
            CHECK(magnitude[p] == doctest::Approx(expected));
            double const expected_angle =
                method == PGM_iterative_linear ? expected / voltages[p] : 2. * std::sqrt(3.) / 100.;
            CHECK(angle[p] == doctest::Approx(expected_angle));
        }
        double column_magnitude[3]{}, column_angle[3]{};
        ID id{};
        std::int8_t energized{};
        DatasetMutable columns{"asym_output", false, 1};
        columns.add_buffer("voltage_uncertainty_observer", 1, 1, nullptr, nullptr);
        columns.add_attribute_buffer("voltage_uncertainty_observer", "id", &id);
        columns.add_attribute_buffer("voltage_uncertainty_observer", "energized", &energized);
        columns.add_attribute_buffer("voltage_uncertainty_observer", "u_sigma", column_magnitude);
        columns.add_attribute_buffer("voltage_uncertainty_observer", "u_angle_sigma", column_angle);
        model.calculate(options, columns);
        CHECK(id == 100);
        CHECK(energized == 1);
        for (int p = 0; p != 3; ++p) {
            CHECK(column_magnitude[p] == magnitude[p]);
            CHECK(column_angle[p] == angle[p]);
        }
    }
}

} // namespace power_grid_model_cpp
