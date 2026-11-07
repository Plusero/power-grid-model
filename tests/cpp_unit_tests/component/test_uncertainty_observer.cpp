// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#include <power_grid_model/all_components.hpp>
#include <power_grid_model/auxiliary/meta_data_gen.hpp>
#include <power_grid_model/auxiliary/static_asserts/input.hpp>
#include <power_grid_model/auxiliary/static_asserts/output.hpp>
#include <power_grid_model/auxiliary/static_asserts/update.hpp>

#include <doctest/doctest.h>

#include <type_traits>

namespace power_grid_model {

static_assert(component_c<VoltageUncertaintyObserver>);
static_assert(component_c<PowerUncertaintyObserver>);
static_assert(component_c<CurrentUncertaintyObserver>);
static_assert(!std::derived_from<Observer, Sensor>);
static_assert(std::same_as<VoltageUncertaintyObserver::OutputType<symmetric_t>, VoltageUncertaintyObserverOutput>);

TEST_CASE("Uncertainty observer identity, status and missing output") {
    VoltageUncertaintyObserver observer{{.id = 100, .observed_object = 1, .status = 1}};
    CHECK(observer.id() == 100);
    CHECK(observer.observed_object() == 1);
    CHECK(observer.status());
    CHECK(observer.math_model_type() == ComponentType::observer);
    CHECK(observer.energized(true));
    CHECK_FALSE(observer.energized(false));
    auto const output = observer.get_output(true, 12.6, nan);
    CHECK(output.id == 100);
    CHECK(output.energized == 1);
    CHECK(output.u_sigma == 12.6);
    CHECK(is_nan(output.u_angle_sigma));
    auto const disconnected = observer.get_output(false, 12.6, 0.001);
    CHECK(disconnected.energized == 0);
    CHECK(is_nan(disconnected.u_sigma));
    CHECK(is_nan(disconnected.u_angle_sigma));
    ObserverUpdate const disable{.id = 100, .status = 0};
    auto const restore = observer.inverse(disable);
    CHECK(restore.status == 1);
    auto const change = observer.update(disable);
    CHECK_FALSE(change.topo);
    CHECK_FALSE(change.param);
    CHECK_FALSE(observer.status());
    auto const disabled = observer.get_output(true, 12.6, 0.001);
    CHECK(disabled.energized == 0);
    CHECK(is_nan(disabled.u_sigma));
    CHECK(is_nan(disabled.u_angle_sigma));
    observer.update({}); // missing ID and status retain positional update semantics
    CHECK_FALSE(observer.status());
    CHECK(is_nan(observer.inverse({}).status));
    observer.update(restore);
    CHECK(observer.status());
    CHECK_THROWS_AS(observer.update({.id = 100, .status = 2}), InvalidArguments);
    CHECK(observer.status());
    CHECK_THROWS_AS(VoltageUncertaintyObserver({.id = 100, .observed_object = 1}), InvalidArguments);
    CHECK_THROWS_AS(VoltageUncertaintyObserver({.id = 100, .status = 1}), InvalidArguments);
}

TEST_CASE("Uncertainty observer terminal selector validation and output") {
    for (IntS code = 0; code <= 10; ++code) {
        auto const terminal = static_cast<ObservedTerminalType>(code);
        bool const branch = code == 0 || code == 1 || code == 6 || code == 7 || code == 8;
        PowerUncertaintyObserverInput const power_input{
            .id = 101, .observed_object = 4, .status = 0, .observed_terminal_type = terminal};
        CurrentUncertaintyObserverInput const current_input{
            .id = 102, .observed_object = 4, .status = 0, .observed_terminal_type = terminal};
        if (branch || code == 9) {
            PowerUncertaintyObserver observer{power_input};
            CHECK(observer.observed_terminal_type() == terminal);
        } else {
            CHECK_THROWS_AS(PowerUncertaintyObserver{power_input}, InvalidArguments);
        }
        if (branch) {
            CurrentUncertaintyObserver observer{current_input};
            CHECK(observer.observed_terminal_type() == terminal);
        } else {
            CHECK_THROWS_AS(CurrentUncertaintyObserver{current_input}, InvalidArguments);
        }
    }
    CHECK_THROWS_AS(CurrentUncertaintyObserver({.id = 102, .observed_object = 4, .status = 1}), InvalidArguments);
    PowerUncertaintyObserver power{
        {.id = 101, .observed_object = 4, .status = 1, .observed_terminal_type = ObservedTerminalType::branch_from}};
    CHECK(power.get_output(true, 2.0, 3.0).p_sigma == 2.0);
    CHECK(power.get_output(true, 2.0, 3.0).q_sigma == 3.0);
    CurrentUncertaintyObserver current{
        {.id = 102, .observed_object = 4, .status = 1, .observed_terminal_type = ObservedTerminalType::branch_to}};
    CHECK(current.get_output(true, 4.0, 5.0).i_sigma == 4.0);
    CHECK(current.get_output(true, 4.0, 5.0).i_angle_sigma == 5.0);
}

TEST_CASE("Uncertainty observer metadata contains scalar sigma pairs and immutable targets") {
    using namespace meta_data;
    auto const& input_attributes = get_attributes_list<CurrentUncertaintyObserverInput>::value;
    auto const& update_attributes = get_attributes_list<ObserverUpdate>::value;
    auto const& output_attributes = get_attributes_list<CurrentUncertaintyObserverOutput>::value;
    CHECK(input_attributes.size() == 4);
    CHECK(update_attributes.size() == 2);
    CHECK(output_attributes.size() == 4);
    CHECK(std::string_view{input_attributes[3].name} == "observed_terminal_type");
    CHECK(input_attributes[3].ctype == CType::c_int8);
    CHECK(std::string_view{update_attributes[1].name} == "status");
    CHECK(std::string_view{output_attributes[2].name} == "i_sigma");
    CHECK(std::string_view{output_attributes[3].name} == "i_angle_sigma");
    CHECK(output_attributes[2].ctype == CType::c_double);
    CHECK(output_attributes[3].ctype == CType::c_double);
    auto const& metadata = meta_data_gen::meta_data;
    CHECK(metadata.get_dataset("sym_output").get_component("current_uncertainty_observer").attributes.size() == 4);
    CHECK(metadata.get_dataset("sc_output").get_component("current_uncertainty_observer").attributes.size() == 2);
}

TEST_CASE("Asymmetric observer records preserve independent phase missingness and scalar metadata") {
    static_assert(
        std::same_as<VoltageUncertaintyObserver::OutputType<asymmetric_t>, AsymVoltageUncertaintyObserverOutput>);
    static_assert(std::same_as<PowerUncertaintyObserver::OutputType<asymmetric_t>, AsymPowerUncertaintyObserverOutput>);
    static_assert(
        std::same_as<CurrentUncertaintyObserver::OutputType<asymmetric_t>, AsymCurrentUncertaintyObserverOutput>);
    VoltageUncertaintyObserver voltage{{.id = 100, .observed_object = 1, .status = 1}};
    RealValue<asymmetric_t> const first{2., nan, 4.}, second{nan, .2, .3};
    auto const output = voltage.get_output<asymmetric_t>(true, first, second);
    CHECK(output.id == 100);
    CHECK(output.energized == 1);
    CHECK(output.u_sigma(0) == 2.);
    CHECK(is_nan(output.u_sigma(1)));
    CHECK(output.u_sigma(2) == 4.);
    CHECK(is_nan(output.u_angle_sigma(0)));
    CHECK(output.u_angle_sigma(1) == .2);
    CHECK(output.u_angle_sigma(2) == .3);
    auto const disconnected = voltage.get_output<asymmetric_t>(false, first, second);
    CHECK(disconnected.energized == 0);
    CHECK(disconnected.u_sigma.isNaN().all());
    CHECK(disconnected.u_angle_sigma.isNaN().all());
    PowerUncertaintyObserver power{
        {.id = 101, .observed_object = 4, .status = 0, .observed_terminal_type = ObservedTerminalType::branch_from}};
    auto const disabled = power.get_output<asymmetric_t>(true, first, second);
    CHECK(disabled.energized == 0);
    CHECK(disabled.p_sigma.isNaN().all());
    CHECK(disabled.q_sigma.isNaN().all());
    CurrentUncertaintyObserver current{
        {.id = 102, .observed_object = 4, .status = 1, .observed_terminal_type = ObservedTerminalType::branch_to}};
    auto const current_output = current.get_output<asymmetric_t>(true, first, second);
    CHECK(current_output.i_sigma(0) == 2.);
    CHECK(is_nan(current_output.i_sigma(1)));
    CHECK(current_output.i_angle_sigma(2) == .3);
    for (char const* name :
         {VoltageUncertaintyObserver::name, PowerUncertaintyObserver::name, CurrentUncertaintyObserver::name}) {
        auto const& symmetric = meta_data::meta_data_gen::meta_data.get_dataset("sym_output").get_component(name);
        auto const& asymmetric = meta_data::meta_data_gen::meta_data.get_dataset("asym_output").get_component(name);
        CHECK(symmetric.attributes[0].ctype == CType::c_int32);
        CHECK(asymmetric.attributes[0].ctype == CType::c_int32);
        CHECK(asymmetric.attributes[1].ctype == CType::c_int8);
        CHECK(symmetric.attributes[2].ctype == CType::c_double);
        CHECK(symmetric.attributes[3].ctype == CType::c_double);
        CHECK(asymmetric.attributes[2].ctype == CType::c_double3);
        CHECK(asymmetric.attributes[3].ctype == CType::c_double3);
        CHECK(symmetric.size == 24);
        CHECK(asymmetric.size == 56);
        CHECK(symmetric.attributes[2].offset == asymmetric.attributes[2].offset);
        CHECK(symmetric.alignment == asymmetric.alignment);
    }
}

} // namespace power_grid_model
