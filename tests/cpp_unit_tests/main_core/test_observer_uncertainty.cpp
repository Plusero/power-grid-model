// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#include <power_grid_model/all_components.hpp>
#include <power_grid_model/auxiliary/meta_data_gen.hpp>
#include <power_grid_model/container.hpp>
#include <power_grid_model/main_core/observer_uncertainty.hpp>

#include <doctest/doctest.h>

#include <vector>

namespace power_grid_model::main_core {
using namespace observer_uncertainty;

TEST_CASE("Observer demand follows registered native row and column storage") {
    MutableDataset data{false, 1, "sym_output", meta_data::meta_data_gen::meta_data};
    data.add_buffer("voltage_uncertainty_observer", 1, 1, nullptr, nullptr);
    auto selection = output_selection<VoltageUncertaintyObserver>(data);
    CHECK_FALSE(selection.requested);
    // A known count without storage is traversable but has no observer bindings.
    // Output must return without indexing those absent bindings.
    using Components = Container<ExtraRetrievableTypes<Base, Observer>, VoltageUncertaintyObserver>;
    MainModelState<Components> state{};
    Scenario<symmetric_t> scenario{data};
    data.for_each_component<meta_data::sym_output_getter_s, VoltageUncertaintyObserver>([&](auto output) {
        CHECK_NOTHROW(output_observer_uncertainty<VoltageUncertaintyObserver>(state, scenario, output));
    });
    ID id{};
    data.add_attribute_buffer("voltage_uncertainty_observer", "id", &id);
    selection = output_selection<VoltageUncertaintyObserver>(data);
    CHECK(selection.requested);
    CHECK_FALSE(selection.sigma[0]);
    CHECK_FALSE(selection.sigma[1]);
    double angle{};
    data.add_attribute_buffer("voltage_uncertainty_observer", "u_angle_sigma", &angle);
    selection = output_selection<VoltageUncertaintyObserver>(data);
    CHECK_FALSE(selection.sigma[0]);
    CHECK(selection.sigma[1]);
    VoltageUncertaintyObserverOutput row{};
    MutableDataset rows{false, 1, "sym_output", meta_data::meta_data_gen::meta_data};
    rows.add_buffer("voltage_uncertainty_observer", 1, 1, nullptr, &row);
    selection = output_selection<VoltageUncertaintyObserver>(rows);
    CHECK(selection.requested);
    CHECK(selection.sigma[0]);
    CHECK(selection.sigma[1]);
}

TEST_CASE("Observers of one bus or branch side share a request row") {
    math_solver::StateEstimationUQRequest request;
    RequestRows row_of;
    auto const add = [&](Quantity quantity, Idx index, BranchSide side, bool first, bool second) {
        return add_request(request, row_of, {.quantity = quantity, .math_index = index, .side = side}, {first, second});
    };

    CHECK(add(Quantity::voltage, 2, BranchSide::from, true, false) == 0);
    CHECK(add(Quantity::voltage, 0, BranchSide::from, false, false) == 1);
    CHECK(add(Quantity::voltage, 2, BranchSide::from, false, true) == 0);
    REQUIRE(request.voltage.size() == 2);
    CHECK(request.voltage[0].bus == 2);
    CHECK(request.voltage[0].magnitude);
    CHECK(request.voltage[0].angle);
    CHECK(request.voltage[1].bus == 0);
    CHECK_FALSE(request.voltage[1].magnitude);
    CHECK_FALSE(request.voltage[1].angle);

    // Power and current observers of one side share a row; the other side of the branch has its own.
    CHECK(add(Quantity::branch_power, 3, BranchSide::to, true, false) == 0);
    CHECK(add(Quantity::branch_current, 3, BranchSide::from, true, false) == 1);
    CHECK(add(Quantity::branch_current, 3, BranchSide::to, false, true) == 0);
    CHECK(add(Quantity::branch_power, 3, BranchSide::to, false, true) == 0);
    REQUIRE(request.branch.size() == 2);
    CHECK(request.branch[0].branch == 3);
    CHECK(request.branch[0].side == BranchSide::to);
    CHECK(request.branch[0].active_power);
    CHECK(request.branch[0].reactive_power);
    CHECK_FALSE(request.branch[0].current_magnitude);
    CHECK(request.branch[0].current_angle);
    CHECK(request.branch[1].side == BranchSide::from);
    CHECK_FALSE(request.branch[1].active_power);
    CHECK_FALSE(request.branch[1].reactive_power);
    CHECK(request.branch[1].current_magnitude);
    CHECK_FALSE(request.branch[1].current_angle);

    // An injection row of a bus is separate from its voltage row.
    CHECK(add(Quantity::injection, 2, BranchSide::from, true, false) == 0);
    CHECK(add(Quantity::injection, 2, BranchSide::from, false, true) == 0);
    REQUIRE(request.injection.size() == 1);
    CHECK(request.injection[0].bus == 2);
    CHECK(request.injection[0].active_power);
    CHECK(request.injection[0].reactive_power);
    CHECK(request.voltage.size() == 2);
}

TEST_CASE("Absent observer output demand does not access topology or allocate island work") {
    MutableDataset data{false, 1, "sym_output", meta_data::meta_data_gen::meta_data};
    using Components = Container<ExtraRetrievableTypes<Base, Node, Observer>, Node, VoltageUncertaintyObserver>;
    MainModelState<Components> state{};
    Scenario<symmetric_t> scenario{data};
    // A null reduced topology is sufficient because absent demand must return before resolving targets.
    CHECK_NOTHROW(prepare_observer_uncertainty(state, scenario, 1000));
    CHECK(scenario.work.empty());
    CHECK(scenario.of<VoltageUncertaintyObserver>().bindings.empty());
}

TEST_CASE("Asymmetric observer output scales each phase and keeps missing values") {
    using Components = Container<ExtraRetrievableTypes<Base, Observer>, VoltageUncertaintyObserver>;
    MainModelState<Components> state{};
    emplace_component<VoltageUncertaintyObserver>(
        state.components, 100, VoltageUncertaintyObserverInput{.id = 100, .observed_object = 1, .status = 1});
    state.components.set_construction_complete();
    AsymVoltageUncertaintyObserverOutput output;
    MutableDataset data{false, 1, "asym_output", meta_data::meta_data_gen::meta_data};
    data.add_buffer("voltage_uncertainty_observer", 1, 1, nullptr, &output);
    Scenario<asymmetric_t> scenario{data};
    scenario.work.resize(1);
    scenario.work[0].result.voltage.resize(1);
    auto& values = scenario.work[0].result.voltage[0];
    for (int p = 0; p != 3; ++p) {
        values.magnitude.phase[p].sigma = p == 1 ? nan : p + 1.;
        values.angle.phase[p].sigma = p == 0 ? nan : .1 * (p + 1.);
    }
    auto& bindings = scenario.of<VoltageUncertaintyObserver>().bindings;
    bindings.push_back({.island = 0, .energized = true, .scale = {100. / sqrt3, 1.0}, .result_index = 0});
    auto produce = [&] {
        data.for_each_component<meta_data::asym_output_getter_s, VoltageUncertaintyObserver>(
            [&](auto span) { output_observer_uncertainty<VoltageUncertaintyObserver>(state, scenario, span); });
    };
    produce();
    CHECK(output.id == 100);
    CHECK(output.energized == 1);
    CHECK(output.u_sigma(0) == doctest::Approx(100. / sqrt3));
    CHECK(is_nan(output.u_sigma(1)));
    CHECK(output.u_sigma(2) == doctest::Approx(300. / sqrt3));
    CHECK(is_nan(output.u_angle_sigma(0)));
    CHECK(output.u_angle_sigma(1) == .2);
    CHECK(output.u_angle_sigma(2) == doctest::Approx(.3));
    bindings[0].result_index = -1;
    produce();
    CHECK(output.energized == 1); // Connectivity is independent of numerical validity.
    CHECK(output.u_sigma.isNaN().all());
    CHECK(output.u_angle_sigma.isNaN().all());
}

} // namespace power_grid_model::main_core
