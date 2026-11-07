// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#include <power_grid_model/all_components.hpp>
#include <power_grid_model/auxiliary/meta_data_gen.hpp>
#include <power_grid_model/container.hpp>
#include <power_grid_model/main_core/observer_uncertainty.hpp>

#include <doctest/doctest.h>

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

namespace power_grid_model::main_core {
using namespace observer_uncertainty;

TEST_CASE("Observer demand follows registered native row and column storage") {
    MutableDataset data{false, 1, "sym_output", meta_data::meta_data_gen::meta_data};
    data.add_buffer("voltage_uncertainty_observer", 1, 1, nullptr, nullptr);
    auto selection = output_selection(data, "voltage_uncertainty_observer", "u_sigma", "u_angle_sigma");
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
    selection = output_selection(data, "voltage_uncertainty_observer", "u_sigma", "u_angle_sigma");
    CHECK(selection.requested);
    CHECK_FALSE(selection.first);
    CHECK_FALSE(selection.second);
    double angle{};
    data.add_attribute_buffer("voltage_uncertainty_observer", "u_angle_sigma", &angle);
    selection = output_selection(data, "voltage_uncertainty_observer", "u_sigma", "u_angle_sigma");
    CHECK_FALSE(selection.first);
    CHECK(selection.second);
    VoltageUncertaintyObserverOutput row{};
    MutableDataset rows{false, 1, "sym_output", meta_data::meta_data_gen::meta_data};
    rows.add_buffer("voltage_uncertainty_observer", 1, 1, nullptr, &row);
    selection = output_selection(rows, "voltage_uncertainty_observer", "u_sigma", "u_angle_sigma");
    CHECK(selection.requested);
    CHECK(selection.first);
    CHECK(selection.second);
}

TEST_CASE("Observer requests normalize unioned quantities in core order") {
    math_solver::StateEstimationUQRequest request;
    request.voltage = {{2, true, false}, {0, true, true}, {2, false, true}};
    request.branch = {{3, BranchSide::to, true, false, false, false},
                      {3, BranchSide::from, false, false, true, false},
                      {3, BranchSide::to, false, true, false, true}};
    request.injection = {{2, true, false}, {2, false, true}};
    auto const positions = normalize(request);
    CHECK(positions[static_cast<std::size_t>(Quantity::voltage)] == std::vector<Idx>{0, -1, 1});
    CHECK(positions[static_cast<std::size_t>(Quantity::branch)] == std::vector<Idx>{-1, -1, -1, -1, -1, -1, 0, 1});
    CHECK(positions[static_cast<std::size_t>(Quantity::injection)] == std::vector<Idx>{-1, -1, 0});
    REQUIRE(request.voltage.size() == 2);
    CHECK(request.voltage[0].bus == 0);
    CHECK(request.voltage[1].magnitude);
    CHECK(request.voltage[1].angle);
    REQUIRE(request.branch.size() == 2);
    CHECK(request.branch[0].side == BranchSide::from);
    CHECK(request.branch[0].current_magnitude);
    CHECK(request.branch[1].active_power);
    CHECK(request.branch[1].reactive_power);
    CHECK(request.branch[1].current_angle);
    REQUIRE(request.injection.size() == 1);
    CHECK(request.injection[0].active_power);
    CHECK(request.injection[0].reactive_power);
}

TEST_CASE("Observer request bucketing matches sorting and merging") {
    std::mt19937 engine{5317};
    auto flag = [&] { return engine() % 3 == 0; };
    for (int trial = 0; trial != 20; ++trial) {
        math_solver::StateEstimationUQRequest request;
        Idx const size = 1 + trial * 7;
        for (Idx k = 0; k != 4 * size; ++k) {
            Idx const index = static_cast<Idx>(engine() % size);
            request.voltage.push_back({index, flag(), flag()});
            request.branch.push_back(
                {index, engine() % 2 == 0 ? BranchSide::from : BranchSide::to, flag(), flag(), flag(), flag()});
            request.injection.push_back({index, flag(), flag()});
        }
        // The previous comparison sort with an order-independent union.
        auto reference = [](auto items, auto key, auto merge) {
            std::ranges::sort(items, {}, key);
            std::vector<typename decltype(items)::value_type> result;
            for (auto const& item : items) {
                if (!result.empty() && key(result.back()) == key(item)) {
                    merge(result.back(), item);
                } else {
                    result.push_back(item);
                }
            }
            return result;
        };
        auto const voltage = reference(
            request.voltage, [](auto const& item) { return item.bus; },
            [](auto& target, auto const& source) {
                target.magnitude = target.magnitude || source.magnitude;
                target.angle = target.angle || source.angle;
            });
        auto const branch = reference(
            request.branch, [](auto const& item) { return std::pair{item.branch, item.side}; },
            [](auto& target, auto const& source) {
                target.active_power = target.active_power || source.active_power;
                target.reactive_power = target.reactive_power || source.reactive_power;
                target.current_magnitude = target.current_magnitude || source.current_magnitude;
                target.current_angle = target.current_angle || source.current_angle;
            });
        auto const injection = reference(
            request.injection, [](auto const& item) { return item.bus; },
            [](auto& target, auto const& source) {
                target.active_power = target.active_power || source.active_power;
                target.reactive_power = target.reactive_power || source.reactive_power;
            });
        normalize(request);
        REQUIRE(request.voltage.size() == voltage.size());
        for (std::size_t i = 0; i != voltage.size(); ++i) {
            CHECK(request.voltage[i].bus == voltage[i].bus);
            CHECK(request.voltage[i].magnitude == voltage[i].magnitude);
            CHECK(request.voltage[i].angle == voltage[i].angle);
        }
        REQUIRE(request.branch.size() == branch.size());
        for (std::size_t i = 0; i != branch.size(); ++i) {
            CHECK(request.branch[i].branch == branch[i].branch);
            CHECK(request.branch[i].side == branch[i].side);
            CHECK(request.branch[i].active_power == branch[i].active_power);
            CHECK(request.branch[i].reactive_power == branch[i].reactive_power);
            CHECK(request.branch[i].current_magnitude == branch[i].current_magnitude);
            CHECK(request.branch[i].current_angle == branch[i].current_angle);
        }
        REQUIRE(request.injection.size() == injection.size());
        for (std::size_t i = 0; i != injection.size(); ++i) {
            CHECK(request.injection[i].bus == injection[i].bus);
            CHECK(request.injection[i].active_power == injection[i].active_power);
            CHECK(request.injection[i].reactive_power == injection[i].reactive_power);
        }
    }
}

TEST_CASE("Absent observer output demand does not access topology or allocate island work") {
    MutableDataset data{false, 1, "sym_output", meta_data::meta_data_gen::meta_data};
    using Components = Container<ExtraRetrievableTypes<Base, Node, Observer>, Node, VoltageUncertaintyObserver>;
    MainModelState<Components> state{};
    Scenario<symmetric_t> scenario{data};
    // A null reduced topology is sufficient because absent demand must return before resolving targets.
    CHECK_NOTHROW(prepare_observer_uncertainty(state, scenario, 1000));
    CHECK(scenario.work.empty());
    CHECK(scenario.voltage.empty());
    CHECK(scenario.power.empty());
    CHECK(scenario.current.empty());
}

TEST_CASE("Observer physical conversion preserves double range and missing values") {
    math_solver::UQPhaseValue value;
    value.sigma = 2.0;
    CHECK(physical_sigma(value, 100.0) == 200.0);
    CHECK(is_nan(physical_sigma(value, -1.0)));
    value.sigma = std::numeric_limits<double>::max();
    CHECK(is_nan(physical_sigma(value, 100.0)));
    value.sigma = std::numeric_limits<double>::denorm_min();
    CHECK(is_nan(physical_sigma(value, 0.25)));
    value.sigma = 0.0;
    CHECK(physical_sigma(value, 100.0) == 0.0);
}
TEST_CASE("Asymmetric observer adapter converts each phase once and keeps unrequested quantities missing") {
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
    scenario.voltage.push_back({.island = 0,
                                .result_index = 0,
                                .energized = true,
                                .first = true,
                                .second = true,
                                .magnitude_scale = 100. / sqrt3});
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
    scenario.voltage[0].first = false;
    produce();
    CHECK(output.u_sigma.isNaN().all());
    CHECK(output.u_angle_sigma(1) == .2);
    scenario.voltage[0].result_index = -1;
    produce();
    CHECK(output.energized == 1); // Connectivity is independent of numerical validity.
    CHECK(output.u_sigma.isNaN().all());
    CHECK(output.u_angle_sigma.isNaN().all());
}

} // namespace power_grid_model::main_core
