// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

// Dump the symmetric nodal admittance matrices (Y_bus) that Power Grid Model builds internally.
//
// The tool deserializes a JSON "input" dataset, rebuilds the same internal model state that
// MainModel uses for a calculation, constructs the math_solver::YBus objects through
// main_core::prepare_y_bus, and writes one CSR sub-network object per isolated mathematical
// network to the output JSON file:
//
// {
//   "version": "...",
//   "system_frequency": 50.0,
//   "n_sub_networks": 1,
//   "sub_networks": [
//     {
//       "n_bus": 33,
//       "is_radial": true,
//       "slack_bus": 0,
//       "bus_node_ids": [ ... input node ID per math bus position ... ],
//       "row_indptr": [ ... ],
//       "col_indices": [ ... ],
//       "admittance_real": [ ... ],
//       "admittance_imag": [ ... ]
//     }
//   ]
// }
//
// Usage: power_grid_model_c_example_extract_y_bus input.json output.json [system_frequency]

#include <power_grid_model/main_model.hpp>
#include <power_grid_model_cpp.hpp>

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using power_grid_model::AllComponents;
using power_grid_model::AllExtraRetrievableTypes;
using power_grid_model::ID;
using power_grid_model::Idx;
using power_grid_model::Idx2D;
using power_grid_model::main_core::MainModelType;

using ModelType = MainModelType<AllExtraRetrievableTypes, AllComponents>;

template <std::ranges::input_range Range>
    requires std::is_arithmetic_v<std::ranges::range_value_t<Range>>
void write_json_number_array(std::ostream& stream, Range const& values, std::string_view key, std::string_view indent,
                             bool last) {
    stream << indent << '"' << key << "\": [";
    char const* separator = "";
    for (auto const& value : values) {
        stream << separator << std::format("{}", value);
        separator = ", ";
    }
    stream << ']' << (last ? "" : ",") << '\n';
}

void write_json_string(std::ostream& stream, std::string_view key, std::string_view value, std::string_view indent,
                       bool last) {
    stream << indent << '"' << key << "\": \"" << value << '"' << (last ? "" : ",") << '\n';
}

struct SubNetworkYBus {
    Idx n_bus;
    bool is_radial;
    Idx slack_bus;
    std::vector<ID> bus_node_ids;
    std::vector<Idx> row_indptr;
    std::vector<Idx> col_indices;
    std::vector<double> admittance_real;
    std::vector<double> admittance_imag;
};

std::vector<SubNetworkYBus> extract_y_bus(power_grid_model::ConstDataset const& input_data, double system_frequency) {
    if (input_data.get_description().dataset->name != std::string_view{"input"}) {
        throw std::invalid_argument{"The input dataset must be an \"input\" dataset"};
    }
    if (input_data.is_batch()) {
        throw std::invalid_argument{"Batch datasets are not supported; provide a single \"input\" dataset"};
    }

    // build the internal model state in the same way as MainModelImpl
    ModelType::MainModelState state;
    auto const add_func = [&state, &input_data, system_frequency]<typename CT>() {
        if (input_data.is_columnar(CT::name)) {
            power_grid_model::main_core::add_component<CT>(
                state.components,
                input_data.get_columnar_buffer_span<power_grid_model::meta_data::input_getter_s, CT>(0),
                system_frequency);
        } else {
            power_grid_model::main_core::add_component<CT>(
                state.components, input_data.get_buffer_span<power_grid_model::meta_data::input_getter_s, CT>(0),
                system_frequency);
        }
    };
    ModelType::run_functor_with_all_component_types_return_void(add_func);
    state.components.set_construction_complete();
    state.comp_topo = std::make_shared<power_grid_model::ComponentTopology const>(
        power_grid_model::main_core::construct_topology<ModelType>(state.components));

    // build topology and Y_bus without constructing any solver
    power_grid_model::SolverPreparationContext solver_context{.math_state = {}, .math_solver_dispatcher = nullptr};
    power_grid_model::SolversCacheStatus<ModelType> solvers_cache_status{};
    power_grid_model::detail::rebuild_topology<ModelType>(state, solver_context, solvers_cache_status);
    Idx const n_math_solvers = power_grid_model::get_n_math_solvers<ModelType>(state);
    power_grid_model::main_core::prepare_y_bus<power_grid_model::symmetric_t, ModelType>(state, n_math_solvers,
                                                                                         solver_context.math_state);

    // couple topological node sequence to input node ID
    Idx const n_node = state.components.template size<power_grid_model::Node>();
    std::vector<ID> node_ids(n_node);
    std::vector<Idx2D> node_math_id(n_node);
    for (Idx node_seq = 0; node_seq != n_node; ++node_seq) {
        node_ids[node_seq] = state.components.template get_item_by_seq<power_grid_model::Node>(node_seq).id();
        node_math_id[node_seq] = state.topo_comp_coup->node[node_seq];
    }

    auto const& y_bus_vec =
        power_grid_model::main_core::get_y_bus<power_grid_model::symmetric_t>(solver_context.math_state);
    std::vector<SubNetworkYBus> sub_networks;
    sub_networks.reserve(y_bus_vec.size());
    for (Idx math_solver_idx = 0; math_solver_idx != std::ssize(y_bus_vec); ++math_solver_idx) {
        auto const& y_bus = y_bus_vec[math_solver_idx];
        auto const& math_topology = y_bus.math_topology();

        SubNetworkYBus sub_network{};
        sub_network.n_bus = y_bus.size();
        sub_network.is_radial = math_topology.is_radial;
        sub_network.slack_bus = math_topology.slack_bus;
        sub_network.bus_node_ids.assign(sub_network.n_bus, power_grid_model::na_IntID);
        for (Idx node_seq = 0; node_seq != n_node; ++node_seq) {
            Idx2D const math_id = node_math_id[node_seq];
            if (math_id.group == math_solver_idx) {
                sub_network.bus_node_ids[math_id.pos] = node_ids[node_seq];
            }
        }
        if (std::ranges::any_of(sub_network.bus_node_ids, [](ID id) { return id == power_grid_model::na_IntID; })) {
            throw std::runtime_error{"Internal error: not every math bus is coupled to a topological node"};
        }

        auto const& row_indptr = y_bus.row_indptr();
        auto const& col_indices = y_bus.col_indices();
        sub_network.row_indptr.assign(row_indptr.begin(), row_indptr.end());
        sub_network.col_indices.assign(col_indices.begin(), col_indices.end());
        for (auto const& entry_admittance : y_bus.admittance()) {
            sub_network.admittance_real.push_back(entry_admittance.real());
            sub_network.admittance_imag.push_back(entry_admittance.imag());
        }
        sub_networks.push_back(std::move(sub_network));
    }
    return sub_networks;
}

void write_output(std::filesystem::path const& output_path, std::vector<SubNetworkYBus> const& sub_networks,
                  double system_frequency) {
    std::ofstream stream{output_path};
    if (!stream) {
        throw std::runtime_error{"Failed to open output file: " + output_path.string()};
    }
    std::string_view outer_indent = "  ";
    std::string_view inner_indent = "    ";
    stream << "{\n";
    write_json_string(stream, "version", PGM_VERSION, outer_indent, false);
    stream << outer_indent << "\"system_frequency\": " << std::format("{}", system_frequency) << ",\n";
    stream << outer_indent << "\"n_sub_networks\": " << sub_networks.size() << ",\n";
    stream << outer_indent << "\"sub_networks\": [\n";
    for (Idx sub_network_idx = 0; sub_network_idx != std::ssize(sub_networks); ++sub_network_idx) {
        auto const& sub_network = sub_networks[sub_network_idx];
        stream << outer_indent << "  {\n";
        stream << inner_indent << "\"n_bus\": " << sub_network.n_bus << ",\n";
        stream << inner_indent << "\"is_radial\": " << (sub_network.is_radial ? "true" : "false") << ",\n";
        stream << inner_indent << "\"slack_bus\": " << sub_network.slack_bus << ",\n";
        write_json_number_array(stream, sub_network.bus_node_ids, "bus_node_ids", inner_indent, false);
        write_json_number_array(stream, sub_network.row_indptr, "row_indptr", inner_indent, false);
        write_json_number_array(stream, sub_network.col_indices, "col_indices", inner_indent, false);
        write_json_number_array(stream, sub_network.admittance_real, "admittance_real", inner_indent, false);
        write_json_number_array(stream, sub_network.admittance_imag, "admittance_imag", inner_indent, true);
        stream << outer_indent << "  }" << (sub_network_idx + 1 != std::ssize(sub_networks) ? "," : "") << '\n';
    }
    stream << outer_indent << "]\n";
    stream << "}\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " input.json output.json [system_frequency]\n";
        return 1;
    }
    try {
        std::filesystem::path const input_path{argv[1]};
        std::filesystem::path const output_path{argv[2]};
        double const system_frequency = argc == 4 ? std::stod(argv[3]) : 50.0;

        auto const owning_dataset = power_grid_model_cpp::load_dataset(input_path, PGM_json);
        power_grid_model_cpp::DatasetConst const const_dataset{owning_dataset.dataset};
        auto const& input_data = *reinterpret_cast<power_grid_model::ConstDataset const*>(const_dataset.get());

        auto const sub_networks = extract_y_bus(input_data, system_frequency);
        write_output(output_path, sub_networks, system_frequency);
    } catch (std::exception const& exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }
    return 0;
}
