// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0
#pragma once
#include <array>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#ifndef PGM_INJECTION_SCHEME
#define PGM_INJECTION_SCHEME 0
#endif
namespace power_grid_model::injection_experiment {
inline constexpr int scheme = PGM_INJECTION_SCHEME;
static_assert(scheme >= 0 && scheme < 4);
inline constexpr std::array names{"full", "zero", "positive", "bound"};
inline bool enabled() { return std::getenv("PGM_INJECTION_DIAGNOSTICS") != nullptr; }
inline void write(nlohmann::json const& record) {
    auto const* path = std::getenv("PGM_INJECTION_DIAGNOSTICS");
    if (path == nullptr) {
        return;
    }
    std::ofstream stream{path, std::ios::app};
    if (!(stream << record.dump() << '\n')) {
        throw std::runtime_error{"Cannot write injection diagnostics"};
    }
}
} // namespace power_grid_model::injection_experiment
