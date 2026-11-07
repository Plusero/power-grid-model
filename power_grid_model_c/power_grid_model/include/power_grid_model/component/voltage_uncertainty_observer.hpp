// SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
//
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "observer.hpp"

namespace power_grid_model {

class VoltageUncertaintyObserver : public Observer {
  public:
    static constexpr char const* name = "voltage_uncertainty_observer";
    using InputType = VoltageUncertaintyObserverInput;
    template <symmetry_tag sym>
    using OutputType =
        std::conditional_t<is_symmetric_v<sym>, VoltageUncertaintyObserverOutput, AsymVoltageUncertaintyObserverOutput>;

    explicit VoltageUncertaintyObserver(InputType const& input) : Observer{input} {}

    // The adapter supplies already converted physical sigmas; numerical UQ stays outside components.
    template <symmetry_tag sym = symmetric_t>
    OutputType<sym> get_output(bool is_connected_to_source, RealValue<sym> const& u_sigma,
                               RealValue<sym> const& u_angle_sigma) const {
        if (!energized(is_connected_to_source)) {
            return get_null_output<sym>();
        }
        return {.id = id(), .energized = 1, .u_sigma = u_sigma, .u_angle_sigma = u_angle_sigma};
    }

    template <symmetry_tag sym> OutputType<sym> get_null_output() const { return {.id = id(), .energized = 0}; }
};

} // namespace power_grid_model
