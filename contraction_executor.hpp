#pragma once

#include "contraction_plan.hpp"
#include <array>
#include <cstddef>
#include <utility>

namespace contraction_detail {

template <typename Plan, std::size_t... Axes>
std::size_t left_offset(const std::array<std::size_t, Plan::loop_rank>& coordinates,
                        std::index_sequence<Axes...>) {
    std::size_t offset = 0;
    ((offset = offset * 4 + coordinates[Plan::left_slots[Axes]]), ...);
    return offset;
}

template <typename Plan, std::size_t... Axes>
std::size_t right_offset(const std::array<std::size_t, Plan::loop_rank>& coordinates,
                         std::index_sequence<Axes...>) {
    std::size_t offset = 0;
    ((offset = offset * 4 + coordinates[Plan::right_slots[Axes]]), ...);
    return offset;
}

template <typename Plan, std::size_t Depth>
struct sum_loop {
    template <typename Scalar, std::size_t LeftSize, std::size_t RightSize>
    static void exec(const std::array<Scalar, LeftSize>& left, const std::array<Scalar, RightSize>& right,
                     std::array<std::size_t, Plan::loop_rank>& coordinates, Scalar& sum) {
        if constexpr (Depth == Plan::contract_rank) {
            sum += left[left_offset<Plan>(coordinates, std::make_index_sequence<Plan::left_rank>{})] *
                   right[right_offset<Plan>(coordinates, std::make_index_sequence<Plan::right_rank>{})];
        } else {
            for (std::size_t component = 0; component < 4; ++component) {
                coordinates[Plan::result_rank + Depth] = component;
                sum_loop<Plan, Depth + 1>::exec(left, right, coordinates, sum);
            }
        }
    }
};

template <typename Plan, std::size_t Depth>
struct output_loop {
    template <typename Scalar, std::size_t LeftSize, std::size_t RightSize>
    static void exec(const std::array<Scalar, LeftSize>& left, const std::array<Scalar, RightSize>& right,
                     std::array<Scalar, Plan::result_size>& result,
                     std::array<std::size_t, Plan::loop_rank>& coordinates, std::size_t& output_offset) {
        if constexpr (Depth == Plan::result_rank) {
            Scalar sum{};
            sum_loop<Plan, 0>::exec(left, right, coordinates, sum);
            result[output_offset++] = sum;
        } else {
            for (std::size_t component = 0; component < 4; ++component) {
                coordinates[Depth] = component;
                output_loop<Plan, Depth + 1>::exec(left, right, result, coordinates, output_offset);
            }
        }
    }
};

} // namespace contraction_detail

// Layer 2: execute a type-level positional plan on contiguous components.
// No index letters, pair discovery, graph construction, or runtime plan input.
template <typename Plan, typename Scalar, std::size_t LeftSize, std::size_t RightSize>
std::array<Scalar, Plan::result_size> contract_components(const std::array<Scalar, LeftSize>& left,
                                                          const std::array<Scalar, RightSize>& right) {
    static_assert(LeftSize == Plan::left_size, "left component count does not match contraction plan");
    static_assert(RightSize == Plan::right_size, "right component count does not match contraction plan");

    std::array<Scalar, Plan::result_size> result{};
    std::array<std::size_t, Plan::loop_rank> coordinates{};
    std::size_t output_offset = 0;
    contraction_detail::output_loop<Plan, 0>::exec(left, right, result, coordinates, output_offset);
    return result;
}
