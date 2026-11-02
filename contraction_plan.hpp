#pragma once

#include "type_lists.hpp"
#include <array>
#include <cstddef>

// Axis numbers are zero-based and local to the left/right operand.
template <std::size_t LeftAxis, std::size_t RightAxis>
struct contractpair {
    static constexpr std::size_t left = LeftAxis;
    static constexpr std::size_t right = RightAxis;
};

namespace contraction_detail {

constexpr std::size_t component_count(std::size_t rank) {
    std::size_t size = 1;
    for (std::size_t axis = 0; axis < rank; ++axis) {
        size *= 4;
    }
    return size;
}

} // namespace contraction_detail

// No index letters and no runtime plan object. Output axes are the free left
// axes followed by the free right axes, each in storage order. Pair order
// determines summation order.
template <std::size_t LeftRank, std::size_t RightRank, typename... Pairs>
struct contraction_plan {
    using pairs = type_list<Pairs...>;
    static constexpr std::size_t left_rank = LeftRank;
    static constexpr std::size_t right_rank = RightRank;
    static constexpr std::size_t contract_rank = sizeof...(Pairs);

private:
    static constexpr bool axes_in_range = ((Pairs::left < LeftRank && Pairs::right < RightRank) && ...);
    static constexpr bool count_in_range = contract_rank <= LeftRank && contract_rank <= RightRank;

    static constexpr bool axes_are_unique() {
        constexpr std::array<std::size_t, contract_rank> left_axes{Pairs::left...};
        constexpr std::array<std::size_t, contract_rank> right_axes{Pairs::right...};
        for (std::size_t i = 0; i < contract_rank; ++i) {
            for (std::size_t j = i + 1; j < contract_rank; ++j) {
                if (left_axes[i] == left_axes[j] || right_axes[i] == right_axes[j]) {
                    return false;
                }
            }
        }
        return true;
    }

public:
    static_assert(axes_in_range, "contraction pair axis is out of range");
    static_assert(count_in_range, "too many contraction pairs for operand ranks");
    static_assert(axes_are_unique(), "an operand axis must occur in at most one contraction pair");

    static constexpr std::size_t result_rank = count_in_range ? LeftRank + RightRank - 2 * contract_rank : 0;
    static constexpr std::size_t loop_rank = result_rank + contract_rank;
    static constexpr std::size_t left_size = contraction_detail::component_count(left_rank);
    static constexpr std::size_t right_size = contraction_detail::component_count(right_rank);
    static constexpr std::size_t result_size = contraction_detail::component_count(result_rank);

private:
    struct layout {
        std::array<std::size_t, LeftRank> left{};
        std::array<std::size_t, RightRank> right{};
    };

    static constexpr layout make_layout() {
        layout result{};
        if constexpr (axes_in_range && count_in_range && axes_are_unique()) {
            // No valid loop variable has the sentinel number loop_rank.
            for (auto& slot : result.left) {
                slot = loop_rank;
            }
            for (auto& slot : result.right) {
                slot = loop_rank;
            }

            std::size_t sum_slot = result_rank;
            ((result.left[Pairs::left] = sum_slot, result.right[Pairs::right] = sum_slot, ++sum_slot), ...);

            std::size_t free_slot = 0;
            for (auto& slot : result.left) {
                if (slot == loop_rank) {
                    slot = free_slot++;
                }
            }
            for (auto& slot : result.right) {
                if (slot == loop_rank) {
                    slot = free_slot++;
                }
            }
        }
        return result;
    }

    inline static constexpr auto layout_ = make_layout();

public:
    // Output coordinates occupy [0, result_rank); summation coordinates follow.
    // These mappings are constant expressions, evaluated during compilation.
    inline static constexpr auto left_slots = layout_.left;
    inline static constexpr auto right_slots = layout_.right;
};
