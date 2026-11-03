#pragma once

#include "contraction_plan.hpp"
#include <array>
#include <cstddef>

namespace contraction_detail {

template <typename Scalar, std::size_t Size, std::size_t Rank>
const Scalar& component_at(const std::array<Scalar, Size>& data,
                           const std::array<std::size_t, Rank>& index) {
    std::size_t offset = 0;
    for (auto i : index) {
        offset = offset * 4 + i;
    }
    return data[offset];
}

template <typename Plan>
struct contract_impl;

// No pairs left: accumulate one product using the coordinates just filled in.
template <std::size_t LeftRank, std::size_t RightRank>
struct contract_impl<contraction_plan<LeftRank, RightRank>> {
    template <typename Scalar, std::size_t LeftSize, std::size_t RightSize>
    static void exec(const std::array<Scalar, LeftSize>& left, const std::array<Scalar, RightSize>& right,
                     std::array<std::size_t, LeftRank>& indexleft,
                     std::array<std::size_t, RightRank>& indexright, Scalar& sum) {
        sum += component_at(left, indexleft) * component_at(right, indexright);
    }
};

// Consume one pair per step, just as addition consumes one axis per step.
template <std::size_t LeftRank, std::size_t RightRank, typename Pair, typename... Pairs>
struct contract_impl<contraction_plan<LeftRank, RightRank, Pair, Pairs...>> {
    template <typename Scalar, std::size_t LeftSize, std::size_t RightSize>
    static void exec(const std::array<Scalar, LeftSize>& left, const std::array<Scalar, RightSize>& right,
                     std::array<std::size_t, LeftRank>& indexleft,
                     std::array<std::size_t, RightRank>& indexright, Scalar& sum) {
        for (std::size_t i = 0; i < 4; ++i) {
            indexleft[Pair::left] = i;
            indexright[Pair::right] = i;
            contract_impl<contraction_plan<LeftRank, RightRank, Pairs...>>::exec(
                left, right, indexleft, indexright, sum);
        }
    }
};

// Decode an output component into free coordinates, skipping contracted axes.
// Decode right first because its free axes are last in the output.
template <typename Plan, std::size_t Rank>
void set_free_indices(std::size_t& output, std::array<std::size_t, Rank>& index, std::size_t side) {
    for (std::size_t axis = Rank; axis > 0; --axis) {
        if (!Plan::is_contracted(axis - 1, side)) {
            index[axis - 1] = output % 4;
            output /= 4;
        }
    }
}

} // namespace contraction_detail

// Layer 2: execute a type-level positional plan on contiguous components.
// No index letters, pair discovery, graph construction, or runtime plan input.
template <typename Plan, typename Scalar, std::size_t LeftSize, std::size_t RightSize>
std::array<Scalar, Plan::result_size> contract_components(const std::array<Scalar, LeftSize>& left,
                                                        const std::array<Scalar, RightSize>& right) {
    static_assert(LeftSize == Plan::left_size, "left component count does not match contraction plan");
    static_assert(RightSize == Plan::right_size, "right component count does not match contraction plan");

    std::array<Scalar, Plan::result_size> result{};
    std::array<std::size_t, Plan::left_rank> indexleft{};
    std::array<std::size_t, Plan::right_rank> indexright{};
    for (std::size_t output = 0; output < result.size(); ++output) {
        auto remaining = output;
        contraction_detail::set_free_indices<Plan>(remaining, indexright, 1);
        contraction_detail::set_free_indices<Plan>(remaining, indexleft, 0);
        contraction_detail::contract_impl<Plan>::exec(left, right, indexleft, indexright, result[output]);
    }
    return result;
}
