#pragma once
#include "tensor_class.hpp"
#include <array>
#include <cstddef>
#include <utility>

// Output axis k takes input axis axes[k]. Every input axis must occur once.
template <std::size_t... Axes>
struct index {
    static constexpr std::size_t rank = sizeof...(Axes);
    static constexpr std::size_t size = pow4(rank);
    inline static constexpr std::array<std::size_t, rank> axes{Axes...};
    inline static constexpr bool is_identity = [] {
        for (std::size_t axis = 0; axis < rank; ++axis) {
            if (axes[axis] != axis) return false;
        }
        return true;
    }();
    static_assert(((Axes < rank) && ...), "permutation axis is out of range; provide the complete mapping");
    static constexpr bool unique_axes() {
        for (std::size_t i = 0; i < rank; ++i) {
            for (std::size_t j = i + 1; j < rank; ++j) {
                if (axes[i] == axes[j]) return false;
            }
        }
        return true;
    }
    static_assert(unique_axes(), "permutation axes must not repeat");

    template <std::size_t Output>
    inline static constexpr std::size_t source = [] {
        const auto output = unpack_index<rank>(Output);
        std::array<std::size_t, rank> input{};
        for (std::size_t axis = 0; axis < rank; ++axis) input[axes[axis]] = output[axis];
        return pack_index(input);
    }();
};

namespace permutation_detail {

inline constexpr std::size_t block_size = 128;

template <typename Plan, std::size_t Begin, typename Scalar, std::size_t Size, std::size_t... I>
void copy_block(const std::array<Scalar, Size>& input, std::array<Scalar, Size>& output,
                std::index_sequence<I...>) {
    ((output[Begin + I] = input[Plan::template source<Begin + I>]), ...);
}

template <typename Plan, std::size_t Begin, std::size_t End, typename Scalar, std::size_t Size>
void copy_blocks(const std::array<Scalar, Size>& input, std::array<Scalar, Size>& output) {
    if constexpr (End - Begin <= block_size) {
        copy_block<Plan, Begin>(input, output, std::make_index_sequence<End - Begin>{});
    } else {
        constexpr auto middle = Begin + (End - Begin) / 2;
        copy_blocks<Plan, Begin, middle>(input, output);
        copy_blocks<Plan, middle, End>(input, output);
    }
}

template <typename Scalar, typename Indices, typename Plan>
struct result_tensor;

template <typename Scalar, typename... Indices, std::size_t... Axes>
struct result_tensor<Scalar, type_list<Indices...>, index<Axes...>> {
    static_assert(sizeof...(Indices) == sizeof...(Axes), "permutation must specify every input axis");
    using type = Tensor<Scalar, typename nth_type<Axes, type_list<Indices...>>::result...>;
};

} // namespace permutation_detail

// Reorder already computed components. All source/destination indices are
// compile-time constants; runtime execution only copies component values.
template <typename Plan, typename Scalar, std::size_t Size>
auto permute_components(const std::array<Scalar, Size>& input) {
    static_assert(Size == Plan::size, "permutation must specify every input axis; component count mismatch");
    if constexpr (Plan::is_identity) {
        return input; // Independent return value still requires value-copy semantics.
    } else {
        std::array<Scalar, Size> output;
        if constexpr (Size == Plan::size) permutation_detail::copy_blocks<Plan, 0, Size>(input, output);
        return output;
    }
}

template <typename Scalar, typename... Indices, std::size_t... Axes>
auto permute(const Tensor<Scalar, Indices...>& tensor, index<Axes...>) {
    using plan = index<Axes...>;
    using result_type = typename permutation_detail::result_tensor<Scalar, type_list<Indices...>, plan>::type;
    if constexpr (plan::is_identity) {
        return tensor;
    } else {
        result_type result;
        result.data = permute_components<plan>(tensor.data);
        return result;
    }
}

template <std::size_t... Axes, typename Scalar, typename... Indices>
auto permute(const Tensor<Scalar, Indices...>& tensor) {
    return permute(tensor, index<Axes...>{});
}
