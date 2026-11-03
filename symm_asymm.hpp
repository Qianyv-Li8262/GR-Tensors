#pragma once
#include "tensor_class.hpp"
#include "type_lists.hpp"
#include <array>
#include <type_traits>
#include <utility>

// Positional plans contain no index letters. Axes are zero-based.
template <std::size_t Rank, std::size_t... Axes>
struct sym_plan {
    using positions = std::index_sequence<Axes...>;
    inline static constexpr std::array<std::size_t, sizeof...(Axes)> axes{Axes...};
    static constexpr std::size_t input_rank = Rank;
    static constexpr std::size_t result_rank = Rank;
    static constexpr std::size_t input_size = pow4(Rank);
    static constexpr std::size_t result_size = input_size;
    static constexpr std::size_t selected_count = sizeof...(Axes);
    static constexpr bool antisymmetric = false;
    static_assert(selected_count >= 2, "symmetry requires at least two axes");
    static_assert(((Axes < Rank) && ...), "symmetry axis is out of range");
    static constexpr bool unique_axes() {
        for (std::size_t i = 0; i < axes.size(); ++i) {
            for (std::size_t j = i + 1; j < axes.size(); ++j) {
                if (axes[i] == axes[j]) return false;
            }
        }
        return true;
    }
    static_assert(unique_axes(), "symmetry axes must not repeat");
};

template <std::size_t Rank, std::size_t... Axes>
struct asym_plan : sym_plan<Rank, Axes...> {
    static constexpr bool antisymmetric = true;
};

namespace symmetry_detail {

// Layer 1: validate selected index variances and generate the positional plan.
template <typename Indices, template <std::size_t, std::size_t...> class Plan, std::size_t... Axes>
struct analysis;

template <typename... Indices, template <std::size_t, std::size_t...> class Plan, std::size_t... Axes>
struct analysis<type_list<Indices...>, Plan, Axes...> {
    using free_indices = type_list<Indices...>;
    using plan = Plan<sizeof...(Indices), Axes...>;
    static_assert(sizeof(plan) > 0, "symmetry plan must be valid");
    static constexpr bool same_variance() {
        if constexpr (sizeof...(Axes) > 0 && ((Axes < sizeof...(Indices)) && ...)) {
            constexpr std::array<bool, sizeof...(Indices)> variance{index_traits<Indices>::variance::value...};
            return ((variance[Axes] == variance[plan::axes[0]]) && ...);
        } else {
            return true; // The plan provides the axis/count diagnostic.
        }
    }
    static_assert(same_variance(), "symmetry axes must have the same variance");
};

} // namespace symmetry_detail

template <typename Indices, std::size_t... Axes>
using sym_analysis = symmetry_detail::analysis<Indices, sym_plan, Axes...>;

template <typename Indices, std::size_t... Axes>
using asym_analysis = symmetry_detail::analysis<Indices, asym_plan, Axes...>;

template <std::size_t N>
constexpr std::array<std::size_t, N> sort_small(std::array<std::size_t, N> values) {
    for (std::size_t i = 1; i < N; ++i) {
        const std::size_t value = values[i];
        std::size_t j = i;

        while (j > 0 && values[j - 1] > value) {
            values[j] = values[j - 1];
            --j;
        }

        values[j] = value;
    }

    return values;
}

template <std::size_t Rank, std::size_t... Positions>
constexpr std::size_t canonical_index(std::size_t lin, std::index_sequence<Positions...>) {
    auto full = unpack_index<Rank>(lin);

    auto selected = sort_small(std::array<std::size_t, sizeof...(Positions)>{full[Positions]...});

    std::size_t cursor = 0;
    ((full[Positions] = selected[cursor++]), ...);

    return pack_index(full);
}

template <std::size_t Rank, std::size_t... Positions>
constexpr std::size_t orbit_size(std::size_t lin, std::index_sequence<Positions...>) {
    const auto index = unpack_index<Rank>(lin);
    std::array<std::size_t, 4> counts{};

    ((++counts[index[Positions]]), ...);

    std::size_t result = factorial(sizeof...(Positions));
    for (std::size_t count : counts) {
        result /= factorial(count);
    }

    return result;
}

template <std::size_t Rank, std::size_t... Positions>
constexpr int permutation_sign(std::size_t lin, std::index_sequence<Positions...>) {
    const auto index = unpack_index<Rank>(lin);
    const std::array<std::size_t, sizeof...(Positions)> selected{index[Positions]...};
    int sign = 1;

    for (std::size_t i = 0; i < selected.size(); ++i) {
        for (std::size_t j = i + 1; j < selected.size(); ++j) {
            if (selected[i] == selected[j]) {
                return 0;
            }
            if (selected[i] > selected[j]) {
                sign = -sign;
            }
        }
    }

    return sign;
}

namespace symmetry_detail {

struct component_mapping {
    std::size_t representative{};
    int sign{};
    std::size_t normalizer{};
};

// Evaluate bounded blocks separately so larger tensors do not exhaust a
// compiler's constexpr evaluation budget. This does not add formula nodes.
inline constexpr std::size_t mapping_block_size = 128;

template <typename Plan, std::size_t Block>
inline constexpr auto mapping_block = [] {
    constexpr std::size_t begin = Block * mapping_block_size;
    constexpr std::size_t size = Plan::input_size - begin < mapping_block_size
                                     ? Plan::input_size - begin : mapping_block_size;
    std::array<component_mapping, size> result{};
    using positions = typename Plan::positions;
    for (std::size_t i = 0; i < size; ++i) {
        const auto flat = begin + i;
        const auto representative = canonical_index<Plan::input_rank>(flat, positions{});
        const int sign = Plan::antisymmetric ? permutation_sign<Plan::input_rank>(flat, positions{}) : 1;
        const auto normalizer = flat == representative && sign != 0
                                    ? (Plan::antisymmetric ? factorial(Plan::selected_count)
                                                          : orbit_size<Plan::input_rank>(flat, positions{})) : 0;
        result[i] = {representative, sign, normalizer};
    }
    return result;
}();

enum class stage { accumulate, normalize, broadcast };

// The map is consumed at compile time, producing fixed source/destination
// subscripts. No map loads or sign/representative decisions remain at runtime.
template <typename Plan, stage Stage, std::size_t Block, typename Scalar, std::size_t Size, std::size_t... I>
void execute_block(const std::array<Scalar, Size>& source, std::array<Scalar, Size>& result,
                   std::index_sequence<I...>) {
    auto step = [&](auto position) {
        constexpr auto local = decltype(position)::value;
        constexpr auto flat = Block * mapping_block_size + local;
        constexpr auto entry = mapping_block<Plan, Block>[local];
        if constexpr (Stage == stage::accumulate) {
            if constexpr (entry.sign > 0) result[entry.representative] += source[flat];
            else if constexpr (entry.sign < 0) result[entry.representative] -= source[flat];
        } else if constexpr (Stage == stage::normalize) {
            if constexpr (entry.normalizer > 1) {
                if constexpr (std::is_floating_point_v<Scalar>) {
                    constexpr Scalar weight = static_cast<Scalar>(1) / static_cast<Scalar>(entry.normalizer);
                    result[flat] *= weight;
                } else {
                    result[flat] /= static_cast<Scalar>(entry.normalizer);
                }
            }
        } else {
            if constexpr (entry.sign > 0 && flat != entry.representative) result[flat] = result[entry.representative];
            else if constexpr (entry.sign < 0) result[flat] = -result[entry.representative];
        }
    };
    (step(std::integral_constant<std::size_t, I>{}), ...);
}

// Bounded folds and a balanced block traversal keep template/expression depth
// logarithmic in the component count instead of one recursive level per value.
template <typename Plan, stage Stage, std::size_t Begin, std::size_t End, typename Scalar, std::size_t Size>
void execute_blocks(const std::array<Scalar, Size>& source, std::array<Scalar, Size>& result) {
    if constexpr (End - Begin == 1) {
        execute_block<Plan, Stage, Begin>(source, result,
            std::make_index_sequence<mapping_block<Plan, Begin>.size()>{});
    } else {
        constexpr auto middle = Begin + (End - Begin) / 2;
        execute_blocks<Plan, Stage, Begin, middle>(source, result);
        execute_blocks<Plan, Stage, middle, End>(source, result);
    }
}

// Layer 2 executes only the data operations specified by the compile-time plan.
template <typename Plan, typename Scalar, std::size_t Size>
auto project_components(const std::array<Scalar, Size>& source) {
    static_assert(Size == Plan::input_size, "input component count does not match symmetry plan");
    std::array<Scalar, Plan::result_size> result{};
    if constexpr (Plan::antisymmetric && Plan::selected_count > 4) {
        return result; // Alternating more than four axes vanishes in four dimensions.
    } else {
        constexpr auto blocks = (Size + mapping_block_size - 1) / mapping_block_size;
        execute_blocks<Plan, stage::accumulate, 0, blocks>(source, result);
        execute_blocks<Plan, stage::normalize, 0, blocks>(source, result);
        execute_blocks<Plan, stage::broadcast, 0, blocks>(source, result);
        return result;
    }
}

} // namespace symmetry_detail

template <typename Plan, typename Scalar, std::size_t Size>
auto sym_components(const std::array<Scalar, Size>& source) {
    static_assert(!Plan::antisymmetric, "sym_components requires a sym_plan");
    return symmetry_detail::project_components<Plan>(source);
}

template <typename Plan, typename Scalar, std::size_t Size>
auto asym_components(const std::array<Scalar, Size>& source) {
    static_assert(Plan::antisymmetric, "asym_components requires an asym_plan");
    return symmetry_detail::project_components<Plan>(source);
}

template <std::size_t... Axes, typename Scalar, typename... Indices>
auto sym(const Tensor<Scalar, Indices...>& tensor) {
    using plan = typename sym_analysis<type_list<Indices...>, Axes...>::plan;
    Tensor<Scalar, Indices...> result;
    result.data = sym_components<plan>(tensor.data);
    return result;
}

template <std::size_t... Axes, typename Scalar, typename... Indices>
auto asym(const Tensor<Scalar, Indices...>& tensor) {
    using plan = typename asym_analysis<type_list<Indices...>, Axes...>::plan;
    Tensor<Scalar, Indices...> result;
    result.data = asym_components<plan>(tensor.data);
    return result;
}

template <typename... symm_idxs, typename datatype, typename... idxs>
auto symm_orbit(const Tensor<datatype, idxs...>& T) {
    static_assert(sizeof...(symm_idxs) > 1, "symm requires at least two indices");
    using symm_list = type_list<symm_idxs...>;
    using original_list = type_list<idxs...>;
    static_assert((index_traits<symm_idxs>::is_index::value && ...),
                  "symm arguments must be up<T> or dn<T> indices");
    static_assert(!has_duplicate<symm_list>::value, "symm arguments must not repeat indices");
    static_assert((is_contain<symm_idxs, original_list>::value && ...),
                  "symm arguments must belong to the tensor index list");
    using first_symm_idx = typename nth_type<0, symm_list>::result;
    static_assert((is_same_variance<first_symm_idx, symm_idxs>::value && ...),
                  "symm arguments must have the same variance");

    using plan = sym_plan<sizeof...(idxs), find_obj<symm_idxs, original_list>::value...>;
    Tensor<datatype, idxs...> result{};
    result.data = sym_components<plan>(T.data);
    return result;
}

template <typename... asymm_idxs, typename datatype, typename... idxs>
auto asymm_orbit(const Tensor<datatype, idxs...>& T) {
    static_assert(sizeof...(asymm_idxs) > 1, "asymm requires at least two indices");
    using asymm_list = type_list<asymm_idxs...>;
    using original_list = type_list<idxs...>;
    static_assert((index_traits<asymm_idxs>::is_index::value && ...),
                  "asymm arguments must be up<T> or dn<T> indices");
    static_assert(!has_duplicate<asymm_list>::value, "asymm arguments must not repeat indices");
    static_assert((is_contain<asymm_idxs, original_list>::value && ...),
                  "asymm arguments must belong to the tensor index list");
    using first_asymm_idx = typename nth_type<0, asymm_list>::result;
    static_assert((is_same_variance<first_asymm_idx, asymm_idxs>::value && ...),
                  "asymm arguments must have the same variance");

    using plan = asym_plan<sizeof...(idxs), find_obj<asymm_idxs, original_list>::value...>;
    Tensor<datatype, idxs...> result{};
    result.data = asym_components<plan>(T.data);
    return result;
}
