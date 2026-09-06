#pragma once
#include "tensor_class.hpp"
#include "type_lists.hpp"
#include <array>
#include <type_traits>
#include <utility>

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

template <std::size_t Rank, std::size_t LinIdx, typename Positions>
struct do_orbit_symm;

template <std::size_t Rank, std::size_t LinIdx, std::size_t... Positions>
struct do_orbit_symm<Rank, LinIdx, std::index_sequence<Positions...>> {
    using positions = std::index_sequence<Positions...>;

    template <typename datatype, typename... Indices>
    static void accumulate(const Tensor<datatype, Indices...>& source, Tensor<datatype, Indices...>& result) {
        constexpr std::size_t representative = canonical_index<Rank>(LinIdx, positions{});
        result.data[representative] += source.data[LinIdx];

        if constexpr (LinIdx + 1 < pow4(Rank)) {
            do_orbit_symm<Rank, LinIdx + 1, positions>::accumulate(source, result);
        }
    }

    template <typename datatype, typename... Indices>
    static void normalize(Tensor<datatype, Indices...>& result) {
        constexpr std::size_t representative = canonical_index<Rank>(LinIdx, positions{});

        if constexpr (representative == LinIdx) {
            constexpr std::size_t size = orbit_size<Rank>(LinIdx, positions{});
            result.data[LinIdx] = result.data[LinIdx] * (static_cast<datatype>(1) / static_cast<datatype>(size));
        }

        if constexpr (LinIdx + 1 < pow4(Rank)) {
            do_orbit_symm<Rank, LinIdx + 1, positions>::normalize(result);
        }
    }

    template <typename datatype, typename... Indices>
    static void broadcast(Tensor<datatype, Indices...>& result) {
        constexpr std::size_t representative = canonical_index<Rank>(LinIdx, positions{});
        result.data[LinIdx] = result.data[representative];

        if constexpr (LinIdx + 1 < pow4(Rank)) {
            do_orbit_symm<Rank, LinIdx + 1, positions>::broadcast(result);
        }
    }
};

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

    using positions = typename locate_find_in_list<symm_list, original_list>::result;
    Tensor<datatype, idxs...> result{};

    do_orbit_symm<sizeof...(idxs), 0, positions>::accumulate(T, result);
    do_orbit_symm<sizeof...(idxs), 0, positions>::normalize(result);
    do_orbit_symm<sizeof...(idxs), 0, positions>::broadcast(result);

    return result;
}

template <std::size_t Rank, std::size_t LinIdx, typename Positions>
struct do_orbit_asymm;

template <std::size_t Rank, std::size_t LinIdx, std::size_t... Positions>
struct do_orbit_asymm<Rank, LinIdx, std::index_sequence<Positions...>> {
    using positions = std::index_sequence<Positions...>;

    template <typename datatype, typename... Indices>
    static void accumulate(const Tensor<datatype, Indices...>& source, Tensor<datatype, Indices...>& result) {
        constexpr int sign = permutation_sign<Rank>(LinIdx, positions{});

        if constexpr (sign != 0) {
            constexpr std::size_t representative = canonical_index<Rank>(LinIdx, positions{});
            if constexpr (sign > 0) {
                result.data[representative] += source.data[LinIdx];
            } else {
                result.data[representative] -= source.data[LinIdx];
            }
        }

        if constexpr (LinIdx + 1 < pow4(Rank)) {
            do_orbit_asymm<Rank, LinIdx + 1, positions>::accumulate(source, result);
        }
    }

    template <typename datatype, typename... Indices>
    static void normalize(Tensor<datatype, Indices...>& result) {
        constexpr std::size_t representative = canonical_index<Rank>(LinIdx, positions{});
        constexpr int sign = permutation_sign<Rank>(LinIdx, positions{});

        if constexpr (representative == LinIdx && sign != 0) {
            result.data[LinIdx] = result.data[LinIdx] * (static_cast<datatype>(1) /
                                                         static_cast<datatype>(factorial(sizeof...(Positions))));
        }

        if constexpr (LinIdx + 1 < pow4(Rank)) {
            do_orbit_asymm<Rank, LinIdx + 1, positions>::normalize(result);
        }
    }

    template <typename datatype, typename... Indices>
    static void broadcast(Tensor<datatype, Indices...>& result) {
        constexpr int sign = permutation_sign<Rank>(LinIdx, positions{});

        if constexpr (sign != 0) {
            constexpr std::size_t representative = canonical_index<Rank>(LinIdx, positions{});
            if constexpr (sign > 0) {
                result.data[LinIdx] = result.data[representative];
            } else {
                result.data[LinIdx] = -result.data[representative];
            }
        }

        if constexpr (LinIdx + 1 < pow4(Rank)) {
            do_orbit_asymm<Rank, LinIdx + 1, positions>::broadcast(result);
        }
    }
};

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

    using positions = typename locate_find_in_list<asymm_list, original_list>::result;
    Tensor<datatype, idxs...> result{};

    do_orbit_asymm<sizeof...(idxs), 0, positions>::accumulate(T, result);
    do_orbit_asymm<sizeof...(idxs), 0, positions>::normalize(result);
    do_orbit_asymm<sizeof...(idxs), 0, positions>::broadcast(result);

    return result;
}
