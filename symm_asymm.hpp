#pragma once
#include "contraction_and_arithmetic.hpp"
#include "tensor_class.hpp"
#include <array>
#include <cstdlib>
#include <type_traits>
#include <utility>


template <typename idx_list>
struct rename_by_list;
template <typename... NewIdx>
struct rename_by_list<type_list<NewIdx...>> {
    template <typename datatype, typename... OldIdx>
    static auto exec(const Tensor<datatype, OldIdx...>& T) {
        return rename<NewIdx...>(T);
    }
};

template <typename list1, typename list2, typename original_list>
struct calc_symm_tree {};
template <typename... peeled_idxs, typename... original_idxs>
struct calc_symm_tree<type_list<peeled_idxs...>, type_list<>, type_list<original_idxs...>> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        using reordered_idxs =
            typename list2_fill_as_list_1<type_list<original_idxs...>, type_list<peeled_idxs...>>::result;

        return rename_by_list<reordered_idxs>::exec(T);
    }
};
template <typename peeled_list, typename rest_list, typename choices_list, typename original_list>
struct loop_symm_tree {};
template <typename... peeled_idxs, typename... idxs, typename choice, typename... original_idxs>
struct loop_symm_tree<type_list<peeled_idxs...>, type_list<idxs...>, type_list<choice>,
                      type_list<original_idxs...>> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        using new_peeled = typename append_back<type_list<peeled_idxs...>, choice>::result;

        using new_idxs = typename remove_first<choice, type_list<idxs...>>::result;

        return calc_symm_tree<new_peeled, new_idxs, type_list<original_idxs...>>::exec(T);
    }
};
template <typename... peeled_idxs, typename... idxs, typename choice, typename... choice_list,
          typename... original_idxs>
struct loop_symm_tree<type_list<peeled_idxs...>, type_list<idxs...>, type_list<choice, choice_list...>,
                      type_list<original_idxs...>> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        using new_peeled = typename append_back<type_list<peeled_idxs...>, choice>::result;

        using new_idxs = typename remove_first<choice, type_list<idxs...>>::result;

        auto S = calc_symm_tree<new_peeled, new_idxs, type_list<original_idxs...>>::exec(T);

        auto U = loop_symm_tree<type_list<peeled_idxs...>, type_list<idxs...>, type_list<choice_list...>,
                                type_list<original_idxs...>>::exec(T);

        return S + U;
    }
};

template <typename... Peeled, typename... Rest, typename... Original>
struct calc_symm_tree<type_list<Peeled...>, type_list<Rest...>, type_list<Original...>> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        return loop_symm_tree<type_list<Peeled...>, type_list<Rest...>, type_list<Rest...>,
                              type_list<Original...>>::exec(T);
    }
};


template <typename list1, typename list2, typename original_list>
struct calc_asymm_tree {};
template <typename... peeled_idxs, typename... original_idxs>
struct calc_asymm_tree<type_list<peeled_idxs...>, type_list<>, type_list<original_idxs...>> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        using reordered_idxs =
            typename list2_fill_as_list_1<type_list<original_idxs...>, type_list<peeled_idxs...>>::result;

        return rename_by_list<reordered_idxs>::exec(T);
    }
};
template <typename peeled_list, typename rest_list, typename choices_list, typename original_list>
struct loop_asymm_tree {};
template <typename... peeled_idxs, typename... idxs, typename choice, typename... original_idxs>
struct loop_asymm_tree<type_list<peeled_idxs...>, type_list<idxs...>, type_list<choice>,
                       type_list<original_idxs...>> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        using new_peeled = typename append_back<type_list<peeled_idxs...>, choice>::result;

        using new_idxs = typename remove_first<choice, type_list<idxs...>>::result;

        return calc_asymm_tree<new_peeled, new_idxs, type_list<original_idxs...>>::exec(T);
    }
};
template <typename... peeled_idxs, typename... idxs, typename choice, typename... choice_list,
          typename... original_idxs>
struct loop_asymm_tree<type_list<peeled_idxs...>, type_list<idxs...>, type_list<choice, choice_list...>,
                       type_list<original_idxs...>> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        using new_peeled = typename append_back<type_list<peeled_idxs...>, choice>::result;

        using new_idxs = typename remove_first<choice, type_list<idxs...>>::result;

        auto S = calc_asymm_tree<new_peeled, new_idxs, type_list<original_idxs...>>::exec(T);

        auto U = loop_asymm_tree<type_list<peeled_idxs...>, type_list<idxs...>, type_list<choice_list...>,
                                 type_list<original_idxs...>>::exec(T);

        return S - U;
    }
};

template <typename... Peeled, typename... Rest, typename... Original>
struct calc_asymm_tree<type_list<Peeled...>, type_list<Rest...>, type_list<Original...>> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        return loop_asymm_tree<type_list<Peeled...>, type_list<Rest...>, type_list<Rest...>,
                               type_list<Original...>>::exec(T);
    }
};


template <typename... symm_idxs, typename datatype, typename... idxs>
auto symm(const Tensor<datatype, idxs...>& T) {
    static_assert(sizeof...(symm_idxs) > 1, "symm requires at least two indices");
    using symm_list = type_list<symm_idxs...>;
    using original_list = type_list<idxs...>;
    auto sum = calc_symm_tree<type_list<>, symm_list, original_list>::exec(T);
    return sum * (static_cast<datatype>(1) / static_cast<datatype>(factorial(sizeof...(symm_idxs))));
}

template <typename... asymm_idxs, typename datatype, typename... idxs>
auto asymm(const Tensor<datatype, idxs...>& T) {
    static_assert(sizeof...(asymm_idxs) > 1, "asymm requires at least two indices");
    using asymm_list = type_list<asymm_idxs...>;
    using original_list = type_list<idxs...>;
    auto sum = calc_asymm_tree<type_list<>, asymm_list, original_list>::exec(T);
    return sum * (static_cast<datatype>(1) / static_cast<datatype>(factorial(sizeof...(asymm_idxs))));
}
