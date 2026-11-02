#pragma once

#include "Index_and_variance.hpp"
#include "contraction_plan.hpp"
#include "type_lists.hpp"
#include <type_traits>

template <typename List>
struct extract_letters;

template <typename... Indices>
struct extract_letters<type_list<Indices...>> {
    using result = type_list<typename index_traits<Indices>::letter...>;
};

// Preserve the existing index checks and left-to-right result/summation order.
template <typename Index, typename List>
struct find_contract_letter;

template <typename Index>
struct find_contract_letter<Index, type_list<>> {
    static constexpr int value = 1;
};

template <typename Index, typename Head, typename... Tail>
struct find_contract_letter<Index, type_list<Head, Tail...>> {
    static constexpr int value =
        std::is_same_v<typename index_traits<Index>::letter, typename index_traits<Head>::letter>
            ? (index_traits<Index>::variance::value == index_traits<Head>::variance::value ? 0 : 2)
            : find_contract_letter<Index, type_list<Tail...>>::value;
    static_assert(value != 0, "Invalid contraction due to same variance contraction.");
};

template <typename Remaining, typename Free, typename Contracted>
struct make_contraction_letters;

template <typename Remaining, typename Free, typename Contracted, int Kind>
struct make_contraction_letters_impl1;

template <typename... Free, typename... Contracted>
struct make_contraction_letters<type_list<>, type_list<Free...>, type_list<Contracted...>> {
    using free = type_list<Free...>;
    using contract = type_list<Contracted...>;
};

template <typename Head, typename... Tail, typename... Free, typename... Contracted>
struct make_contraction_letters_impl1<type_list<Head, Tail...>, type_list<Free...>, type_list<Contracted...>, 1>
    : make_contraction_letters<type_list<Tail...>, type_list<Free..., Head>, type_list<Contracted...>> {};

template <typename Head, typename... Tail, typename... Free, typename... Contracted>
struct make_contraction_letters_impl1<type_list<Head, Tail...>, type_list<Free...>, type_list<Contracted...>, 2>
    : make_contraction_letters<typename remove_all<typename opposite<Head>::result, type_list<Tail...>>::result,
                               type_list<Free...>, type_list<Contracted..., typename index_traits<Head>::letter>> {
};

template <typename Head, typename... Tail, typename... Free, typename... Contracted>
struct make_contraction_letters<type_list<Head, Tail...>, type_list<Free...>, type_list<Contracted...>>
    : make_contraction_letters_impl1<type_list<Head, Tail...>, type_list<Free...>, type_list<Contracted...>,
                                     find_contract_letter<Head, type_list<Tail...>>::value> {};

namespace contraction_detail {

template <typename LeftLetters, typename RightLetters, typename ContractedLetters>
struct plan_from_letters;

template <typename... Left, typename... Right, typename... Contracted>
struct plan_from_letters<type_list<Left...>, type_list<Right...>, type_list<Contracted...>> {
    using type = contraction_plan<
        sizeof...(Left), sizeof...(Right),
        contractpair<static_cast<std::size_t>(find_obj<Contracted, type_list<Left...>>::value),
                     static_cast<std::size_t>(find_obj<Contracted, type_list<Right...>>::value)>...>;
};

} // namespace contraction_detail

// Layer 1: named indices -> checked result indices and a letter-free plan type.
template <typename LeftIndices, typename RightIndices>
struct contraction_analysis;

template <typename... Left, typename... Right>
struct contraction_analysis<type_list<Left...>, type_list<Right...>> {
    static_assert((index_traits<Left>::is_index::value && ...), "left tensor indices must be up<T> or dn<T>");
    static_assert((index_traits<Right>::is_index::value && ...), "right tensor indices must be up<T> or dn<T>");

    using left_letters = typename extract_letters<type_list<Left...>>::result;
    using right_letters = typename extract_letters<type_list<Right...>>::result;
    static_assert(!has_duplicate<left_letters>::value, "left tensor indices must not repeat letters");
    static_assert(!has_duplicate<right_letters>::value, "right tensor indices must not repeat letters");

    using analysis = make_contraction_letters<type_list<Left..., Right...>, type_list<>, type_list<>>;
    using free_indices = typename analysis::free;
    using contract_letters = typename analysis::contract;
    using plan =
        typename contraction_detail::plan_from_letters<left_letters, right_letters, contract_letters>::type;
};
