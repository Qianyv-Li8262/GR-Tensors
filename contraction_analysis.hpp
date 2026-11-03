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

// Scan left to right: unmatched indices are free; opposite indices form a pair.
template <typename Remaining, typename Free, typename Contracted>
struct make_contraction_letters;

template <typename... Free, typename... Contracted>
struct make_contraction_letters<type_list<>, type_list<Free...>, type_list<Contracted...>> {
    using free = type_list<Free...>;
    using contract = type_list<Contracted...>;
};

template <typename Head, typename... Tail, typename... Free, typename... Contracted>
struct make_contraction_letters<type_list<Head, Tail...>, type_list<Free...>, type_list<Contracted...>> {
    static_assert(!is_contain<Head, type_list<Tail...>>::value,
                  "Invalid contraction due to same variance contraction.");
    using opposite_index = typename opposite<Head>::result;
    static constexpr bool contracted = is_contain<opposite_index, type_list<Tail...>>::value;
    using next = make_contraction_letters<
        typename remove_all<opposite_index, type_list<Tail...>>::result,
        std::conditional_t<contracted, type_list<Free...>, type_list<Free..., Head>>,
        std::conditional_t<contracted, type_list<Contracted..., typename index_traits<Head>::letter>,
                           type_list<Contracted...>>>;
    using free = typename next::free;
    using contract = typename next::contract;
};

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

template <typename Indices, typename LetterA, typename LetterB>
struct trace_analysis;

template <typename... Indices, typename LetterA, typename LetterB>
struct trace_analysis<type_list<Indices...>, LetterA, LetterB> {
    using indices = type_list<Indices...>;
    using letters = typename extract_letters<indices>::result;
    static_assert(!std::is_same_v<LetterA, LetterB>, "trace requires two distinct index letters");
    static_assert(is_contain<LetterA, letters>::value && is_contain<LetterB, letters>::value,
                  "trace index letter must exist in the tensor");

    using plan = trace_plan<sizeof...(Indices), find_obj<LetterA, letters>::value,
                           find_obj<LetterB, letters>::value>;
    using idxA = typename nth_type<plan::axis_a, indices>::result;
    using idxB = typename nth_type<plan::axis_b, indices>::result;
    static_assert(!is_same_variance<idxA, idxB>::value, "trace requires opposite index variance");
    using free_indices = typename remove_first<idxB, typename remove_first<idxA, indices>::result>::result;
};
