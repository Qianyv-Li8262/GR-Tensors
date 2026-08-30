#pragma once
#include <type_traits>

template <typename... Ts>
struct type_list {};


// 接口统一为：返回一个类型用result，返回一个值用value
template <typename T1, typename T2>
struct append_back {};
template <typename obj, typename... Ts>
struct append_back<type_list<Ts...>, obj> {
    using result = type_list<Ts..., obj>;
};

template <typename T1, typename T2>
struct append_front {};
template <typename obj, typename... Ts>
struct append_front<obj, type_list<Ts...>> {
    using result = type_list<obj, Ts...>;
};

template <typename L1, typename L2>
struct concat;
template <typename... As, typename... Bs>
struct concat<type_list<As...>, type_list<Bs...>> {
    using result = type_list<As..., Bs...>;
};
template <typename l1, typename l2>
struct reverse_impl {};
template <typename... l1>
struct reverse_impl<type_list<l1...>, type_list<>> {
    using result = type_list<l1...>;
};
template <typename... l1, typename head, typename... l2>
struct reverse_impl<type_list<l1...>, type_list<head, l2...>> {
    using result = typename reverse_impl<type_list<head, l1...>, type_list<l2...>>::result;
};
template <typename list1>
struct reverse {};
template <typename... list>
struct reverse<type_list<list...>> {
    using result = typename reverse_impl<type_list<>, type_list<list...>>::result;
};
template <typename List>
struct list_size;
template <typename... Ts>
struct list_size<type_list<Ts...>> : std::integral_constant<std::size_t, sizeof...(Ts)> {};

template <typename obj, typename list>
struct is_contain {};
template <typename obj>
struct is_contain<obj, type_list<>> : std::false_type {};
template <typename obj, typename head, typename... Ts>
struct is_contain<obj, type_list<head, Ts...>>
    : std::conditional_t<std::is_same_v<obj, head>, std::true_type, is_contain<obj, type_list<Ts...>>> {};


template <typename obj, typename orig_list, typename result>
struct remove_all_impl {};
template <typename obj, typename... Ts>
struct remove_all_impl<obj, type_list<>, type_list<Ts...>> {
    using result = type_list<Ts...>;
};
template <typename obj, typename head, typename... Ts1, typename... Ts2>
struct remove_all_impl<obj, type_list<head, Ts1...>, type_list<Ts2...>>
    : std::conditional_t<std::is_same_v<obj, head>, remove_all_impl<obj, type_list<Ts1...>, type_list<Ts2...>>,
                         remove_all_impl<obj, type_list<Ts1...>, type_list<Ts2..., head>>> {};

template <typename obj, typename List>
struct remove_all {
    using result = typename remove_all_impl<obj, List, type_list<>>::result;
};

template <typename obj, typename orig_list, typename result>
struct remove_first_impl {};
template <typename obj, typename... Ts>
struct remove_first_impl<obj, type_list<>, type_list<Ts...>> {
    using result = type_list<Ts...>;
};
template <typename obj, typename head, typename... Ts1, typename... Ts2>
struct remove_first_impl<obj, type_list<head, Ts1...>, type_list<Ts2...>>
    : std::conditional_t<std::is_same_v<obj, head>, concat<type_list<Ts2...>, type_list<Ts1...>>,
                         remove_first_impl<obj, type_list<Ts1...>, type_list<Ts2..., head>>> {};

template <typename obj, typename List>
struct remove_first {
    using result = typename remove_first_impl<obj, List, type_list<>>::result;
};

template <typename T1, typename T2>
struct remove_duplicate {};
template <typename... Ts>
struct remove_duplicate<type_list<>, type_list<Ts...>> {
    using result = type_list<Ts...>;
};
template <typename head, typename... Ts1, typename... Ts2>
struct remove_duplicate<type_list<head, Ts1...>, type_list<Ts2...>> {
    using result =
        typename std::conditional_t<is_contain<head, type_list<Ts2...>>::value,
                                    remove_duplicate<type_list<Ts1...>, type_list<Ts2...>>,
                                    remove_duplicate<type_list<Ts1...>, type_list<Ts2..., head>>>::result;
};

template <typename T1>
struct has_duplicate {};
template <>
struct has_duplicate<type_list<>> {
    static constexpr bool value = false;
};
template <typename head, typename... Ts1>
struct has_duplicate<type_list<head, Ts1...>> {
    static constexpr bool value =
        is_contain<head, type_list<Ts1...>>::value ? true : has_duplicate<type_list<Ts1...>>::value;
};

template <std::size_t N, typename List>
struct nth_type;
template <std::size_t N, typename Head, typename... Tail>
struct nth_type<N, type_list<Head, Tail...>> {
    using result = typename nth_type<N - 1, type_list<Tail...>>::result;
};
template <typename Head, typename... Tail>
struct nth_type<0, type_list<Head, Tail...>> {
    using result = Head;
};

template <int N, typename Letter, typename List>
struct find_obj_impl;
template <int N, typename Letter, typename Head, typename... Tail>
struct find_obj_impl<N, Letter, type_list<Head, Tail...>>
    : std::conditional_t<std::is_same_v<Letter, Head>, std::integral_constant<int, N>,
                         find_obj_impl<N + 1, Letter, type_list<Tail...>>> {};

template <typename letter, typename list>
struct find_obj : find_obj_impl<0, letter, list> {};


template <typename list1, typename list2>
struct is_unorder_same {};
template <>
struct is_unorder_same<type_list<>, type_list<>> {
    static constexpr bool value = true;
};
template <typename... Ts2>
struct is_unorder_same<type_list<>, type_list<Ts2...>> {
    static constexpr bool value = false;
};
template <typename... Ts2>
struct is_unorder_same<type_list<Ts2...>, type_list<>> {
    static constexpr bool value = false;
};
template <typename head, typename... Ts1, typename... Ts2>
struct is_unorder_same<type_list<head, Ts1...>, type_list<Ts2...>> {
    static constexpr bool value =
        is_unorder_same<type_list<Ts1...>, typename remove_all<head, type_list<Ts2...>>::result>::value;
};


template <typename List1, typename List2>
struct rmv_list2_from_list1 {};
template <typename... List1>
struct rmv_list2_from_list1<type_list<List1...>, type_list<>> {
    using result = type_list<List1...>;
};
template <typename... List1, typename Head, typename... Tail>
struct rmv_list2_from_list1<type_list<List1...>, type_list<Head, Tail...>> {
    using result = typename rmv_list2_from_list1<typename remove_first<Head, type_list<List1...>>::result,
                                                 type_list<Tail...>>::result;
};


template <typename all, typename selected, typename result, typename original>
struct list2_fill_as_list1_impl {};
template <typename AnySelected, typename... result_lett, typename... original>
struct list2_fill_as_list1_impl<type_list<>, AnySelected, type_list<result_lett...>, type_list<original...>> {
    using result = type_list<result_lett...>;
};
template <typename head_all, typename... all, typename... result_lett, typename... original>
struct list2_fill_as_list1_impl<type_list<head_all, all...>, type_list<>, type_list<result_lett...>,
                                type_list<original...>> {
    using result = type_list<result_lett..., head_all, all...>;
};
template <typename head_all, typename... all, typename head_selected, typename... selected,
          typename... result_lett, typename... original>
struct list2_fill_as_list1_impl<type_list<head_all, all...>, type_list<head_selected, selected...>,
                                type_list<result_lett...>, type_list<original...>> {
    using next_step = std::conditional_t<
        is_contain<head_all, type_list<original...>>::value,
        list2_fill_as_list1_impl<type_list<all...>, type_list<selected...>,
                                 type_list<result_lett..., head_selected>, type_list<original...>>,
        list2_fill_as_list1_impl<type_list<all...>, type_list<head_selected, selected...>,
                                 type_list<result_lett..., head_all>, type_list<original...>>>;
    using result = typename next_step::result;
};
template <typename all, typename selected>
struct embed_permutation_into_original {};
template <typename... all, typename... selected>
struct embed_permutation_into_original<type_list<all...>, type_list<selected...>> {
    using result = typename list2_fill_as_list1_impl<type_list<all...>, type_list<selected...>, type_list<>,
                                                     type_list<selected...>>::result;
};

template <typename all, typename second>
struct is_head_coincident {};
template <typename... first>
struct is_head_coincident<type_list<first...>, type_list<>> {
    static constexpr bool value = true;
};
template <typename head, typename... first, typename... second>
struct is_head_coincident<type_list<head, first...>, type_list<head, second...>>
    : is_head_coincident<type_list<first...>, type_list<second...>> {};
template <typename head1, typename head2, typename... first, typename... second>
struct is_head_coincident<type_list<head1, first...>, type_list<head2, second...>> {
    static constexpr bool value = false;
};
template <typename head, typename... second>
struct is_head_coincident<type_list<>, type_list<head, second...>> {
    static constexpr bool value = false;
};