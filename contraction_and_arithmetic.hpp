#pragma once
#include "Index_and_variance.hpp"
#include "contraction_analysis.hpp"
#include "contraction_executor.hpp"
#include "tensor_class.hpp"
#include "type_lists.hpp"
#include <array>
#include <cstdlib>
#include <type_traits>
#include <utility>
template <typename datatype, typename List>
struct construct_tensor_from_list;
template <typename datatype, typename... Indices>
struct construct_tensor_from_list<datatype, type_list<Indices...>> {
    using result = Tensor<datatype, Indices...>;
};

template <typename Letter, typename List>
struct find_index_from_letter;

template <bool Matches, typename Letter, typename Head, typename... Tail>
struct find_index_from_letter_impl;

template <typename Letter, typename Head, typename... Tail>
struct find_index_from_letter_impl<true, Letter, Head, Tail...> {
    using result = Head;
};

template <typename Letter, typename Head, typename... Tail>
struct find_index_from_letter_impl<false, Letter, Head, Tail...> {
    using result = typename find_index_from_letter<Letter, type_list<Tail...>>::result;
};

template <typename Letter>
struct find_index_from_letter<Letter, type_list<>> {
    static_assert(!std::is_same_v<Letter, Letter>, "trace index letter must exist in the tensor");
};

template <typename Letter, typename Head, typename... Tail>
struct find_index_from_letter<Letter, type_list<Head, Tail...>>
    : find_index_from_letter_impl<std::is_same_v<Letter, typename index_traits<Head>::letter>, Letter, Head,
                                  Tail...> {};

template <typename LIndicies, typename RIndicies, typename con_letters>
struct is_gemm {};
template <typename... LIndicies, typename... RIndicies, typename... con_letters>
struct is_gemm<type_list<LIndicies...>, type_list<RIndicies...>, type_list<con_letters...>> {
    using Lletts = typename extract_letters<type_list<LIndicies...>>::result;
    using Rletts = typename extract_letters<type_list<RIndicies...>>::result;
    using Con = type_list<con_letters...>;
    using RevL = typename reverse<Lletts>::result;
    using RevR = typename reverse<Rletts>::result;
    using RevCon = typename reverse<Con>::result;
    static constexpr bool lefthead = is_head_coincident<Lletts, Con>::value;
    static constexpr bool lefttail = is_head_coincident<RevL, RevCon>::value;
    static constexpr bool righthead = is_head_coincident<Rletts, Con>::value;
    static constexpr bool righttail = is_head_coincident<RevR, RevCon>::value;
    static constexpr bool value = (lefthead || lefttail) && (righthead || righttail);
    static constexpr bool trans_A = !lefttail;
    static constexpr bool trans_B = !righthead;
    static constexpr std::size_t contract_rank = sizeof...(con_letters);
    static constexpr std::size_t left_free_rank = sizeof...(LIndicies) - contract_rank;
    static constexpr std::size_t right_free_rank = sizeof...(RIndicies) - contract_rank;
    static constexpr std::size_t M = pow4(left_free_rank);
    static constexpr std::size_t K = pow4(contract_rank);
    static constexpr std::size_t N = pow4(right_free_rank);
};

template <std::size_t M, std::size_t N, std::size_t K, typename datatype, typename... LIndices,
          typename... RIndices, typename result_tensor_type>
void gemm_backend_0(const Tensor<datatype, LIndices...>& L, const Tensor<datatype, RIndices...>& R,
                    result_tensor_type& C) {
    const datatype* l = L.data.data();
    const datatype* r = R.data.data();
    datatype* __restrict__ c = C.data.data();

    // C already zero
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const datatype a = l[i * K + k];

            for (std::size_t j = 0; j < N; ++j) {
                c[i * N + j] += a * r[k * N + j];
            }
        }
    }
}

template <std::size_t M, std::size_t N, std::size_t K, typename datatype, typename... LIndices,
          typename... RIndices, typename result_tensor_type>
void gemm_backend_1(const Tensor<datatype, LIndices...>& L, const Tensor<datatype, RIndices...>& R,
                    result_tensor_type& C) {

    const datatype* l = L.data.data();
    const datatype* r = R.data.data();
    datatype* __restrict__ c = C.data.data();
    std::array<datatype, K * N> Rt;
    for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t k = 0; k < K; ++k) {
            Rt[k * N + j] = r[j * K + k];
        }
    }
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {

            const datatype a = l[i * K + k];

            for (std::size_t j = 0; j < N; ++j) {
                c[i * N + j] += a * Rt[k * N + j];
            }
        }
    }
}

template <std::size_t M, std::size_t N, std::size_t K, typename datatype, typename... LIndices,
          typename... RIndices, typename result_tensor_type>
void gemm_backend_2(const Tensor<datatype, LIndices...>& L, const Tensor<datatype, RIndices...>& R,
                    result_tensor_type& C) {
    const datatype* l = L.data.data();
    const datatype* r = R.data.data();
    datatype* __restrict__ c = C.data.data();

    // C already zero
    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t i = 0; i < M; ++i) {
            const datatype a = l[k * M + i];

            for (std::size_t j = 0; j < N; ++j) {
                c[i * N + j] += a * r[k * N + j];
            }
        }
    }
}

template <std::size_t M, std::size_t N, std::size_t K, typename datatype, typename... LIndices,
          typename... RIndices, typename result_tensor_type>
void gemm_backend_3(const Tensor<datatype, LIndices...>& L, const Tensor<datatype, RIndices...>& R,
                    result_tensor_type& C) {
    const datatype* l = L.data.data();
    const datatype* r = R.data.data();
    datatype* __restrict__ c = C.data.data();

    std::array<datatype, N * M> tmp{};

    // tmp = R * L
    for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t k = 0; k < K; ++k) {
            const datatype b = r[j * K + k];

            for (std::size_t i = 0; i < M; ++i) {
                tmp[j * M + i] += b * l[k * M + i];
            }
        }
    }

    // C = tmp^T
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            c[i * N + j] = tmp[j * M + i];
        }
    }
}

template <typename datatype, typename... LIndices, typename... RIndices>
auto operator*(const Tensor<datatype, LIndices...>& L, const Tensor<datatype, RIndices...>& R) {
    using analysis = contraction_analysis<type_list<LIndices...>, type_list<RIndices...>>;
    using result_tensor_type =
        typename construct_tensor_from_list<datatype, typename analysis::free_indices>::result;
    result_tensor_type C{};
    C.data = contract_components<typename analysis::plan>(L.data, R.data);
    return C;
}


template <typename datatype, typename... Indices>
auto operator*(datatype scalar, const Tensor<datatype, Indices...>& T) {
    Tensor<datatype, Indices...> C;
    for (std::size_t i = 0; i < C.data.size(); ++i)
        C.data[i] = scalar * T.data[i];
    return C;
}


template <typename datatype, typename... Indices>
auto operator*(const Tensor<datatype, Indices...>& T, datatype scalar) {
    return scalar * T;
}

template <int... N>
struct add_plan {
    static constexpr std::size_t rank = sizeof...(N);
};


template <typename Left, typename Right>
struct analysis_plus_impl;
template <typename... L, typename... R>
struct analysis_plus_impl<type_list<L...>, type_list<R...>> {
    using result = add_plan<find_obj<R, type_list<L...>>::value...>;
};

template <typename datatype, typename Plan>
struct tensorplus_impl {};
template <typename datatype>
struct tensorplus_impl<datatype, add_plan<>> {
    template <typename TensorType, std::size_t rank, std::size_t... Is>
    static decltype(auto) unpk(TensorType&& tensor, const std::array<std::size_t, rank>& index,
                               std::index_sequence<Is...>) {
        return std::forward<TensorType>(tensor)(index[Is]...);
    }
    template <typename... left_idx, typename... right_idx, std::size_t rank>
    static void exec(const Tensor<datatype, left_idx...>& T, const Tensor<datatype, right_idx...>& U,
                     Tensor<datatype, left_idx...>& C, const std::array<std::size_t, rank>& indexleft,
                     const std::array<std::size_t, rank>& indexright) {
        unpk(C, indexleft, std::make_index_sequence<rank>{}) =
            unpk(T, indexleft, std::make_index_sequence<rank>{}) +
            unpk(U, indexright, std::make_index_sequence<rank>{});
    }
};
template <typename datatype, int N, int... Ns>
struct tensorplus_impl<datatype, add_plan<N, Ns...>> {
    template <typename... left_idx, typename... right_idx, std::size_t rank>
    static void exec(const Tensor<datatype, left_idx...>& T, const Tensor<datatype, right_idx...>& U,
                     Tensor<datatype, left_idx...>& C, std::array<std::size_t, rank>& indexleft,
                     std::array<std::size_t, rank>& indexright) {
        constexpr int rightpos = rank - sizeof...(Ns) - 1;
        constexpr int leftpos = N;
        for (std::size_t i = 0; i < 4; ++i) {
            indexleft[leftpos] = i;
            indexright[rightpos] = i;
            tensorplus_impl<datatype, add_plan<Ns...>>::exec(T, U, C, indexleft, indexright);
        }
    }
};

template <typename datatype, typename... left_idx, typename... right_idx>
auto operator+(const Tensor<datatype, left_idx...>& T, const Tensor<datatype, right_idx...>& U) {
    static_assert(is_unorder_same<type_list<left_idx...>, type_list<right_idx...>>::value,
                  "Invalid addition due to index non-compatitance");
    constexpr std::size_t rank = list_size<type_list<left_idx...>>::value;
    using plan = typename analysis_plus_impl<type_list<left_idx...>, type_list<right_idx...>>::result;
    std::array<std::size_t, rank> indexleft{};
    std::array<std::size_t, rank> indexright{};
    Tensor<datatype, left_idx...> C;
    tensorplus_impl<datatype, plan>::exec(T, U, C, indexleft, indexright);
    return C;
}

template <typename datatype, typename... left_idx, typename... right_idx>
auto operator-(const Tensor<datatype, left_idx...>& T, const Tensor<datatype, right_idx...>& U) {
    return T + (-U);
}

template <typename... NewIndices, typename datatype, typename... OldIndices>
auto rename(const Tensor<datatype, OldIndices...>& T) {
    static_assert(sizeof...(NewIndices) == sizeof...(OldIndices), "rename requires same rank");
    static_assert((index_traits<NewIndices>::is_index::value && ...),
                  "rename target indices must all be up<T> or dn<T>");
    static_assert((is_same_variance<OldIndices, NewIndices>::value && ...),
                  "rename may only change index letters, not raise/lower index positions");

    Tensor<datatype, NewIndices...> result;
    result.data = T.data;
    return result;
}


template <typename FreeLetters, typename AllLetters, typename LetterA, typename LetterB>
struct trace_loop;
template <typename AllLetters, typename LetterA, typename LetterB>
struct trace_loop<type_list<>, AllLetters, LetterA, LetterB> {
    template <typename datatype, typename... Indices, typename... ResIdx, std::size_t N>
    static void exec(const Tensor<datatype, Indices...>& T, Tensor<datatype, ResIdx...>& result,
                     std::array<std::size_t, N>& vals) {
        constexpr std::size_t posA = find_obj<LetterA, AllLetters>::value;
        constexpr std::size_t posB = find_obj<LetterB, AllLetters>::value;

        datatype sum{};

        for (std::size_t i = 0; i < 4; ++i) {
            vals[posA] = i;
            vals[posB] = i;

            sum += T(vals[find_obj<typename index_traits<Indices>::letter, AllLetters>::value]...);
        }

        result(vals[find_obj<typename index_traits<ResIdx>::letter, AllLetters>::value]...) = sum;
    }
};
template <typename Head, typename... Tail, typename AllLetters, typename LetterA, typename LetterB>
struct trace_loop<type_list<Head, Tail...>, AllLetters, LetterA, LetterB> {
    template <typename datatype, typename... Indices, typename... ResIdx, std::size_t N>
    static void exec(const Tensor<datatype, Indices...>& T, Tensor<datatype, ResIdx...>& result,
                     std::array<std::size_t, N>& vals) {
        constexpr std::size_t pos = find_obj<Head, AllLetters>::value;

        for (std::size_t i = 0; i < 4; ++i) {
            vals[pos] = i;
            trace_loop<type_list<Tail...>, AllLetters, LetterA, LetterB>::exec(T, result, vals);
        }
    }
};


template <typename LetterA, typename LetterB, typename datatype, typename... Indices>
auto trace(const Tensor<datatype, Indices...>& T) {
    using idx_list = type_list<Indices...>;

    using idxA = typename find_index_from_letter<LetterA, idx_list>::result;
    using idxB = typename find_index_from_letter<LetterB, idx_list>::result;
    static_assert(!is_same_variance<idxA, idxB>::value,
                  "Invalid contraction due to duplicate upper/lower indices");

    using step1 = typename remove_first<idxA, idx_list>::result;
    using result_indices = typename remove_first<idxB, step1>::result;

    using result_tensor = typename construct_tensor_from_list<datatype, result_indices>::result;

    using free_letters = typename extract_letters<result_indices>::result;
    using all_letters = typename concat<free_letters, type_list<LetterA, LetterB>>::result;

    constexpr std::size_t num_all = list_size<all_letters>::value;

    result_tensor result{};
    std::array<std::size_t, num_all> vals{};

    trace_loop<free_letters, all_letters, LetterA, LetterB>::exec(T, result, vals);

    return result;
}
