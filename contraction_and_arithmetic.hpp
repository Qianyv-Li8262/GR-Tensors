#pragma once
#include "Index_and_variance.hpp"
#include "tensor_class.hpp"
#include "type_lists.hpp"
#include <array>
#include <cstdlib>
#include <type_traits>
#include <utility>
template <typename contract_letters, typename all_letters>
struct contract_loop;
template <typename all_letters>
struct contract_loop<type_list<>, all_letters> {
    template <typename datatype, typename... left_idx, typename... right_idx, std::size_t N>
    static void exec(const Tensor<datatype, left_idx...>& L, const Tensor<datatype, right_idx...>& R,
                     std::array<std::size_t, N>& vals, datatype& sum) {
        sum += L(vals[find_obj<typename index_traits<left_idx>::letter, all_letters>::value]...) *
               R(vals[find_obj<typename index_traits<right_idx>::letter, all_letters>::value]...);
    }
};
template <typename Head, typename... Tail, typename all_letters>
struct contract_loop<type_list<Head, Tail...>, all_letters> {
    template <typename datatype, typename... left_idx, typename... right_idx, std::size_t N>
    static void exec(const Tensor<datatype, left_idx...>& L, const Tensor<datatype, right_idx...>& R,
                     std::array<std::size_t, N>& vals, datatype& sum) {
        constexpr std::size_t pos = find_obj<Head, all_letters>::value;
        for (std::size_t i = 0; i < 4; ++i) {
            vals[pos] = i;
            contract_loop<type_list<Tail...>, all_letters>::exec(L, R, vals, sum);
        }
    }
};


template <typename FreeLetters, typename all_letters, typename contract_letters>
struct free_loop;
template <typename all_letters, typename contract_letters>
struct free_loop<type_list<>, all_letters, contract_letters> {
    template <typename datatype, typename... left_idx, typename... right_idx, typename... res_idx, std::size_t N>
    static void exec(const Tensor<datatype, left_idx...>& L, const Tensor<datatype, right_idx...>& R,
                     Tensor<datatype, res_idx...>& C, std::array<std::size_t, N>& vals) {
        datatype sum = 0;
        contract_loop<contract_letters, all_letters>::exec(L, R, vals, sum);
        C(vals[find_obj<typename index_traits<res_idx>::letter, all_letters>::value]...) = sum;
    }
};
template <typename Head, typename... Tail, typename all_letters, typename contract_letters>
struct free_loop<type_list<Head, Tail...>, all_letters, contract_letters> {
    template <typename datatype, typename... left_idx, typename... right_idx, typename... res_idx, std::size_t N>
    static void exec(const Tensor<datatype, left_idx...>& L, const Tensor<datatype, right_idx...>& R,
                     Tensor<datatype, res_idx...>& C, std::array<std::size_t, N>& vals) {
        constexpr std::size_t pos = find_obj<Head, all_letters>::value;
        for (std::size_t i = 0; i < 4; ++i) {
            vals[pos] = i;
            free_loop<type_list<Tail...>, all_letters, contract_letters>::exec(L, R, C, vals);
        }
    }
};

template <typename T1, typename T2>
struct find_contract_letter {};
template <typename obj>
struct find_contract_letter<obj, type_list<>> {
    static constexpr int value = 1;
}; // letter not found, free letter
template <typename obj, typename head, typename... Ts1>
struct find_contract_letter<obj, type_list<head, Ts1...>> {
    static constexpr int value =
        std::is_same_v<typename index_traits<obj>::letter, typename index_traits<head>::letter>
            ? (index_traits<obj>::variance::value == index_traits<head>::variance::value ? 0 : 2)
            : find_contract_letter<obj, type_list<Ts1...>>::value;
    static_assert(value != 0, "Invalid contraction due to same variance contraction.");
};


template <typename list1, typename result1, typename result2>
struct make_contraction_letters {};
template <typename list1, typename list2, typename list3, int code>
struct make_contraction_letters_impl1 {};
template <typename... Ts1, typename... Ts2>
struct make_contraction_letters<type_list<>, type_list<Ts1...>, type_list<Ts2...>> {
    using free = type_list<Ts1...>;
    using contract = type_list<Ts2...>;
};
template <typename head, typename... Ts, typename... Ts1, typename... Ts2>
struct make_contraction_letters_impl1<type_list<head, Ts...>, type_list<Ts1...>, type_list<Ts2...>, 1> {
    using impl = make_contraction_letters<type_list<Ts...>, type_list<Ts1..., head>, type_list<Ts2...>>;
    using free = typename impl::free;
    using contract = typename impl::contract;
};
template <typename head, typename... Ts, typename... Ts1, typename... Ts2>
struct make_contraction_letters_impl1<type_list<head, Ts...>, type_list<Ts1...>, type_list<Ts2...>, 2> {
    using new_list = typename remove_all<typename opposite<head>::result, type_list<Ts...>>::result;
    using impl = make_contraction_letters<new_list, type_list<Ts1...>,
                                          type_list<Ts2..., typename index_traits<head>::letter>>;
    using free = typename impl::free;
    using contract = typename impl::contract;
};
template <typename head, typename... Ts, typename... Ts1, typename... Ts2>
struct make_contraction_letters<type_list<head, Ts...>, type_list<Ts1...>, type_list<Ts2...>> {
    static constexpr int cntr_res = find_contract_letter<head, type_list<Ts...>>::value;
    using impl =
        make_contraction_letters_impl1<type_list<head, Ts...>, type_list<Ts1...>, type_list<Ts2...>, cntr_res>;
    using free = typename impl::free;
    using contract = typename impl::contract;
};

template <typename datatype, typename List>
struct construct_tensor_from_list;
template <typename datatype, typename... Indices>
struct construct_tensor_from_list<datatype, type_list<Indices...>> {
    using result = Tensor<datatype, Indices...>;
};

template <typename List>
struct extract_letters;
template <typename... Indices>
struct extract_letters<type_list<Indices...>> {
    using result = type_list<typename index_traits<Indices>::letter...>;
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
    // L = [M][K]
    // R = [K][N]
    // C = [M][N]
    // C initially zero
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const datatype a = L.data[i * K + k];
            for (std::size_t j = 0; j < N; ++j) {
                C.data[i * N + j] += a * R.data[k * N + j];
            }
        }
    }
}

template <std::size_t M, std::size_t N, std::size_t K, typename datatype, typename... LIndices,
          typename... RIndices, typename result_tensor_type>
void gemm_backend_1(const Tensor<datatype, LIndices...>& L, const Tensor<datatype, RIndices...>& R,
                    result_tensor_type& C) {
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            datatype sum = 0;
            for (std::size_t k = 0; k < K; ++k) {
                sum += L.data[i * K + k] * R.data[j * K + k];
            }
            C.data[i * N + j] = sum;
        }
    }
}

template <std::size_t M, std::size_t N, std::size_t K, typename datatype, typename... LIndices,
          typename... RIndices, typename result_tensor_type>
void gemm_backend_2(const Tensor<datatype, LIndices...>& L, const Tensor<datatype, RIndices...>& R,
                    result_tensor_type& C) {
    // C initially zero
    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t i = 0; i < M; ++i) {
            const datatype a = L.data[k * M + i];
            for (std::size_t j = 0; j < N; ++j) {
                C.data[i * N + j] += a * R.data[k * N + j];
            }
        }
    }
}

template <std::size_t M, std::size_t N, std::size_t K, typename datatype, typename... LIndices,
          typename... RIndices, typename result_tensor_type>
void gemm_backend_3(const Tensor<datatype, LIndices...>& L, const Tensor<datatype, RIndices...>& R,
                    result_tensor_type& C) {
    // L physical: K x M
    // R physical: N x K
    //
    // C = L^T * R^T
    //   = (R * L)^T
    //
    // tmp = R * L : N x M
    std::array<datatype, N * M> tmp{};
    // 普通 NN: (N x K) * (K x M)
    for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t k = 0; k < K; ++k) {
            const datatype b = R.data[j * K + k];
            for (std::size_t i = 0; i < M; ++i) {
                tmp[j * M + i] += b * L.data[k * M + i];
            }
        }
    }
    // C = tmp^T
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            C.data[i * N + j] = tmp[j * M + i];
        }
    }
}

template <typename datatype, typename... LIndices, typename... RIndices>
auto operator*(const Tensor<datatype, LIndices...>& L, const Tensor<datatype, RIndices...>& R) {
    using result_impl = make_contraction_letters<type_list<LIndices..., RIndices...>, type_list<>, type_list<>>;
    using free_indices = typename result_impl::free;
    using contract_letters = typename result_impl::contract;
    using result_tensor_type = typename construct_tensor_from_list<datatype, free_indices>::result;
    using gemm_info = is_gemm<type_list<LIndices...>, type_list<RIndices...>, contract_letters>;
    result_tensor_type C{};
    if constexpr (gemm_info::value) {
        if constexpr (!gemm_info::trans_A && !gemm_info::trans_B) {
            gemm_backend_0<gemm_info::M, gemm_info::N, gemm_info::K>(L, R, C);
        } else if constexpr (!gemm_info::trans_A && gemm_info::trans_B) {
            gemm_backend_1<gemm_info::M, gemm_info::N, gemm_info::K>(L, R, C);
        } else if constexpr (gemm_info::trans_A && !gemm_info::trans_B) {
            gemm_backend_2<gemm_info::M, gemm_info::N, gemm_info::K>(L, R, C);
        } else {
            gemm_backend_3<gemm_info::M, gemm_info::N, gemm_info::K>(L, R, C);
        }
    } else {
        // 原来的通用 contraction fallback
        using free_letters = typename extract_letters<free_indices>::result;
        using all_letters = typename concat<free_letters, contract_letters>::result;
        constexpr std::size_t num_all = list_size<all_letters>::value;
        std::array<std::size_t, num_all> vals{};
        free_loop<free_letters, all_letters, contract_letters>::exec(L, R, C, vals);
    }
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

template <typename datatype, typename listleft, typename listright>
struct tensorplus_impl {};
template <typename datatype, typename... listleft>
struct tensorplus_impl<datatype, type_list<listleft...>, type_list<>> {
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
template <typename datatype, typename... listleft, typename righthead, typename... listright>
struct tensorplus_impl<datatype, type_list<listleft...>, type_list<righthead, listright...>> {
    template <typename... left_idx, typename... right_idx, std::size_t rank>
    static void exec(const Tensor<datatype, left_idx...>& T, const Tensor<datatype, right_idx...>& U,
                     Tensor<datatype, left_idx...>& C, std::array<std::size_t, rank>& indexleft,
                     std::array<std::size_t, rank>& indexright) {
        constexpr int leftpos = find_obj<righthead, type_list<left_idx...>>::value;
        constexpr int rightpos = find_obj<righthead, type_list<right_idx...>>::value;
        for (std::size_t i = 0; i < 4; ++i) {
            indexleft[leftpos] = i;
            indexright[rightpos] = i;
            tensorplus_impl<datatype, type_list<listleft...>, type_list<listright...>>::exec(T, U, C, indexleft,
                                                                                             indexright);
        }
    }
};

template <typename datatype, typename... left_idx, typename... right_idx>
auto operator+(const Tensor<datatype, left_idx...>& T, const Tensor<datatype, right_idx...>& U) {
    static_assert(is_unorder_same<type_list<left_idx...>, type_list<right_idx...>>::value,
                  "Invalid addition due to index non-compatitance");
    constexpr std::size_t rank = list_size<type_list<left_idx...>>::value;
    std::array<std::size_t, rank> indexleft{};
    std::array<std::size_t, rank> indexright{};
    Tensor<datatype, left_idx...> C;
    tensorplus_impl<datatype, type_list<left_idx...>, type_list<right_idx...>>::exec(T, U, C, indexleft,
                                                                                     indexright);
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
// 鑷敱鎸囨爣閫掑綊锛氭瘡鍓ヤ竴涓嚜鐢卞瓧姣嶏紝鐢熸垚涓€灞?for
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
