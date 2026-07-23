#pragma once
#include "Index_and_variance.hpp"
#include "type_lists.hpp"
#include <array>
#include <cassert>
#include <type_traits>

constexpr std::size_t factorial(std::size_t n) {
    std::size_t r = 1;
    for (std::size_t i = 2; i <= n; ++i) {
        r *= i;
    }
    return r;
}

constexpr std::size_t pow4(std::size_t rank) {
    std::size_t r = 1;
    for (std::size_t i = 0; i < rank; ++i) {
        r *= 4;
    }
    return r;
}


// linear_index(i, j, k, ...)
// 把多重指标转换为 data[] 的线性下标。
//
// 示例：rank=3 时
//   linear_index(i,j,k) == ((i * 4) + j) * 4 + k
template <typename... Ints>
constexpr std::size_t linear_index(Ints... idxs) {
    std::array<std::size_t, sizeof...(Ints)> arr{static_cast<std::size_t>(idxs)...};
    std::size_t result = 0;
    for (std::size_t i = 0; i < arr.size(); ++i) {
        result = result * 4 + arr[i];
    }
    return result;
}


// unpack_index<Rank>(lin)
// 把线性下标 lin 还原成 rank 个 0..3 的指标。
//
// 用途：
//   reorder 时枚举新张量的线性下标，再映射回旧张量下标。
template <std::size_t Rank>
std::array<std::size_t, Rank> unpack_index(std::size_t lin) {
    std::array<std::size_t, Rank> idx{};

    for (std::size_t rev = 0; rev < Rank; ++rev) {
        std::size_t d = Rank - 1 - rev;
        idx[d] = lin % 4;
        lin /= 4;
    }

    return idx;
}


// pack_index(idx)
// unpack_index 的逆操作。
template <std::size_t Rank>
std::size_t pack_index(const std::array<std::size_t, Rank>& idx) {
    std::size_t lin = 0;

    for (std::size_t i = 0; i < Rank; ++i) {
        lin = lin * 4 + idx[i];
    }

    return lin;
}

template <typename datatype, typename... Indices>
struct Tensor {
    // 所有指标都必须是 up<T> 或 dn<T>。
    static_assert((index_traits<Indices>::is_index::value && ...), "Tensor indices must all be up<T> or dn<T>");

    // 同一个 Tensor 类型内部不允许重复字母。
    // 例如 Tensor<double, up<mu>, dn<mu>> 不允许。
    // 如果需要缩并，请使用 trace 或张量乘法。
    static_assert(!has_duplicate<type_list<typename index_traits<Indices>::letter...>>::value,
                  "Tensor indices must not repeat");

    static constexpr std::size_t rank = sizeof...(Indices);

    // 一维连续存储，共 4^rank 个分量。
    std::array<datatype, pow4(rank)> data{};

    Tensor() = default;

    // 用同一个初值填满所有分量。
    explicit Tensor(datatype init_value) {
        data.fill(init_value);
    }

    // 非 const 分量访问。
    template <typename... Ints>
    datatype& operator()(Ints... idxs) {
        static_assert(sizeof...(Ints) == rank, "number of indices must match tensor rank");
        static_assert((std::is_integral_v<std::decay_t<Ints>> && ...),
                      "tensor component indices must be integral");
        assert(((static_cast<std::size_t>(idxs) < 4) && ...));
        return data[linear_index(idxs...)];
    }

    // const 分量访问。
    template <typename... Ints>
    const datatype& operator()(Ints... idxs) const {
        static_assert(sizeof...(Ints) == rank, "number of indices must match tensor rank");
        static_assert((std::is_integral_v<std::decay_t<Ints>> && ...),
                      "tensor component indices must be integral");
        assert(((static_cast<std::size_t>(idxs) < 4) && ...));
        return data[linear_index(idxs...)];
    }
};
template <typename datatype, typename... Indices>
auto operator-(const Tensor<datatype, Indices...>& T) {
    Tensor<datatype, Indices...> result;

    for (std::size_t i = 0; i < result.data.size(); ++i) {
        result.data[i] = -T.data[i];
    }

    return result;
}
