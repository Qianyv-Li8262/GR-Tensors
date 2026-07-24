#pragma once
#include "contraction_and_arithmetic.hpp"
#include "tensor_class.hpp"
#include "num_deriv.hpp"
#include <cstdlib>
#include <type_traits>
#include <utility>
template <typename F>
struct Field {
    F func;

    Field(F f) : func(std::move(f)) {
    }

    // 让 Field 可以像函数一样调用
    template <typename... Args>
    auto operator()(Args&&... args) const {
        return func(std::forward<Args>(args)...);
    }
};
template <typename F>
Field(F) -> Field<F>;


// make_field(f)
// 辅助函数：把 lambda 或函数对象包装成 Field。
//
// 示例：
//   auto g = make_field([](const X& x) { return metric<mu,nu>(x); });
template <typename F>
auto make_field(F&& f) {
    return Field<std::decay_t<F>>{std::forward<F>(f)};
}

// ==================== Field 运算符重载 ====================

// Field + Field
template <typename F1, typename F2>
auto operator+(const Field<F1>& a, const Field<F2>& b) {
    return make_field([a, b](const auto& x) {
        return a(x) + b(x);
    });
}

// Field - Field
template <typename F1, typename F2>
auto operator-(const Field<F1>& a, const Field<F2>& b) {
    return make_field([a, b](const auto& x) {
        return a(x) - b(x);
    });
}

// Field * Field（张量缩并）
template <typename F1, typename F2>
auto operator*(const Field<F1>& a, const Field<F2>& b) {
    return make_field([a, b](const auto& x) {
        return a(x) * b(x);
    });
}

// scalar * Field
template <typename F>
auto operator*(double scalar, const Field<F>& a) {
    return make_field([scalar, a](const auto& x) {
        return scalar * a(x);
    });
}

// Field * scalar
template <typename F>
auto operator*(const Field<F>& a, double scalar) {
    return scalar * a;
}

// ==================== Field 版工具函数 ====================

template <typename... NewIndices, typename F>
auto rename_field(const Field<F>& field) {
    return make_field([field](const auto& x) {
        return rename<NewIndices...>(field(x));
    });
}

// partial 返回 Field
template <typename DerivLetter, typename F>
auto partial_field(const Field<F>& field, double h = 1e-2) {
    return make_field([field, h](const auto& x) {
        return partial<DerivLetter>(field, x, h);
    });
}

// trace 的 Field 版
template <typename LetterA, typename LetterB, typename F>
auto trace_field(const Field<F>& field) {
    return make_field([field](const auto& x) {
        return trace<LetterA, LetterB>(field(x));
    });
}

// Define a two-index tensor field variable template.
//
// Usage:
//   DEFINE_TENSOR_FIELD2(g, metric);
//   auto gx = g<mu, nu>(x);
//   auto dg = partial_field<rho>(g<mu, nu>);
#define DEFINE_TENSOR_FIELD2(field_name, tensor_function)                                                      \
    template <typename FirstIndex, typename SecondIndex>                                                       \
    inline auto field_name = make_field([](const auto& x) {                                                    \
        return tensor_function<FirstIndex, SecondIndex>(x);                                                    \
    })
