#pragma once
#include "contraction_and_arithmetic.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

enum class DiffScheme { Central2, Central4, Central6, Central8 };

template <typename Letter>
using coord = Tensor<double, up<Letter>>;

// Integer weights with a common denominator; all strategy choices are types.
template <DiffScheme Scheme>
struct derivative_stencil;

template <>
struct derivative_stencil<DiffScheme::Central2> {
    inline static constexpr std::array<int, 2> offsets{-1, 1};
    inline static constexpr std::array<int, 2> weights{-1, 1};
    static constexpr int denominator = 2;
};
template <>
struct derivative_stencil<DiffScheme::Central4> {
    inline static constexpr std::array<int, 4> offsets{-2, -1, 1, 2};
    inline static constexpr std::array<int, 4> weights{1, -8, 8, -1};
    static constexpr int denominator = 12;
};
template <>
struct derivative_stencil<DiffScheme::Central6> {
    inline static constexpr std::array<int, 6> offsets{-3, -2, -1, 1, 2, 3};
    inline static constexpr std::array<int, 6> weights{-1, 9, -45, 45, -9, 1};
    static constexpr int denominator = 60;
};
template <>
struct derivative_stencil<DiffScheme::Central8> {
    inline static constexpr std::array<int, 8> offsets{-4, -3, -2, -1, 1, 2, 3, 4};
    inline static constexpr std::array<int, 8> weights{3, -32, 168, -672, 672, -168, 32, -3};
    static constexpr int denominator = 840;
};

namespace derivative_detail {

template <typename Coordinate>
void check_step(const Coordinate& x, double h) {
    assert(std::isfinite(h) && [&] {
        double scale = 1.0;
        for (double component : x.data) {
            scale = std::max(scale, std::abs(component));
        }
        return std::abs(h) > 32.0 * std::numeric_limits<double>::epsilon() * scale;
    }());
    (void)x;
    (void)h;
}

} // namespace derivative_detail

// Opaque callable fields retain direct evaluation, with compile-time strategy selection.
// Result metadata is obtained in an unevaluated context: no extra center sample.
namespace derivative_detail {

template <typename DerivLetter, DiffScheme Scheme, typename F, typename CoordLetter,
          typename Scalar, typename... Indices, std::size_t... I>
auto evaluate_callable(const F& field, const coord<CoordLetter>& x, double h,
                       const Tensor<Scalar, Indices...>*, std::index_sequence<I...>) {
    using result_type = Tensor<Scalar, dn<DerivLetter>, Indices...>;
    using stencil = derivative_stencil<Scheme>;
    constexpr std::size_t size = pow4(sizeof...(Indices));
    result_type result;
    for (std::size_t axis = 0; axis < 4; ++axis) {
        auto accumulate = [&](auto position) {
            constexpr std::size_t i = decltype(position)::value;
            auto sample = x;
            sample(axis) += stencil::offsets[i] * h;
            const auto values = field.eval(sample);
            for (std::size_t j = 0; j < size; ++j) {
                constexpr double weight = static_cast<double>(stencil::weights[i]) / stencil::denominator;
                result.data[axis * size + j] += weight * values.data[j];
            }
        };
        (accumulate(std::integral_constant<std::size_t, I>{}), ...);
        for (std::size_t j = 0; j < size; ++j) {
            result.data[axis * size + j] /= h;
        }
    }
    return result;
}

} // namespace derivative_detail

template <typename DerivLetter, DiffScheme Scheme = DiffScheme::Central8, typename CoordLetter, typename F>
auto partial(const F& field, const coord<CoordLetter>& x, double h = 1e-2) {
    derivative_detail::check_step(x, h);
    using source_type = std::decay_t<decltype(field.eval(x))>;
    return derivative_detail::evaluate_callable<DerivLetter, Scheme>(
        field, x, h, static_cast<const source_type*>(nullptr),
        std::make_index_sequence<derivative_stencil<Scheme>::offsets.size()>{});
}

template <typename UpLetter, typename DummyLetter, DiffScheme Scheme = DiffScheme::Central8,
          typename CoordLetter, typename F, typename InvMetricFunc>
auto partial_up(const F& field, const InvMetricFunc& inv_metric, const coord<CoordLetter>& x, double h = 1e-2) {
    auto lower = partial<DummyLetter, Scheme>(field, x, h);
    auto ginv = rename<up<UpLetter>, up<DummyLetter>>(inv_metric.eval(x));
    return ginv * lower;
}
