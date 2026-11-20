#pragma once
#include "tensor_class.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>

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
