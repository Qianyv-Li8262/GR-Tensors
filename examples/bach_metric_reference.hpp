#pragma once
#include "tensor_expression.hpp"
#include <cmath>
#include <cstdint>

// Only the test metric is configurable. The expression and its difference scheme
// stay fixed while each evaluation starts with a fresh set of parameters.
struct MetricParameters {
    double first = 0.6;
    double second = 1.0;
    double conformal = 0;
};
inline MetricParameters parameters;
inline std::uint64_t metric_calls = 0;

inline double conformal_factor(const coord<mu>& x) {
    return parameters.conformal *
           (0.13 * x(0) - 0.17 * x(1) + 0.11 * x(2) + 0.07 * x(3) + 0.05 * x(0) * x(2));
}

// g0 = dx0^2 + exp(2 A x0) dx1^2 + dx2^2 + exp(2 D x2) dx3^2.
// Its two surface curvatures are k1 = -A^2 and k2 = -D^2.
// g = exp(2 phi) g0 also tests nonparallel Weyl curvature in all four coordinates.
inline auto metric(const coord<mu>& x) {
    ++metric_calls;
    Tensor<double, dn<a>, dn<b>> result;
    const double omega2 = std::exp(2 * conformal_factor(x));
    result(0, 0) = result(2, 2) = omega2;
    result(1, 1) = omega2 * std::exp(2 * parameters.first * x(0));
    result(3, 3) = omega2 * std::exp(2 * parameters.second * x(2));
    return result;
}

// An independent analytic reference, without differences or expression helpers.
// Parallel Weyl curvature of the product gives B_ij = (k1^2-k2^2)/6 g0_ij
// on the first surface and the negative of this on the second surface.
// In dimension four, B[exp(2 phi) g0]_ij = exp(-2 phi) B[g0]_ij.
inline auto exact_bach(const coord<mu>& x) {
    Tensor<double, dn<a>, dn<b>> result;
    const double a2 = parameters.first * parameters.first;
    const double b2 = parameters.second * parameters.second;
    const double coefficient = (a2 * a2 - b2 * b2) / 6 * std::exp(-2 * conformal_factor(x));
    result(0, 0) = coefficient;
    result(1, 1) = coefficient * std::exp(2 * parameters.first * x(0));
    result(2, 2) = -coefficient;
    result(3, 3) = -coefficient * std::exp(2 * parameters.second * x(2));
    return result;
}

inline double exact_weyl(const coord<mu>& x, int a, int b, int c, int d) {
    const double diagonal[]{1, std::exp(2 * parameters.first * x(0)),
                            1, std::exp(2 * parameters.second * x(2))};
    const double k[]{-parameters.first * parameters.first, -parameters.second * parameters.second};
    auto g = [&](int i, int j) { return i == j ? diagonal[i] : 0.0; };
    auto ric = [&](int i, int j) { return k[i / 2] * g(i, j); };
    const double riemann = a / 2 == b / 2 && a / 2 == c / 2 && a / 2 == d / 2
                               ? k[a / 2] * (g(a, c) * g(b, d) - g(a, d) * g(b, c)) : 0;
    const double weyl = riemann - (g(a, c) * ric(b, d) - g(a, d) * ric(b, c) -
                                   g(b, c) * ric(a, d) + g(b, d) * ric(a, c)) / 2 +
                        (k[0] + k[1]) / 3 * (g(a, c) * g(b, d) - g(a, d) * g(b, c));
    return std::exp(2 * conformal_factor(x)) * weyl;
}

