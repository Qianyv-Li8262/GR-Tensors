#pragma once
#include "tensor_expression.hpp"
#include <cmath>
#include <type_traits>

inline int metric_calls = 0;

// The only function leaf. The inverse metric is an InverseExpr of this metric.
inline auto metric(const coord<mu>& x) {
    ++metric_calls;
    Tensor<double, dn<a>, dn<b>> g;
    const double r = x(1);
    const double f = 1.0 - 2.0 / r;
    g(0, 0) = -f;
    g(1, 1) = 1.0 / f;
    g(2, 2) = r * r;
    g(3, 3) = r * r * std::sin(x(2)) * std::sin(x(2));
    return g;
}

template <typename A, typename B>
auto g() {
    return rename_field<dn<A>, dn<B>>(make_field<&metric>());
}

template <typename A, typename B>
auto inverse_g() {
    return rename_field<up<A>, up<B>>(inverse_field(make_field<&metric>()));
}

template <typename Rho, typename Mu, typename Nu, DiffScheme Scheme>
auto christoffel_formula() {
    const auto first = partial_field<Mu, Scheme>(g<kappa, Nu>());
    const auto second = partial_field<Nu, Scheme>(g<kappa, Mu>());
    const auto third = partial_field<kappa, Scheme>(g<Mu, Nu>());
    return scale_field<1, 2>(inverse_g<Rho, kappa>() * (first + second - third));
}

template <typename Rho, typename Sigma, typename Mu, typename Nu, DiffScheme Scheme>
auto riemann_formula() {
    const auto q1 = christoffel_formula<Rho, Mu, lambda, Scheme>() *
                    christoffel_formula<lambda, Nu, Sigma, Scheme>();
    const auto q2 = christoffel_formula<Rho, Nu, lambda, Scheme>() *
                    christoffel_formula<lambda, Mu, Sigma, Scheme>();
    const auto d1 = partial_field<Mu, Scheme>(christoffel_formula<Rho, Nu, Sigma, Scheme>());
    const auto d2 = partial_field<Nu, Scheme>(christoffel_formula<Rho, Mu, Sigma, Scheme>());
    // Starting with q1 fixes the storage order to [Rho, Mu, Nu, Sigma].
    return q1 - q2 + d1 - d2;
}

template <DiffScheme Scheme = DiffScheme::Central8>
auto kretschmann_formula() {
    const auto lower = g<a, rho>() * riemann_formula<rho, b, c, d, Scheme>();
    // lower has the storage order [a, c, d, b]; rename dummy letters in that order.
    const auto renamed = rename_field<dn<mu>, dn<kappa>, dn<sigma>, dn<nu>>(lower);
    const auto raise_a = inverse_g<a, mu>() * renamed;
    const auto raise_b = inverse_g<b, nu>() * raise_a;
    const auto raise_c = inverse_g<c, kappa>() * raise_b;
    const auto upper = inverse_g<d, sigma>() * raise_c;
    auto K = lower * upper;
    static_assert(std::is_same_v<typename decltype(K)::result_type, Tensor<double>>);
    return K;
}

