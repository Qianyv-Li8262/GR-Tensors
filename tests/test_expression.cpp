#include "tensor_expression.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <type_traits>

namespace {

int metric_calls = 0;
int inverse_calls = 0;
int failures = 0;

// Dense, coordinate-dependent Lorentz metric: a nondiagonal test catches axis mistakes.
auto metric(const Tensor<double, up<mu>>& x) {
    ++metric_calls;
    Tensor<double, dn<a>, dn<b>> g;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const double u_i = 0.1 * (i + 1);
            const double u_j = 0.1 * (j + 1);
            g(i, j) = (i == j ? (i == 0 ? -2.0 - x(1) : 2.0 + i) : 0.0) + u_i * u_j;
        }
    }
    return g;
}

// Independent Sherman-Morrison inverse of D + u u^T.
auto inverse_metric(const Tensor<double, up<mu>>& x) noexcept {
    ++inverse_calls;
    Tensor<double, up<a>, up<b>> inverse;
    const double diagonal[] = {-2.0 - x(1), 3.0, 4.0, 5.0};
    double denominator = 1.0;
    for (int i = 0; i < 4; ++i) {
        const double u = 0.1 * (i + 1);
        denominator += u * u / diagonal[i];
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            inverse(i, j) = (i == j ? 1.0 / diagonal[i] : 0.0) -
                            (0.1 * (i + 1) / diagonal[i]) * (0.1 * (j + 1) / diagonal[j]) / denominator;
        }
    }
    return inverse;
}

auto scalar_field(const Tensor<double, up<mu>>& x) {
    return Tensor<double>(2.0 + x(1));
}

auto other_scalar_field(const Tensor<double, up<mu>>& x) {
    return Tensor<double>(3.0 + x(1));
}

static_assert(std::is_same_v<decltype(&scalar_field), decltype(&other_scalar_field)>);
static_assert(!std::is_same_v<FieldExpr<&scalar_field>, FieldExpr<&other_scalar_field>>);

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        ++failures;
    }
}

bool close(double actual, double expected) {
    return std::isfinite(actual) && std::abs(actual - expected) < 1e-11 * std::max(1.0, std::abs(expected));
}

} // namespace

int main() {
    const auto g = make_field<&metric>();
    const auto inv = make_field<&inverse_metric>();
    const auto inv_bc = rename_field<up<b>, up<c>>(inv);
    const auto g_cd = rename_field<dn<c>, dn<d>>(g);
    const auto inv_dmu = rename_field<up<d>, up<mu>>(inv);
    const auto first = g * inv_bc;
    const auto triple = first * g_cd;
    const auto chain = triple * inv_dmu;
    const auto second = g_cd * inv_dmu;
    const auto balanced = first * second;
    const auto scalar = make_field<&scalar_field>();
    const auto scaled = scalar * g;
    const auto scaled_right = g * scalar;
    const auto squared_scalar = scalar * scalar;
    const auto outer = g * g_cd;
    const auto inverse_outer = inv * rename_field<up<c>, up<d>>(inv);
    const auto fully_contracted = outer * inverse_outer;

    using G = FieldExpr<&metric>;
    using I = FieldExpr<&inverse_metric>;
    using P = contraction_plan<2, 2, contractpair<1, 0>>;
    using GI = ContractExpr<G, I, P>;
    static_assert(std::is_same_v<typename decltype(first)::node_type, GI>);
    static_assert(std::is_same_v<typename decltype(second)::node_type, GI>);
    static_assert(std::is_same_v<typename decltype(g_cd)::node_type, G>);
    static_assert(std::is_same_v<typename decltype(chain)::node_type, ContractExpr<ContractExpr<GI, G, P>, I, P>>);
    static_assert(std::is_same_v<typename decltype(balanced)::node_type, ContractExpr<GI, GI, P>>);
    static_assert(std::is_same_v<typename decltype(chain)::result_type, Tensor<double, dn<a>, up<mu>>>);
    static_assert(std::is_same_v<typename decltype(triple)::result_type, Tensor<double, dn<a>, dn<d>>>);
    static_assert(std::is_same_v<typename decltype(outer)::node_type::Plan, contraction_plan<2, 2>>);
    static_assert(std::is_same_v<typename decltype(fully_contracted)::node_type::Plan,
                                 contraction_plan<4, 4, contractpair<0, 0>, contractpair<1, 1>, contractpair<2, 2>,
                                                  contractpair<3, 3>>>);
    static_assert(std::is_same_v<typename decltype(fully_contracted)::result_type, Tensor<double>>);
    static_assert(std::is_same_v<typename decltype(scaled)::node_type::Plan, contraction_plan<0, 2>>);
    static_assert(std::is_same_v<typename decltype(scaled_right)::node_type::Plan, contraction_plan<2, 0>>);
    check(metric_calls == 0 && inverse_calls == 0, "constructing the tree must not evaluate fields");

    for (double radius : {0.5, 3.0, 10.0}) {
        Tensor<double, up<mu>> x;
        x(1) = radius;
        const auto result = chain.eval(x);
        const auto balanced_result = balanced(x);
        const auto triple_result = triple.eval(x);
        const auto metric_value = metric(x);
        const auto inverse_value = inverse_metric(x);
        const auto outer_result = outer.eval(x);
        const auto scaled_result = scaled.eval(x);
        const auto scaled_right_result = scaled_right.eval(x);
        check(close(squared_scalar.eval(x)(), (2.0 + radius) * (2.0 + radius)), "scalar * scalar mismatch");
        check(close(fully_contracted.eval(x)(), 16.0), "rank-four full contraction mismatch");
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                check(close(result(i, j), i == j ? 1.0 : 0.0), "metric chain must equal identity");
                check(close(balanced_result(i, j), result(i, j)), "balanced tree mismatch");
                check(close(triple_result(i, j), metric_value(i, j)), "g * inverse * g must equal g");
                check(close(scaled_result(i, j), (2.0 + radius) * metric_value(i, j)), "scalar * tensor mismatch");
                check(close(scaled_right_result(i, j), scaled_result(i, j)), "tensor * scalar mismatch");
                double independent = 0.0;
                for (int k = 0; k < 4; ++k) {
                    for (int l = 0; l < 4; ++l) {
                        for (int m = 0; m < 4; ++m) {
                            independent += metric_value(i, k) * inverse_value(k, l) * metric_value(l, m) *
                                           inverse_value(m, j);
                        }
                        check(close(outer_result(i, j, k, l), metric_value(i, j) * metric_value(k, l)),
                              "outer product mismatch");
                    }
                }
                check(close(result(i, j), independent), "chain differs from independent component loops");
            }
        }
        metric_calls = inverse_calls = 0;
        (void)balanced.eval(x);
        check(metric_calls == 2 && inverse_calls == 2, "tree evaluation must visit both repeated subtrees");
    }
    if (failures) {
        return 1;
    }
    std::cout
        << "expression types, lazy construction, dense metric chains, outer/scalar/full contractions passed\n";
}
