#include "tensors_all.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <type_traits>

// Coordinates (t, r, theta, phi), Schwarzschild mass M = 1.
auto metric(const coord<mu>& x) {
    Tensor<double, dn<a>, dn<b>> g;
    const double f = 1.0 - 2.0 / x(1);
    g(0, 0) = -f;
    g(1, 1) = 1.0 / f;
    g(2, 2) = x(1) * x(1);
    g(3, 3) = x(1) * x(1) * std::sin(x(2)) * std::sin(x(2));
    return g;
}

auto inverse_metric(const coord<mu>& x) {
    const auto g = metric(x);
    Tensor<double, up<a>, up<b>> inverse;
    for (int i = 0; i < 4; ++i) {
        inverse(i, i) = 1.0 / g(i, i);
    }
    return inverse;
}

template <typename T>
void print_type(const char* label) {
    std::cout << label << '\n';
#if defined(__clang__) || defined(__GNUC__)
    std::cout << __PRETTY_FUNCTION__ << "\n\n";
#elif defined(_MSC_VER)
    std::cout << __FUNCSIG__ << "\n\n";
#else
#error This example needs a compiler with a function signature string extension.
#endif
}

int main() {
    const auto g = make_field<&metric>();
    const auto inverse = make_field<&inverse_metric>();
    const auto g_ab = g;
    const auto inverse_bc = rename_field<up<b>, up<c>>(inverse);
    const auto g_cd = rename_field<dn<c>, dn<d>>(g);
    const auto inverse_dmu = rename_field<up<d>, up<mu>>(inverse);

    // g_ab g^bc g_cd g^dmu = delta_a^mu. Multiplication is left associative.
    const auto chain = g_ab * inverse_bc * g_cd * inverse_dmu;
    const auto balanced = (g_ab * inverse_bc) * (g_cd * inverse_dmu);
    using Plan = contraction_plan<2, 2, contractpair<1, 0>>;
    using G = FieldExpr<&metric>;
    using I = FieldExpr<&inverse_metric>;
    using GI = ContractExpr<G, I, Plan>;
    using Chain = ContractExpr<ContractExpr<GI, G, Plan>, I, Plan>;
    static_assert(std::is_same_v<typename decltype(chain)::node_type, Chain>);
    static_assert(std::is_same_v<typename decltype(chain)::result_type, Tensor<double, dn<a>, up<mu>>>);
    static_assert(std::is_same_v<typename decltype(balanced)::node_type, ContractExpr<GI, GI, Plan>>);

    coord<mu> x;
    x(1) = 10.0;
    x(2) = 1.1;
    const auto value = chain.eval(x);
    const auto balanced_value = balanced.eval(x);
    double max_error = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const double expected = i == j ? 1.0 : 0.0;
            max_error = std::max(max_error, std::abs(value(i, j) - expected));
            max_error = std::max(max_error, std::abs(balanced_value(i, j) - expected));
        }
    }
    std::cout << "g_ab g^bc g_cd g^dmu = delta_a^mu; max error = " << max_error << "\n\n";
    print_type<typename decltype(chain)::result_type>("Result tensor type:");
    print_type<typename decltype(chain)::node_type>("Left-associated computation tree:");
    print_type<std::decay_t<decltype(chain)>>("Complete indexed expression type:");
    print_type<typename decltype(balanced)::node_type>("Balanced computation tree (same GI subtree twice):");
    return max_error < 1e-12 ? 0 : 1;
}
