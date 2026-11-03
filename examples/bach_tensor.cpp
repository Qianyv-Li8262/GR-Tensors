#include "tensor_expression.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>

// Only the test metric is configurable. The expression and its difference scheme
// stay fixed while each evaluation starts with a fresh set of parameters.
struct MetricParameters {
    double first = 0.6;
    double second = 1.0;
    double conformal = 0;
};
MetricParameters parameters;
std::uint64_t metric_calls = 0;

double conformal_factor(const coord<mu>& x) {
    return parameters.conformal *
           (0.13 * x(0) - 0.17 * x(1) + 0.11 * x(2) + 0.07 * x(3) + 0.05 * x(0) * x(2));
}

// g0 = dx0^2 + exp(2 A x0) dx1^2 + dx2^2 + exp(2 D x2) dx3^2.
// Its two surface curvatures are k1 = -A^2 and k2 = -D^2.
// g = exp(2 phi) g0 also tests nonparallel Weyl curvature in all four coordinates.
auto metric(const coord<mu>& x) {
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
auto exact_bach(const coord<mu>& x) {
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

double exact_weyl(const coord<mu>& x, int a, int b, int c, int d) {
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

// Count the mathematical tree without ever lowering a partial to stencil nodes.
template <typename Node> struct Stats;
template <typename List> struct ChildStats;
template <typename... Nodes> struct ChildStats<type_list<Nodes...>> {
    static constexpr std::size_t nodes = (Stats<Nodes>::nodes + ... + 0);
    static constexpr std::size_t partials = (Stats<Nodes>::partials + ... + 0);
    static constexpr std::size_t depth = [] {
        std::size_t maximum = 0;
        ((maximum = std::max(maximum, Stats<Nodes>::depth)), ...);
        return maximum;
    }();
};
template <typename Node> struct IsPartial : std::false_type {};
template <typename Node, DiffScheme S, typename O>
struct IsPartial<PartialExpr<Node, S, O>> : std::true_type {};
template <typename Node> struct Stats {
    using child = ChildStats<typename Node::children>;
    static constexpr std::size_t nodes = 1 + child::nodes;
    static constexpr std::size_t partials = IsPartial<Node>::value + child::partials;
    static constexpr std::size_t depth = IsPartial<Node>::value + child::depth;
};

int main(int argc, char** argv) {
#if defined(BACH_CENTRAL4)
    constexpr auto scheme = DiffScheme::Central4;
    constexpr int accuracy = 4;
#else
    constexpr auto scheme = DiffScheme::Central2;
    constexpr int accuracy = 2;
#endif
    // All mathematical expressions are constructed here, from one metric leaf.
    auto g = make_field<&metric>();
    auto inverse = inverse_field(g);
    auto pg = partial_field<mu, scheme>(g);
    auto Gamma = scale_field<1, 2>(
        rename_field<up<a>, up<kappa>>(inverse) *
        (rename_field<dn<b>, dn<kappa>, dn<c>>(pg) +
         rename_field<dn<c>, dn<kappa>, dn<b>>(pg) -
         rename_field<dn<kappa>, dn<b>, dn<c>>(pg)));
    auto pGamma = partial_field<mu, scheme>(Gamma);

    // R^rho_{b c d}; the storage order is [rho, c, b, d].
    auto q1 = rename_field<up<rho>, dn<c>, dn<lambda>>(Gamma) *
              rename_field<up<lambda>, dn<b>, dn<d>>(Gamma);
    auto q2 = rename_field<up<rho>, dn<d>, dn<lambda>>(Gamma) *
              rename_field<up<lambda>, dn<b>, dn<c>>(Gamma);
    auto R = q1 - q2 + rename_field<dn<c>, up<rho>, dn<b>, dn<d>>(pGamma) -
                       rename_field<dn<d>, up<rho>, dn<b>, dn<c>>(pGamma);
    auto Ricci = trace_field<rho, c>(R); // [b, d], Ric_bd = R^rho_{b rho d}
    auto scalar = rename_field<up<b>, up<d>>(inverse) * Ricci;
    auto lower = rename_field<dn<a>, dn<rho>>(g) * R; // R_abcd, storage [a,c,b,d]

    auto ricci_terms =
        rename_field<dn<a>, dn<c>>(g) * rename_field<dn<b>, dn<d>>(Ricci) -
        rename_field<dn<a>, dn<d>>(g) * rename_field<dn<b>, dn<c>>(Ricci) -
        rename_field<dn<b>, dn<c>>(g) * rename_field<dn<a>, dn<d>>(Ricci) +
        rename_field<dn<b>, dn<d>>(g) * rename_field<dn<a>, dn<c>>(Ricci);
    auto metric_terms = rename_field<dn<a>, dn<c>>(g) * rename_field<dn<b>, dn<d>>(g) -
                        rename_field<dn<a>, dn<d>>(g) * rename_field<dn<b>, dn<c>>(g);
    auto Weyl = lower - scale_field<1, 2>(ricci_terms) + scale_field<1, 6>(scalar) * metric_terms;

    // T_abcd := C_acbd. This view follows the physical storage order of Weyl.
    auto T = rename_field<dn<a>, dn<b>, dn<c>, dn<d>>(Weyl);
    auto nabla_T = partial_field<mu, scheme>(T) -
        rename_field<up<lambda>, dn<mu>, dn<a>>(Gamma) *
            rename_field<dn<lambda>, dn<b>, dn<c>, dn<d>>(T) -
        rename_field<up<lambda>, dn<mu>, dn<b>>(Gamma) *
            rename_field<dn<a>, dn<lambda>, dn<c>, dn<d>>(T) -
        rename_field<up<lambda>, dn<mu>, dn<c>>(Gamma) *
            rename_field<dn<a>, dn<b>, dn<lambda>, dn<d>>(T) -
        rename_field<up<lambda>, dn<mu>, dn<d>>(Gamma) *
            rename_field<dn<a>, dn<b>, dn<c>, dn<lambda>>(T);
    auto D = rename_field<up<d>, up<mu>>(inverse) * nabla_T; // D_abc = nabla^d C_acbd
    auto nabla_D = partial_field<nu, scheme>(D) -
        rename_field<up<lambda>, dn<nu>, dn<a>>(Gamma) * rename_field<dn<lambda>, dn<b>, dn<c>>(D) -
        rename_field<up<lambda>, dn<nu>, dn<b>>(Gamma) * rename_field<dn<a>, dn<lambda>, dn<c>>(D) -
        rename_field<up<lambda>, dn<nu>, dn<c>>(Gamma) * rename_field<dn<a>, dn<b>, dn<lambda>>(D);
    auto raised_once = rename_field<up<c>, up<kappa>>(inverse) * rename_field<dn<kappa>, dn<lambda>>(Ricci);
    auto Ricci_up = raised_once * rename_field<up<lambda>, up<d>>(inverse);
    auto B = rename_field<up<c>, up<nu>>(inverse) * nabla_D + scale_field<1, 2>(Ricci_up * T);
    static_assert(std::is_same_v<typename decltype(B)::result_type, Tensor<double, dn<a>, dn<b>>>);
    using Tree = Stats<typename decltype(B)::node_type>;
    static_assert(Tree::depth == 4, "Bach must retain four nested derivatives of the metric");
    if (metric_calls != 0) return 1;
    std::cout << "Central" << accuracy << "; mathematical nodes=" << Tree::nodes
              << "; partial nodes=" << Tree::partials << "; maximum derivative nesting=" << Tree::depth
              << "; signature characters=" << B.formula().size() << "; metric calls while building=" << metric_calls << '\n';
    if (argc > 1 && std::string_view(argv[1]) == "--type") {
        std::cout << B.formula() << '\n';
        return 0;
    }
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "case h B00 expected00 max_abs_error normalized_error symmetry_error abs_trace Weyl_error metric_calls\n";
    struct Case { const char* name; MetricParameters parameters; coord<mu> x; };
    coord<mu> x;
    x(0) = 0.12; x(1) = -0.18; x(2) = 0.15; x(3) = 0.11;
    const Case cases[]{
        {"product", {0.6, 1.0, 0}, x},
        {"conformal_product", {0.6, 1.0, 1}, x},
        {"equal_curvature", {0.8, 0.8, 0}, x},
        {"conformally_flat", {0, 0, 1}, x}
    };
    bool passed = true;
    for (const auto& item : cases) {
        parameters = item.parameters;
        const auto expected = exact_bach(item.x);
        double reference_scale = 0;
        for (double value : expected.data) reference_scale = std::max(reference_scale, std::abs(value));
        for (double h : {0.02, 0.01, 0.005, 0.0025}) {
            metric_calls = 0;
            const auto value = B.eval(item.x, h);
            const auto weyl = T.eval(item.x, h);
            double error = 0, symmetry = 0, trace = 0, weyl_error = 0;
            const auto diagonal = metric(item.x);
            for (int i = 0; i < 4; ++i) {
                trace += value(i, i) / diagonal(i, i);
                for (int j = 0; j < 4; ++j) {
                    passed = passed && std::isfinite(value(i, j));
                    error = std::max(error, std::abs(value(i, j) - expected(i, j)));
                    symmetry = std::max(symmetry, std::abs(value(i, j) - value(j, i)));
                    for (int k = 0; k < 4; ++k) {
                        for (int l = 0; l < 4; ++l) {
                            passed = passed && std::isfinite(weyl(i, j, k, l));
                            weyl_error = std::max(weyl_error, std::abs(weyl(i, j, k, l) - exact_weyl(item.x, i, k, j, l)));
                        }
                    }
                }
            }
            const double normalized_error = error / std::max(1.0, reference_scale);
            std::cout << item.name << ' ' << h << ' ' << value(0, 0) << ' ' << expected(0, 0) << ' '
                      << error << ' ' << normalized_error << ' ' << symmetry << ' ' << std::abs(trace) << ' '
                      << weyl_error << ' ' << metric_calls << '\n' << std::flush;
            // Scan every step, and require a useful error level at the chosen step.
            if (h == (accuracy == 2 ? 0.0025 : 0.01)) {
                const double tolerance = accuracy == 2 ? 1e-4 : 5e-6;
                passed = passed && std::isfinite(error) && normalized_error < tolerance &&
                         symmetry < tolerance && std::abs(trace) < 4 * tolerance && weyl_error < tolerance;
            }
        }
    }
    std::cout << (passed ? "PASS" : "FAIL") << ": all 16 Bach and 256 Weyl components checked against analytic references\n";
    return passed ? 0 : 1;
}
