#include "tensor_expression.hpp"
#include "bach_formula.hpp"
#if defined(BACH_COMPILE_GRAPH)
#include "expression_graph.hpp"
#endif
#if defined(BACH_EXPORT_GRAPH)
#include "graph_type_export.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>

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
    auto expressions = make_bach_expressions<scheme>();
    auto B = std::move(expressions.bach);
    auto T = std::move(expressions.weyl);
    static_assert(std::is_same_v<typename decltype(B)::result_type, Tensor<double, dn<a>, dn<b>>>);
    using Tree = Stats<typename decltype(B)::node_type>;
    static_assert(Tree::depth == 4, "Bach must retain four nested derivatives of the metric");
#if defined(BACH_COMPILE_GRAPH)
    const auto graph = compile(B);
    using Graph = decltype(graph);
    static_assert(std::is_same_v<typename Graph::result_type, typename decltype(B)::result_type>);
    std::cout << "Bach Central" << accuracy << "; mathematical nodes=" << Tree::nodes
              << "; maximum derivative nesting=" << Tree::depth
              << "; DAG nodes=" << Graph::node_count << "; cached nodes=" << Graph::cache_count
              << "; root=" << Graph::root_id << "; metric calls while building=" << metric_calls << '\n';
#if defined(BACH_EXPORT_GRAPH)
    const char* path = argc > 1 ? argv[1] : "outputs/bach_compiled_graph_central2.hpp";
    if (!export_graph_type<Graph>(path, "BachCompiledGraph")) return 2;
    std::cout << "Saved reusable graph type: " << path << '\n';
#endif
    return metric_calls == 0 ? 0 : 1;
#else
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
#endif
}
