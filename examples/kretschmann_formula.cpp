#include "kretschmann_formula.hpp"
#include <iostream>
#include <string_view>

// Diagnostics only: count mathematical node occurrences and distinct node types.
template <typename... Lists>
struct join_types;
template <>
struct join_types<> { using result = type_list<>; };
template <typename Head, typename... Tail>
struct join_types<Head, Tail...> {
    using result = typename concat<Head, typename join_types<Tail...>::result>::result;
};

template <typename Node>
struct is_partial : std::false_type {};
template <typename Node, DiffScheme Scheme, typename O>
struct is_partial<PartialExpr<Node, Scheme, O>> : std::true_type {};

template <typename Node>
struct is_sampled : std::false_type {};
template <typename D0, typename D1, typename D2, typename D3, typename O>
struct is_sampled<SampledPartialExpr<D0, D1, D2, D3, O>> : std::true_type {};
template <DiffScheme Scheme, typename O, typename... Samples>
struct is_sampled<StencilSumExpr<Scheme, O, Samples...>> : std::true_type {};

template <typename Node>
struct tree_stats;
template <typename Node>
struct is_metric_leaf : std::false_type {};
template <typename O>
struct is_metric_leaf<FieldExpr<&metric, O>> : std::true_type {};
template <typename List>
struct child_stats;
template <typename... Children>
struct child_stats<type_list<Children...>> {
    static constexpr std::size_t nodes = (tree_stats<Children>::nodes + ... + 0);
    static constexpr std::size_t leaves = (tree_stats<Children>::leaves + ... + 0);
    static constexpr std::size_t partials = (tree_stats<Children>::partials + ... + 0);
    static constexpr std::size_t sampled = (tree_stats<Children>::sampled + ... + 0);
    static constexpr bool only_metric_leaves = (tree_stats<Children>::only_metric_leaves && ...);
    static constexpr std::size_t derivative_order = [] {
        std::size_t maximum = 0;
        ((maximum = std::max(maximum, tree_stats<Children>::derivative_order)), ...);
        return maximum;
    }();
    using types = typename join_types<typename tree_stats<Children>::types...>::result;
};
template <typename Node>
struct tree_stats {
    using child = child_stats<typename Node::children>;
    static constexpr std::size_t nodes = 1 + child::nodes;
    static constexpr std::size_t leaves = list_size<typename Node::children>::value == 0 ? 1 : child::leaves;
    static constexpr std::size_t partials = is_partial<Node>::value + child::partials;
    static constexpr std::size_t sampled = is_sampled<Node>::value + child::sampled;
    static constexpr std::size_t derivative_order = is_partial<Node>::value + child::derivative_order;
    static constexpr bool only_metric_leaves = list_size<typename Node::children>::value == 0
                                                   ? is_metric_leaf<Node>::value : child::only_metric_leaves;
    using types = typename concat<type_list<Node>, typename child::types>::result;
};

int main(int argc, char** argv) {
    // All expression objects are created here; metric is the only external function leaf.
    auto g = make_field<&metric>();
    auto inverse = inverse_field(g);
    auto pg = partial_field<mu, DiffScheme::Central8>(g); // [mu, a, b]

    // Gamma^a_bc = 1/2 g^{a kappa}(d_b g_{kappa c} + d_c g_{kappa b} - d_kappa g_{bc}).
    auto Gamma = scale_field<1, 2>(
        rename_field<up<a>, up<kappa>>(inverse) *
        (rename_field<dn<b>, dn<kappa>, dn<c>>(pg) +
         rename_field<dn<c>, dn<kappa>, dn<b>>(pg) -
         rename_field<dn<kappa>, dn<b>, dn<c>>(pg)));
    auto pGamma = partial_field<mu, DiffScheme::Central8>(Gamma); // [mu, a, b, c]

    auto quadratic1 = rename_field<up<rho>, dn<c>, dn<lambda>>(Gamma) *
                      rename_field<up<lambda>, dn<d>, dn<b>>(Gamma);
    auto quadratic2 = rename_field<up<rho>, dn<d>, dn<lambda>>(Gamma) *
                      rename_field<up<lambda>, dn<c>, dn<b>>(Gamma);
    auto derivative1 = rename_field<dn<c>, up<rho>, dn<d>, dn<b>>(pGamma);
    auto derivative2 = rename_field<dn<d>, up<rho>, dn<c>, dn<b>>(pGamma);
    auto R = quadratic1 - quadratic2 + derivative1 - derivative2; // [rho, c, d, b]

    auto R_lower = rename_field<dn<a>, dn<rho>>(g) * R; // [a, c, d, b]
    auto renamed = rename_field<dn<mu>, dn<kappa>, dn<sigma>, dn<nu>>(R_lower);
    auto raise_a = rename_field<up<a>, up<mu>>(inverse) * renamed;
    auto raise_b = rename_field<up<b>, up<nu>>(inverse) * raise_a;
    auto raise_c = rename_field<up<c>, up<kappa>>(inverse) * raise_b;
    auto R_upper = rename_field<up<d>, up<sigma>>(inverse) * raise_c;
    auto K = R_lower * R_upper;

    // Unevaluated comparison with the reusable builder used by the independent numeric check.
    static_assert(std::is_same_v<decltype(K), decltype(kretschmann_formula())>,
                  "building inside main must produce exactly the same mathematical type");
    using Formula = decltype(K)::formula_type;
    using Stats = tree_stats<Formula::node_type>;
    using Unique = typename remove_duplicate<typename Stats::types, type_list<>>::result;
    static_assert(Stats::sampled == 0, "the formula must not instantiate derivative sampling subtrees");
    static_assert(Stats::derivative_order == 2, "K reaches second derivatives of the metric");
    static_assert(Stats::only_metric_leaves, "the metric must be the only opaque function leaf");

    if (metric_calls != 0) return 1;
    std::cout << "K = R_abcd R^abcd; result = Tensor<double>\n"
              << "mathematical node occurrences: " << Stats::nodes << '\n'
              << "distinct mathematical node types: " << list_size<Unique>::value << '\n'
              << "metric leaf occurrences: " << Stats::leaves << '\n'
              << "partial node occurrences: " << Stats::partials << '\n'
              << "maximum derivative nesting: " << Stats::derivative_order << '\n'
              << "sampling nodes in formula: " << Stats::sampled << '\n'
              << "metric evaluations while building/inspecting: " << metric_calls << '\n'
              << "compiler signature characters: " << K.formula().size() << '\n';
    if (argc > 1 && std::string_view(argv[1]) == "--type") {
        std::cout << "\nK.formula():\n" << K.formula() << '\n';
    }
}
