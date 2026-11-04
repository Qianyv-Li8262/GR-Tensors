#include "expression_graph.hpp"
#include <cmath>
#include <iostream>
#include <vector>

namespace {
int failures = 0;
int calls = 0;
int other_calls = 0;
std::vector<std::array<double, 4>> samples;

void check(bool condition, const char* message) {
    if (!condition) { ++failures; std::cerr << message << '\n'; }
}

template <typename T>
void compare(const T& actual, const T& expected, double tolerance = 1e-9) {
    for (std::size_t i = 0; i < actual.data.size(); ++i)
        check(std::isfinite(actual.data[i]) && std::abs(actual.data[i] - expected.data[i]) <=
            tolerance * std::max(1.0, std::abs(expected.data[i])), "DAG/direct component mismatch");
}

auto polynomial(const coord<mu>& x) {
    ++calls;
    samples.push_back(x.data);
    return Tensor<double>(std::pow(x(0), 4) + 2*x(0)*x(1) + 3*x(2)*x(2) + x(1)*x(3));
}

auto other(const coord<mu>& x) {
    ++other_calls;
    return Tensor<double>(2 + x(0)*x(1));
}

auto matrix(const coord<mu>& x) {
    Tensor<double, dn<a>, dn<b>> result;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            result(i,j) = (i == j ? 5 + i + x(0) : 0) + 0.1*(i+1) - 0.03*j + 0.02*j*x(1);
    return result;
}

auto cube(const coord<mu>& x) {
    Tensor<double, dn<a>, dn<b>, dn<c>> result;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k) result(i,j,k) = 10*i + 2*j - k + (j+1)*x(0);
    return result;
}

template <typename Operation>
struct is_field : std::false_type {};
template <auto Function, typename Offset>
struct is_field<dag_detail::FieldOp<Function, Offset>> : std::true_type {};

template <typename Graph>
struct graph_stats;
template <typename Output, typename... Nodes>
struct graph_stats<ExprGraph<Output, Nodes...>> {
    static constexpr std::size_t leaf_count = (0 + ... + is_field<typename Nodes::operation>::value);
    template <typename Operation>
    static constexpr std::size_t find = [] {
        std::size_t id = dag_detail::not_found;
        ((std::is_same_v<Operation, typename Nodes::operation> ? id = Nodes::id : id), ...);
        return id;
    }();
};

template <DiffScheme Scheme>
void check_first(const coord<mu>& x) {
    const auto p = partial_field<nu, Scheme>(make_field<&polynomial>());
    calls = 0;
    const auto graph = compile(p);
    static_assert(decltype(graph)::requires_step);
    constexpr auto count = 4 * derivative_stencil<Scheme>::offsets.size();
    static_assert(graph_stats<decltype(compile(p))>::leaf_count == count);
    check(calls == 0, "compile must not sample derivatives");
    const double h = 0.05;
    const auto actual = graph.eval(x, h);
    check(calls == count, "first derivative must visit exactly its stencil points");
    compare(actual, p.eval(x, h));
    const double bias = Scheme == DiffScheme::Central2 ? 4*x(0)*h*h : 0;
    check(std::abs(actual(0) - (4*std::pow(x(0),3) + 2*x(1) + bias)) < 1e-9,
          "first derivative differs from analytic polynomial");
}

void check_second(const coord<mu>& x) {
    const auto first = partial_field<nu, DiffScheme::Central2>(make_field<&polynomial>());
    const auto second = partial_field<rho, DiffScheme::Central2>(first);
    calls = 0;
    const auto graph = compile(second);
    using Graph = decltype(compile(second));
    using Stats = graph_stats<Graph>;
    constexpr auto center = Stats::template find<dag_detail::FieldOp<&polynomial, offset<0,0,0,0>>>;
    static_assert(center != dag_detail::not_found);
    static_assert(Graph::template use_count<center> > 1);
    static_assert(Stats::leaf_count == 33);
    check(calls == 0, "second derivative construction must not sample");
    for (double h : {0.05, -0.025}) {
        calls = 0;
        samples.clear();
        const auto actual = graph.eval(x, h);
        check(calls == 33, "second derivative must reuse equal absolute offsets");
        check(std::count(samples.begin(), samples.end(), x.data) == 1, "zero-offset sample must run once");
        calls = 0;
        const auto direct = second.eval(x, h);
        check(calls == 64, "direct second derivative must retain 64 sampling paths");
        compare(actual, direct);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                double expected = 0;
                if (i == 0 && j == 0) expected = 12*x(0)*x(0) + 8*h*h;
                if ((i == 0 && j == 1) || (i == 1 && j == 0)) expected = 2;
                if ((i == 1 && j == 3) || (i == 3 && j == 1)) expected = 1;
                if (i == 2 && j == 2) expected = 6;
                check(std::abs(actual(i,j) - expected) < 1e-8, "second derivative analytic reference mismatch");
            }
    }
    const auto shifted = expand_at<offset<1,-1,0,2>>(second);
    compare(compile(shifted).eval(x, 0.05), shifted.eval(x, 0.05));
    // A fresh coordinate must also start with an empty sample cache.
    auto next = x;
    next(0) += 0.7;
    calls = 0;
    const auto next_value = graph.eval(next, 0.05);
    check(calls == 33, "sample cache must reset for each eval");
    compare(next_value, second.eval(next, 0.05));
    std::cout << "Central2 Hessian: direct=64 samples, DAG=33, center=1; nodes=" << Graph::node_count << '\n';
}

void check_operations(const coord<mu>& x) {
    const auto M = make_field<&matrix>();
    const auto R = make_field<&cube>();
    const auto transposed = permute<1,0>(M);
    static_assert(decltype(compile(transposed))::node_count == 1);
    compare(compile(transposed).eval(x), transposed.eval(x));
    const auto scaled = scale_field<2,3>(transposed);
    compare(compile(scaled).eval(x), scaled.eval(x));
    const auto addition = M + rename_field<dn<b>,dn<a>>(M) - scale_field<1,3>(M);
    compare(compile(addition).eval(x), addition.eval(x));
    const auto cycle = R + rename_field<dn<b>,dn<c>,dn<a>>(R);
    compare(compile(cycle).eval(x), cycle.eval(x));
    const auto composed = permute<2,0,1>(cycle);
    compare(compile(composed).eval(x), composed.eval(x));

    const auto symmetric = sym<0,1>(transposed);
    const auto antisymmetric = asym<0,1>(transposed);
    static_assert(decltype(compile(symmetric))::node_count == 2);
    static_assert(decltype(compile(antisymmetric))::node_count == 2);
    compare(compile(symmetric).eval(x), symmetric.eval(x));
    compare(compile(antisymmetric).eval(x), antisymmetric.eval(x));
    const auto reused = symmetric + symmetric;
    static_assert(decltype(compile(reused))::cache_count == 1);
    compare(compile(reused).eval(x), reused.eval(x));
    const auto selected = asym<0,2>(permute<2,0,1>(R));
    compare(compile(selected).eval(x), selected.eval(x));

    const auto inverse = inverse_field(M);
    compare(compile(inverse).eval(x), inverse.eval(x));
    const auto inverse_transpose = inverse_field(transposed);
    compare(compile(inverse_transpose).eval(x), inverse_transpose.eval(x));
    const auto identity = M * rename_field<up<b>,up<c>>(inverse);
    const auto trace = trace_field<a,c>(identity);
    compare(compile(trace).eval(x), trace.eval(x));
    check(std::abs(compile(trace).eval(x)() - 4) < 1e-10, "trace of M * inverse(M) must equal four");
    const auto outer = M * rename_field<up<c>,up<d>>(inverse);
    const auto mapped_trace = trace_field<b,c>(permute<3,1,2,0>(outer));
    compare(compile(mapped_trace).eval(x), mapped_trace.eval(x));

    // Derivatives lower through the other modules too, without treating them
    // as opaque recursive evaluation leaves.
    const auto composite = partial_field<nu, DiffScheme::Central2>(
        sym<0,1>(M) + permute<1,0>(asym<0,1>(M)));
    compare(compile(composite).eval(x, 0.05), composite.eval(x, 0.05));
    const auto inverse_derivative = partial_field<nu, DiffScheme::Central2>(inverse);
    compare(compile(inverse_derivative).eval(x, 0.05), inverse_derivative.eval(x, 0.05));
    const auto trace_derivative = partial_field<nu, DiffScheme::Central2>(trace);
    compare(compile(trace_derivative).eval(x, 0.05), trace_derivative.eval(x, 0.05));
    const auto product = partial_field<nu, DiffScheme::Central2>(make_field<&polynomial>() * make_field<&other>());
    compare(compile(product).eval(x, 0.05), product.eval(x, 0.05));
}
} // namespace

int main() {
    coord<mu> x;
    x.data = {0.5,-0.25,0.75,1.25};
    check_first<DiffScheme::Central2>(x);
    check_first<DiffScheme::Central4>(x);
    check_first<DiffScheme::Central6>(x);
    check_first<DiffScheme::Central8>(x);
    check_second(x);
    check_operations(x);
    if (!failures) std::cout << "DAG derivative lowering, offset CSE and all expression modules passed\n";
    return failures ? 1 : 0;
}
