#include "kretschmann_formula.hpp"
#include "expression_graph.hpp"
#include "expression_graph_export.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#ifndef K_BENCH_SCHEME
#define K_BENCH_SCHEME 0
#endif

#if defined(_MSC_VER)
#define K_NOINLINE __declspec(noinline)
#else
#define K_NOINLINE __attribute__((noinline))
#endif

inline volatile double benchmark_sink = 0;

template <typename Expression>
K_NOINLINE double evaluate_once(const Expression& expression, const coord<mu>& x, double h) {
    return expression.eval(x, h)();
}

struct Timing {
    double median_us;
    double minimum_us;
    double maximum_us;
    int iterations;
};

template <typename Expression>
Timing measure(const Expression& expression, const coord<mu>& x, double h) {
    using clock = std::chrono::steady_clock;
    for (int i = 0; i < 3; ++i) benchmark_sink = evaluate_once(expression, x, h);
    auto start = clock::now();
    for (int i = 0; i < 3; ++i) benchmark_sink = evaluate_once(expression, x, h);
    const double seconds = std::chrono::duration<double>(clock::now() - start).count() / 3;
    const int iterations = std::max(5, std::min(100000, static_cast<int>(0.06 / std::max(seconds, 1e-8))));
    std::vector<double> values;
    for (int round = 0; round < 7; ++round) {
        start = clock::now();
        for (int i = 0; i < iterations; ++i) benchmark_sink = evaluate_once(expression, x, h);
        values.push_back(std::chrono::duration<double, std::micro>(clock::now() - start).count() / iterations);
    }
    std::sort(values.begin(), values.end());
    return {values[3], values.front(), values.back(), iterations};
}

int main(int argc, char** argv) {
    constexpr auto scheme = static_cast<DiffScheme>(K_BENCH_SCHEME);
    constexpr int order = 2 + 2*K_BENCH_SCHEME;
    const auto K = kretschmann_formula<scheme>();
    const auto graph = compile(K);
    using Graph = decltype(compile(K));
    if (argc > 2) graph_export::export_all<decltype(K), Graph>(argv[2]);
    if (argc > 3 && std::string(argv[3]) == "--export-only") return 0;
    bool passed = metric_calls == 0;
    std::ofstream file;
    if (argc > 1) file.open(argv[1]);
    std::ostream& json = file.is_open() ? file : std::cout;
    json << std::setprecision(17) << "{\"scheme\":\"Central" << order << "\",\"nodes\":" << Graph::node_count
         << ",\"cached_nodes\":" << Graph::cache_count << ",\"checks\":[";
    bool first = true;
    const double points[][2] = {{4,0.7},{6,1.1},{10,std::acos(-1.0)/2}};
    const double base_step = order == 2 ? 0.002 : 0.02;
    for (const auto& point : points) {
        coord<mu> x{};
        x(1) = point[0]; x(2) = point[1];
        for (double h : {base_step, base_step/2}) {
            metric_calls = 0;
            const auto direct = evaluate_once(K, x, h);
            const auto direct_calls = metric_calls;
            metric_calls = 0;
            const auto shared = evaluate_once(graph, x, h);
            const auto graph_calls = metric_calls;
            const double reference = 48/std::pow(x(1),6);
            const double difference = std::abs(shared/direct - 1);
            const double reference_error = std::abs(shared/reference - 1);
            passed = passed && std::isfinite(shared) && difference < 5e-8 &&
                reference_error < (order == 2 ? 2e-4 : 5e-7) && graph_calls < direct_calls;
            if (!first) json << ',';
            first = false;
            json << "{\"r\":" << x(1) << ",\"theta\":" << x(2) << ",\"h\":" << h
                 << ",\"direct\":" << direct << ",\"dag\":" << shared << ",\"closed\":" << reference
                 << ",\"relative_difference\":" << difference << ",\"relative_reference_error\":" << reference_error
                 << ",\"direct_metric_calls\":" << direct_calls << ",\"dag_metric_calls\":" << graph_calls << '}';
        }
    }
    coord<mu> x{};
    x(1) = 6; x(2) = 1.1;
    const auto direct = measure(K, x, base_step/2);
    const auto shared = measure(graph, x, base_step/2);
    json << "],\"benchmark\":{\"r\":6,\"theta\":1.1,\"h\":" << base_step/2
         << ",\"rounds\":7,\"warmup\":3,\"direct_us\":" << direct.median_us
         << ",\"dag_us\":" << shared.median_us << ",\"speedup\":" << direct.median_us/shared.median_us
         << ",\"direct_min_us\":" << direct.minimum_us << ",\"direct_max_us\":" << direct.maximum_us
         << ",\"dag_min_us\":" << shared.minimum_us << ",\"dag_max_us\":" << shared.maximum_us
         << ",\"direct_iterations\":" << direct.iterations << ",\"dag_iterations\":" << shared.iterations
         << "},\"passed\":" << (passed ? "true" : "false") << "}\n";
    if (file.is_open()) std::cout << "Central" << order << ": nodes=" << Graph::node_count
        << " cached=" << Graph::cache_count << " direct_us=" << direct.median_us
        << " dag_us=" << shared.median_us << " speedup=" << direct.median_us/shared.median_us
        << " passed=" << passed << '\n';
    return passed ? 0 : 1;
}
