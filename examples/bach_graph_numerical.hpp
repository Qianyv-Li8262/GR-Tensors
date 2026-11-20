#pragma once
#include "bach_metric_reference.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace bach_test {
inline volatile double sink = 0;
#if defined(_MSC_VER)
#define BACH_NOINLINE __declspec(noinline)
#else
#define BACH_NOINLINE __attribute__((noinline))
#endif

template <typename Expression>
BACH_NOINLINE auto evaluate(const Expression& expression, const coord<mu>& x, double h) {
    return expression.eval(x, h);
}

struct Timing { double median_us, minimum_us, maximum_us; int iterations; };

template <typename Expression>
Timing measure(const Expression& expression, const coord<mu>& x, double h) {
    using clock = std::chrono::steady_clock;
    for (int i=0; i<3; ++i) sink = evaluate(expression,x,h).data[0];
    auto start = clock::now();
    sink = evaluate(expression,x,h).data[0];
    const double seconds = std::chrono::duration<double>(clock::now()-start).count();
    const int iterations = std::max(1,std::min(10000,static_cast<int>(0.05/std::max(seconds,1e-8))));
    std::vector<double> rounds;
    for (int round=0; round<7; ++round) {
        start = clock::now();
        for (int i=0; i<iterations; ++i) sink = evaluate(expression,x,h).data[0];
        rounds.push_back(std::chrono::duration<double,std::micro>(clock::now()-start).count()/iterations);
    }
    std::sort(rounds.begin(),rounds.end());
    return {rounds[3],rounds.front(),rounds.back(),iterations};
}

template <typename Graph, typename Direct>
int run(const Graph& graph, const Direct& direct, const char* result_path) {
    static_assert(std::is_same_v<typename Graph::result_type,typename Direct::result_type>);
    std::ofstream json(result_path);
    if (!json) return 2;
    bool passed = true, first = true;
    std::cout << std::scientific << std::setprecision(10);
    json << std::setprecision(17) << "{\"scheme\":\"Central2\",\"nodes\":" << Graph::node_count
         << ",\"cached_nodes\":" << Graph::cache_count << ",\"cache_bytes\":" << sizeof(typename Graph::cache_type)
         << ",\"checks\":[";
    coord<mu> x{};
    x.data = {0.12,-0.18,0.15,0.11};
    struct Case { const char* name; MetricParameters parameters; };
    const Case cases[] = {{"product",{0.6,1,0}}, {"conformal_product",{0.6,1,1}},
                          {"equal_curvature",{0.8,0.8,0}}, {"conformally_flat",{0,0,1}}};
    std::cout << "case h max_direct_dag_difference analytic_error symmetry_error abs_trace direct_calls dag_calls\n";
    for (const auto& item : cases) {
        parameters = item.parameters;
        const auto expected = exact_bach(x);
        const auto diagonal = metric(x);
        double reference_scale = 1;
        for (auto value : expected.data) reference_scale = std::max(reference_scale,std::abs(value));
        for (double h : {0.02,0.01,0.005,0.0025}) {
            metric_calls = 0;
            const auto shared = evaluate(graph,x,h);
            const auto graph_calls = metric_calls;
            metric_calls = 0;
            const auto unshared = evaluate(direct,x,h);
            const auto direct_calls = metric_calls;
            double difference = 0, error = 0, symmetry = 0, trace = 0;
            bool finite = true;
            for (int i=0; i<4; ++i) {
                trace += shared(i,i)/diagonal(i,i);
                for (int j=0; j<4; ++j) {
                    finite = finite && std::isfinite(shared(i,j)) && std::isfinite(unshared(i,j));
                    difference = std::max(difference,std::abs(shared(i,j)-unshared(i,j)));
                    error = std::max(error,std::abs(shared(i,j)-expected(i,j)));
                    symmetry = std::max(symmetry,std::abs(shared(i,j)-shared(j,i)));
                }
            }
            const double comparison_tolerance = h < 0.005 ? 5e-5 : 2e-6;
            bool good = finite && difference/reference_scale < comparison_tolerance && graph_calls < direct_calls;
            if (h == 0.0025) good = good && error/reference_scale < 1e-4 && symmetry < 1e-4 && std::abs(trace) < 4e-4;
            passed = passed && good;
            std::cout << item.name << ' ' << h << ' ' << difference << ' ' << error/reference_scale << ' '
                      << symmetry << ' ' << std::abs(trace) << ' ' << direct_calls << ' ' << graph_calls << '\n' << std::flush;
            if (!first) json << ',';
            first = false;
            json << "{\"case\":\"" << item.name << "\",\"h\":" << h << ",\"B00\":" << shared(0,0)
                 << ",\"expected00\":" << expected(0,0) << ",\"max_absolute_difference\":" << difference
                 << ",\"normalized_difference\":" << difference/reference_scale
                 << ",\"normalized_analytic_error\":" << error/reference_scale << ",\"symmetry_error\":" << symmetry
                 << ",\"abs_trace\":" << std::abs(trace) << ",\"direct_metric_calls\":" << direct_calls
                 << ",\"dag_metric_calls\":" << graph_calls << ",\"passed\":" << (good?"true":"false") << '}';
            json.flush();
        }
    }
    parameters = {0.6,1,1};
    std::cout << "Running warmed seven-round CPU benchmark...\n" << std::flush;
    const auto direct_time = measure(direct,x,0.01);
    const auto graph_time = measure(graph,x,0.01);
    const double speedup = direct_time.median_us/graph_time.median_us;
    json << "],\"benchmark\":{\"h\":0.01,\"case\":\"conformal_product\",\"warmup\":3,\"rounds\":7,"
            "\"direct_us\":" << direct_time.median_us << ",\"dag_us\":" << graph_time.median_us
         << ",\"speedup\":" << speedup << ",\"direct_min_us\":" << direct_time.minimum_us
         << ",\"direct_max_us\":" << direct_time.maximum_us << ",\"dag_min_us\":" << graph_time.minimum_us
         << ",\"dag_max_us\":" << graph_time.maximum_us << ",\"direct_iterations\":" << direct_time.iterations
         << ",\"dag_iterations\":" << graph_time.iterations << "},\"passed\":" << (passed?"true":"false") << "}\n";
    std::cout << (passed?"PASS":"FAIL") << "; direct_us=" << direct_time.median_us
              << "; dag_us=" << graph_time.median_us << "; speedup=" << speedup << '\n';
    return passed?0:1;
}
} // namespace bach_test
