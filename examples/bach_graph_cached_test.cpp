#include "bach_formula.hpp"
#include "../outputs/bach_compiled_graph_central2.hpp"
#include "bach_graph_numerical.hpp"

int main(int argc, char** argv) {
    const auto expressions = make_bach_expressions<DiffScheme::Central2>();
    const BachCompiledGraph graph;
    std::cout << "Loaded saved Bach DAG: nodes=" << BachCompiledGraph::node_count
              << "; cached=" << BachCompiledGraph::cache_count << '\n' << std::flush;
    return bach_test::run(graph,expressions.bach,argc>1?argv[1]:"outputs/bach_graph_test_results.json");
}
