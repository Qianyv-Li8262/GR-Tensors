#pragma once

#include "expression_graph.hpp"
#include <fstream>

// Save the already interned node pack as C++, without the source tree or build
// states. Include the leaf function definitions before including this header's
// generated output. Loading it never invokes compile() again.
template <typename Graph>
bool export_graph_type(const char* path, const char* alias) {
#if defined(__clang__)
    const auto signature = dag_detail::type_name<std::remove_cv_t<Graph>>();
    const auto begin = signature.find("T = ");
    const auto end = signature.rfind(']');
    if (begin == std::string_view::npos || end == std::string_view::npos || end <= begin + 4) return false;
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "// Generated from an actual compiled ExprGraph. Requires its leaf definitions.\n"
           "#pragma once\n#include \"expression_graph.hpp\"\n\nusing " << alias << " = "
        << signature.substr(begin + 4, end - begin - 4) << ";\n";
    return static_cast<bool>(out);
#else
    static_assert(dag_detail::unsupported_node<Graph>, "graph type export currently requires Clang signatures");
#endif
}
