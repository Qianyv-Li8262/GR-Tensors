#include "graph_type_export.hpp"

auto export_sample(const coord<mu>& x) { return Tensor<double>(2+x(0)*x(0)); }

int main(int argc, char** argv) {
    const auto f = make_field<&export_sample>();
    const auto expression = partial_field<nu,DiffScheme::Central2>(f*f);
    return argc == 2 && export_graph_type<decltype(compile(expression))>(argv[1], "SavedGraph") ? 0 : 1;
}
