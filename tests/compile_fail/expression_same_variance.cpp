#include "tensor_expression.hpp"

auto metric(const Tensor<double, up<mu>>&) {
    return Tensor<double, dn<a>, dn<b>>{};
}

// Rejection must happen when constructing the tree, without eval().
auto invalid = make_field<&metric>() * make_field<&metric>();
