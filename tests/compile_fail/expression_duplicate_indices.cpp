#include "tensor_expression.hpp"

auto metric(const Tensor<double, up<mu>>&) {
    return Tensor<double, dn<a>, dn<b>>{};
}

auto invalid = rename_field<dn<a>, dn<a>>(make_field<&metric>());
