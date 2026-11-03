#include "tensor_expression.hpp"

auto metric(const Tensor<double, up<mu>>&) {
    return Tensor<double, dn<a>, dn<b>>{};
}

auto invalid = rename_field<up<a>, dn<b>>(make_field<&metric>());
