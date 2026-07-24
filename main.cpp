#include "tensors_all.hpp"
template <typename a, typename b, typename datatype, typename letter>
Tensor<datatype, dn<a>, dn<b>> metric(const Tensor<datatype, up<letter>>& x) {
    Tensor<datatype, dn<a>, dn<b>> g{};

    double r = x(1);
    double theta = x(2);
    double rs = 2.0;
    double f = 1.0 - rs / r;

    g(0, 0) = -f;
    g(1, 1) = 1.0 / f;
    g(2, 2) = r * r;
    g(3, 3) = r * r * std::sin(theta) * std::sin(theta);

    return g;
}
template <typename a, typename b, typename datatype, typename letter>
Tensor<datatype, up<a>, up<b>> inv_metric(const Tensor<datatype, up<letter>>& x) {
    Tensor<datatype, up<a>, up<b>> g{};

    double r = x(1);
    double theta = x(2);
    double rs = 2.0;
    double f = 1.0 - rs / r;

    g(0, 0) = -1.0/f;
    g(1, 1) = f;
    g(2, 2) = 1.0/r / r;
    g(3, 3) = 1.0/(r * r * std::sin(theta) * std::sin(theta));

    return g;
}

DEFINE_TENSOR_FIELD2(g, metric);
DEFINE_TENSOR_FIELD2(ginv, inv_metric);
