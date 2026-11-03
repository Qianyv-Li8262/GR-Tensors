#include "field_wrapper.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

namespace {

int metric_calls = 0;
int inverse_calls = 0;
int coordinate_calls = 0;
int derivative_calls = 0;
std::vector<std::array<double, 4>> derivative_samples;
int failures = 0;

// Dense, coordinate-dependent Lorentz metric: a nondiagonal test catches axis mistakes.
auto metric(const Tensor<double, up<mu>>& x) {
    ++metric_calls;
    Tensor<double, dn<a>, dn<b>> g;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const double u_i = 0.1 * (i + 1);
            const double u_j = 0.1 * (j + 1);
            g(i, j) = (i == j ? (i == 0 ? -2.0 - x(1) : 2.0 + i) : 0.0) + u_i * u_j;
        }
    }
    return g;
}

// Independent Sherman-Morrison inverse of D + u u^T.
auto inverse_metric(const Tensor<double, up<mu>>& x) noexcept {
    ++inverse_calls;
    Tensor<double, up<a>, up<b>> inverse;
    const double diagonal[] = {-2.0 - x(1), 3.0, 4.0, 5.0};
    double denominator = 1.0;
    for (int i = 0; i < 4; ++i) {
        const double u = 0.1 * (i + 1);
        denominator += u * u / diagonal[i];
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            inverse(i, j) = (i == j ? 1.0 / diagonal[i] : 0.0) -
                            (0.1 * (i + 1) / diagonal[i]) * (0.1 * (j + 1) / diagonal[j]) / denominator;
        }
    }
    return inverse;
}

auto scalar_field(const Tensor<double, up<mu>>& x) {
    return Tensor<double>(2.0 + x(1));
}

auto other_scalar_field(const Tensor<double, up<mu>>& x) {
    return Tensor<double>(3.0 + x(1));
}

auto coordinate_field(const Tensor<double, up<mu>>& x) {
    ++coordinate_calls;
    Tensor<double, up<a>> result;
    result.data = x.data;
    return result;
}

template <int Power>
auto monomial_field(const coord<mu>& x) {
    ++derivative_calls;
    derivative_samples.push_back(x.data);
    Tensor<double, up<a>> result;
    for (int axis = 0; axis < 4; ++axis) {
        result(axis) = std::pow(x(axis), Power);
    }
    return result;
}

auto polynomial_field(const coord<mu>& x) {
    ++derivative_calls;
    derivative_samples.push_back(x.data);
    return Tensor<double>(std::pow(x(0), 4) + x(0) * x(1) * x(2));
}

auto identity_inverse(const coord<mu>&) {
    Tensor<double, up<a>, up<b>> result;
    for (int axis = 0; axis < 4; ++axis) {
        result(axis, axis) = 1.0;
    }
    return result;
}

static_assert(std::is_same_v<decltype(&scalar_field), decltype(&other_scalar_field)>);
static_assert(!std::is_same_v<FieldExpr<&scalar_field>, FieldExpr<&other_scalar_field>>);

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        ++failures;
    }
}

bool close(double actual, double expected) {
    return std::isfinite(actual) && std::abs(actual - expected) < 1e-11 * std::max(1.0, std::abs(expected));
}

void check_offsets() {
    using O = offset<1, -2, 3, -4>;
    using S = FieldExpr<&scalar_field, O>;
    using V = FieldExpr<&coordinate_field, O>;
    using P00 = contraction_plan<0, 0>;
    using P01 = contraction_plan<0, 1>;

    const auto scalar = make_field<&scalar_field>();
    const auto original = scalar * scalar * make_field<&coordinate_field>();
    const auto shifted = expand_at<O>(original);
    const auto renamed = rename_field<up<b>>(shifted);
    const auto direct = make_field<&scalar_field, O>() * make_field<&scalar_field, O>() *
                        make_field<&coordinate_field, O>();
    const auto reset = expand_at<>(shifted);
    const auto relocated = expand_at<offset<-1, 0, 0, 2>>(shifted);

    static_assert(std::is_same_v<offset<>, offset<0, 0, 0, 0>>);
    static_assert(std::is_same_v<typename decltype(original)::point_offset, offset<>>);
    static_assert(std::is_same_v<typename decltype(shifted)::point_offset, O>);
    static_assert(!std::is_same_v<typename decltype(original)::node_type, typename decltype(shifted)::node_type>);
    static_assert(std::is_same_v<typename decltype(shifted)::node_type,
                                 ContractExpr<ContractExpr<S, S, P00, O>, V, P01, O>>);
    static_assert(std::is_same_v<typename decltype(direct)::node_type, typename decltype(shifted)::node_type>);
    static_assert(std::is_same_v<typename decltype(renamed)::node_type, typename decltype(shifted)::node_type>);
    static_assert(std::is_same_v<typename decltype(reset)::node_type, typename decltype(original)::node_type>);
    check(coordinate_calls == 0, "building and expanding offset trees must not evaluate leaves");

    Tensor<double, up<mu>> x;
    x.data = {1.0, 2.0, 3.0, 4.0};
    for (double h : {0.25, 0.5}) {
        const auto shifted_value = shifted.eval(x, h);
        const auto direct_value = direct.eval(x, h);
        const auto renamed_value = renamed(x, h);
        const auto reset_value = reset.eval(x, h);
        const auto relocated_value = relocated.eval(x, h);
        for (std::size_t axis = 0; axis < 4; ++axis) {
            const double sample = x(axis) + h * O::values[axis];
            const double factor = 2.0 + x(1) - 2 * h;
            const double expected = factor * factor * sample;
            check(close(shifted_value(axis), expected), "nested tree must shift each leaf exactly once");
            check(close(direct_value(axis), expected), "factory offsets must propagate through multiplication");
            check(close(renamed_value(axis), expected), "rename must preserve offsets");
            check(close(reset_value(axis), 16.0 * x(axis)), "zero offset must restore root sampling");
            const int new_offset[] = {-1, 0, 0, 2};
            check(close(relocated_value(axis), 16.0 * (x(axis) + h * new_offset[axis])),
                  "expand_at must replace the absolute offset rather than add it again");
        }
    }
    const auto default_step = shifted.eval(x);
    for (std::size_t axis = 0; axis < 4; ++axis) {
        check(close(default_step(axis), (4.0 - 0.02) * (4.0 - 0.02) * (x(axis) + 0.01 * O::values[axis])),
              "offset evaluation must support the default step");
        check(x(axis) == axis + 1.0, "leaf sampling must not modify the root coordinate");
    }
}

template <DiffScheme Scheme, int Power, int Points>
void check_derivative_scheme() {
    derivative_calls = 0;
    derivative_samples.clear();
    const auto field = make_field<&monomial_field<Power>>();
    const auto derivative = partial_field<mu, Scheme>(field);
    using tree = typename decltype(derivative)::node_type;
    using sampled = typename expression_detail::partial_samples<typename decltype(field)::node_type, Scheme>::result;
    using direction = typename nth_type<0, typename sampled::children>::result;
    static_assert(list_size<typename tree::children>::value == 1);
    static_assert(list_size<typename sampled::children>::value == 4);
    static_assert(list_size<typename direction::children>::value == Points);
    static_assert(std::is_same_v<typename decltype(derivative)::result_type, Tensor<double, dn<mu>, up<a>>>);
    check(derivative_calls == 0, "building a derivative must not evaluate the source");
    check(derivative.formula().find("PartialExpr") != std::string_view::npos,
          "formula must expose the mathematical derivative type");
    check(derivative.formula().find("StencilSumExpr") == std::string_view::npos,
          "formula must not contain sampling subtrees");

    coord<mu> x;
    x.data = {0.5, 0.75, 1.0, 1.25};
    const double h = 0.125;
    const auto value = derivative.eval(x, h);
    check(derivative_calls == 4 * Points, "derivative must evaluate exactly the expanded sample count");
    for (int axis = 0; axis < 4; ++axis) {
        for (int component = 0; component < 4; ++component) {
            const double expected = axis == component ? Power * std::pow(x(axis), Power - 1) : 0.0;
            check(close(value(axis, component), expected), "compile-time stencil polynomial derivative mismatch");
        }
        for (int point = 0; point < Points; ++point) {
            const int distance = point < Points / 2 ? point - Points / 2 : point - Points / 2 + 1;
            auto expected = x.data;
            expected[axis] += distance * h;
            check(derivative_samples[axis * Points + point] == expected,
                  "sampling subtree must use its compile-time offset from the original root");
        }
    }
}

void check_derivatives() {
    check_derivative_scheme<DiffScheme::Central2, 2, 2>();
    check_derivative_scheme<DiffScheme::Central4, 4, 4>();
    check_derivative_scheme<DiffScheme::Central6, 6, 6>();
    check_derivative_scheme<DiffScheme::Central8, 8, 8>();

    const auto field = make_field<&polynomial_field>();
    const auto first = partial<mu, DiffScheme::Central2>(field);
    const auto second = partial_field<nu, DiffScheme::Central2>(first);
    const auto shifted = expand_at<offset<1, -1, 0, 0>>(second);
    const auto reset = expand_at<>(shifted);
    using tree = typename decltype(second)::node_type;
    static_assert(list_size<typename tree::children>::value == 1);
    using sampled_first = typename expression_detail::partial_samples<FieldExpr<&polynomial_field>,
                                                                      DiffScheme::Central2>::result;
    using sampled_second = typename expression_detail::partial_samples<sampled_first, DiffScheme::Central2>::result;
    using outer0 = typename nth_type<0, typename sampled_second::children>::result;
    using negative = typename nth_type<0, typename outer0::children>::result;
    using positive = typename nth_type<1, typename outer0::children>::result;
    using negative0 = typename nth_type<0, typename negative::children>::result;
    using positive0 = typename nth_type<0, typename positive::children>::result;
    using minus_plus = typename nth_type<1, typename negative0::children>::result;
    using plus_minus = typename nth_type<0, typename positive0::children>::result;
    using plus_plus = typename nth_type<1, typename positive0::children>::result;
    static_assert(std::is_same_v<minus_plus, plus_minus>);
    static_assert(std::is_same_v<typename plus_minus::point_offset, offset<>>);
    static_assert(std::is_same_v<typename plus_plus::point_offset, offset<2, 0, 0, 0>>);
    static_assert(std::is_same_v<typename decltype(reset)::node_type, tree>);

    coord<mu> x;
    x.data = {0.5, 0.75, 1.0, 1.25};
    const double h = 0.125;
    derivative_calls = 0;
    const auto value = second.eval(x, h);
    check(derivative_calls == 64, "nested Central2 must retain all 64 sampling paths without a center probe");
    const auto shifted_value = shifted.eval(x, h);
    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            auto expected_hessian = [&](double p0, double p1, double p2) {
                if (a == 0 && b == 0) return 12 * p0 * p0 + 8 * h * h;
                if ((a == 0 && b == 1) || (a == 1 && b == 0)) return p2;
                if ((a == 0 && b == 2) || (a == 2 && b == 0)) return p1;
                if ((a == 1 && b == 2) || (a == 2 && b == 1)) return p0;
                return 0.0;
            };
            check(close(value(a, b), expected_hessian(x(0), x(1), x(2))),
                  "nested differentiation must preserve composition of first-difference operators");
            check(close(shifted_value(a, b), expected_hessian(x(0) + h, x(1) - h, x(2))),
                  "moving a derivative tree must retain inner and outer stencil displacements");
        }
    }

    const auto product = make_field<&scalar_field>() * make_field<&coordinate_field>();
    const auto product_derivative = partial_field<nu, DiffScheme::Central4>(product).eval(x, h);
    for (int a = 0; a < 4; ++a) {
        for (int j = 0; j < 4; ++j) {
            const double expected = (a == 1 ? x(j) : 0.0) + (a == j ? 2.0 + x(1) : 0.0);
            check(close(product_derivative(a, j), expected), "each sampling branch must contain the full product");
        }
    }

    const auto opaque = make_field([](const coord<mu>& point) { return polynomial_field(point); });
    derivative_calls = 0;
    const auto legacy_value = partial<mu, DiffScheme::Central4>(opaque, x, h);
    check(derivative_calls == 16, "opaque callable derivative must not add a result-type probe");
    const auto wrapped = partial_field<mu, DiffScheme::Central4>(opaque, h).eval(x);
    const auto typed = partial<mu, DiffScheme::Central4>(field, x, h);
    const auto shifted_typed = partial<mu, DiffScheme::Central4>(
        expand_at<offset<1, -1, 0, 0>>(field), x, h);
    const auto raised = partial_up<nu, mu, DiffScheme::Central4>(field, make_field<&identity_inverse>(), x, h);
    const double expected_gradient[] = {4 * std::pow(x(0), 3) + x(1) * x(2), x(0) * x(2), x(0) * x(1), 0};
    const double shifted_gradient[] = {4 * std::pow(x(0) + h, 3) + (x(1) - h) * x(2),
                                       (x(0) + h) * x(2), (x(0) + h) * (x(1) - h), 0};
    for (int axis = 0; axis < 4; ++axis) {
        check(close(legacy_value(axis), expected_gradient[axis]), "opaque compile-time strategy mismatch");
        check(close(wrapped(axis), expected_gradient[axis]), "Field wrapper must forward the compile-time strategy");
        check(close(typed(axis), expected_gradient[axis]), "direct typed partial must evaluate its sampling tree");
        check(close(shifted_typed(axis), shifted_gradient[axis]), "direct typed partial must forward h to offsets");
        check(close(raised(axis), expected_gradient[axis]), "partial_up must forward the compile-time strategy");
    }

    // A derivative inside an ordinary operation must also expand at every outer sample.
    const auto gradient_product = make_field<&scalar_field>() * partial_field<mu, DiffScheme::Central4>(field);
    const auto mixed = partial_field<nu, DiffScheme::Central2>(gradient_product).eval(x, h);
    auto analytic_gradient = [](const auto& p, int axis) {
        if (axis == 0) return 4 * std::pow(p(0), 3) + p(1) * p(2);
        if (axis == 1) return p(0) * p(2);
        if (axis == 2) return p(0) * p(1);
        return 0.0;
    };
    for (int a = 0; a < 4; ++a) {
        auto plus = x;
        auto minus = x;
        plus(a) += h;
        minus(a) -= h;
        for (int b = 0; b < 4; ++b) {
            const double expected = ((2 + plus(1)) * analytic_gradient(plus, b) -
                                     (2 + minus(1)) * analytic_gradient(minus, b)) / (2 * h);
            check(close(mixed(a, b), expected), "derivative of a product containing a derivative must expand recursively");
        }
    }
}

void check_mathematical_operations() {
    const auto a = make_field<&coordinate_field>();
    const auto b = rename_field<up<::b>>(make_field<&monomial_field<2>>());
    const auto c = rename_field<up<::c>>(make_field<&monomial_field<3>>());
    const auto left = a * b * c;
    const auto right = c * a * b;
    const auto expression = (left + right) - scale_field<1, 2>(right);
    static_assert(std::is_same_v<typename decltype(expression)::formula_type,
                                 std::remove_cv_t<decltype(expression)>>);
    coord<mu> x;
    x.data = {0.5, 0.75, 1.0, 1.25};
    const auto result = expression.eval(x);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                check(close(result(i, j, k), 1.5 * x(i) * std::pow(x(j), 2) * std::pow(x(k), 3)),
                      "mathematical addition/subtraction must respect index permutations");
            }
        }
    }
    const auto g = make_field<&metric>();
    const int before = metric_calls;
    const auto inverse = inverse_field(g);
    check(metric_calls == before, "inverse expression construction must not sample its operand");
    const auto inverse_value = inverse.eval(x);
    const auto expected_inverse = inverse_metric(x);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            check(close(inverse_value(i, j), expected_inverse(i, j)), "inverse node must match analytic dense inverse");
        }
    }
    const auto identity = g * rename_field<up<::b>, up<::c>>(inverse);
    check(close(trace_field<::a, ::c>(identity).eval(x)(), 4.0), "trace must compose with mathematical nodes");
    const auto derivative = partial_field<nu, DiffScheme::Central4>(scale_field<1, 2>(identity + identity));
    const auto differentiated = derivative.eval(x, 0.125);
    for (auto value : differentiated.data) {
        check(std::abs(value) < 1e-11, "derivatives must compose with add, scale and inverse nodes");
    }
}

} // namespace

int main() {
    const auto g = make_field<&metric>();
    const auto inv = make_field<&inverse_metric>();
    const auto inv_bc = rename_field<up<b>, up<c>>(inv);
    const auto g_cd = rename_field<dn<c>, dn<d>>(g);
    const auto inv_dmu = rename_field<up<d>, up<mu>>(inv);
    const auto first = g * inv_bc;
    const auto triple = first * g_cd;
    const auto chain = triple * inv_dmu;
    const auto second = g_cd * inv_dmu;
    const auto balanced = first * second;
    const auto scalar = make_field<&scalar_field>();
    const auto scaled = scalar * g;
    const auto scaled_right = g * scalar;
    const auto squared_scalar = scalar * scalar;
    const auto outer = g * g_cd;
    const auto inverse_outer = inv * rename_field<up<c>, up<d>>(inv);
    const auto fully_contracted = outer * inverse_outer;

    using G = FieldExpr<&metric>;
    using I = FieldExpr<&inverse_metric>;
    using P = contraction_plan<2, 2, contractpair<1, 0>>;
    using GI = ContractExpr<G, I, P>;
    static_assert(std::is_same_v<typename decltype(first)::node_type, GI>);
    static_assert(std::is_same_v<typename decltype(second)::node_type, GI>);
    static_assert(std::is_same_v<typename decltype(g_cd)::node_type, G>);
    static_assert(std::is_same_v<typename decltype(chain)::node_type, ContractExpr<ContractExpr<GI, G, P>, I, P>>);
    static_assert(std::is_same_v<typename decltype(balanced)::node_type, ContractExpr<GI, GI, P>>);
    static_assert(std::is_same_v<typename decltype(chain)::result_type, Tensor<double, dn<a>, up<mu>>>);
    static_assert(std::is_same_v<typename decltype(triple)::result_type, Tensor<double, dn<a>, dn<d>>>);
    static_assert(std::is_same_v<typename decltype(outer)::node_type::Plan, contraction_plan<2, 2>>);
    static_assert(std::is_same_v<typename decltype(fully_contracted)::node_type::Plan,
                                 contraction_plan<4, 4, contractpair<0, 0>, contractpair<1, 1>, contractpair<2, 2>,
                                                  contractpair<3, 3>>>);
    static_assert(std::is_same_v<typename decltype(fully_contracted)::result_type, Tensor<double>>);
    static_assert(std::is_same_v<typename decltype(scaled)::node_type::Plan, contraction_plan<0, 2>>);
    static_assert(std::is_same_v<typename decltype(scaled_right)::node_type::Plan, contraction_plan<2, 0>>);
    check(metric_calls == 0 && inverse_calls == 0, "constructing the tree must not evaluate fields");

    for (double radius : {0.5, 3.0, 10.0}) {
        Tensor<double, up<mu>> x;
        x(1) = radius;
        const auto result = chain.eval(x);
        const auto balanced_result = balanced(x);
        const auto triple_result = triple.eval(x);
        const auto metric_value = metric(x);
        const auto inverse_value = inverse_metric(x);
        const auto outer_result = outer.eval(x);
        const auto scaled_result = scaled.eval(x);
        const auto scaled_right_result = scaled_right.eval(x);
        check(close(squared_scalar.eval(x)(), (2.0 + radius) * (2.0 + radius)), "scalar * scalar mismatch");
        check(close(fully_contracted.eval(x)(), 16.0), "rank-four full contraction mismatch");
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                check(close(result(i, j), i == j ? 1.0 : 0.0), "metric chain must equal identity");
                check(close(balanced_result(i, j), result(i, j)), "balanced tree mismatch");
                check(close(triple_result(i, j), metric_value(i, j)), "g * inverse * g must equal g");
                check(close(scaled_result(i, j), (2.0 + radius) * metric_value(i, j)), "scalar * tensor mismatch");
                check(close(scaled_right_result(i, j), scaled_result(i, j)), "tensor * scalar mismatch");
                double independent = 0.0;
                for (int k = 0; k < 4; ++k) {
                    for (int l = 0; l < 4; ++l) {
                        for (int m = 0; m < 4; ++m) {
                            independent += metric_value(i, k) * inverse_value(k, l) * metric_value(l, m) *
                                           inverse_value(m, j);
                        }
                        check(close(outer_result(i, j, k, l), metric_value(i, j) * metric_value(k, l)),
                              "outer product mismatch");
                    }
                }
                check(close(result(i, j), independent), "chain differs from independent component loops");
            }
        }
        metric_calls = inverse_calls = 0;
        (void)balanced.eval(x);
        check(metric_calls == 2 && inverse_calls == 2, "tree evaluation must visit both repeated subtrees");
    }
    check_offsets();
    check_derivatives();
    check_mathematical_operations();
    if (failures) {
        return 1;
    }
    std::cout
        << "expression types, offsets, compile-time derivative trees, metric chains and contractions passed\n";
}
