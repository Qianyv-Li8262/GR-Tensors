#include "tensor_expression.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>

namespace {
int failures = 0;
int calls = 0;
void check(bool condition, const char* message) {
    if (!condition) { ++failures; std::cerr << message << '\n'; }
}

// Independent input-to-output reference; the executor uses output-to-input.
template <std::size_t... Axes>
void check_mapping(index<Axes...>) {
    constexpr std::size_t rank = sizeof...(Axes);
    constexpr std::array<std::size_t, rank> axes{Axes...};
    std::array<int, pow4(rank)> input{}, expected{};
    for (std::size_t flat = 0; flat < input.size(); ++flat) input[flat] = 13 * static_cast<int>(flat) - 7;
    for (std::size_t flat = 0; flat < input.size(); ++flat) {
        std::array<std::size_t, rank> coordinates{};
        auto remainder = flat;
        for (std::size_t axis = rank; axis > 0; --axis) {
            coordinates[axis - 1] = remainder % 4;
            remainder /= 4;
        }
        std::size_t destination = 0;
        for (auto axis : axes) destination = 4 * destination + coordinates[axis];
        expected[destination] = input[flat];
    }
    check(permute_components<index<Axes...>>(input) == expected, "permutation differs from independent reference");
    check(input[0] == -7, "permutation must not mutate its input");
}

auto field(const coord<mu>& x) {
    ++calls;
    Tensor<double, up<a>, dn<b>, up<c>> result;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                result(i, j, k) = 100 * i + 10 * j + k + (i + 1) * x(0) + (j + 1) * x(2);
    return result;
}

void check_expressions() {
    static_assert(index<>::is_identity && index<0>::is_identity && index<0, 1, 2>::is_identity);
    static_assert(!index<1, 0>::is_identity && !index<0, 2, 1>::is_identity);
    const auto A = make_field<&field>();
    const auto B = permute(A, index<2, 0, 1>{});
    using Expected = PermuteExpr<FieldExpr<&field>, index<2, 0, 1>>;
    static_assert(std::is_same_v<typename decltype(B)::node_type, Expected>);
    static_assert(std::is_same_v<typename Expected::children, type_list<FieldExpr<&field>>>);
    static_assert(std::is_same_v<typename decltype(B)::result_type, Tensor<double, up<c>, up<a>, dn<b>>>);
    static_assert(std::is_same_v<std::remove_cv_t<decltype(B)>, decltype(permute<2, 0, 1>(A))>);
    check(calls == 0, "permutation construction must not evaluate its operand");
    check(B.formula().find("PermuteExpr") != std::string_view::npos, "formula must retain the permutation node");
    coord<mu> x;
    x.data = {0.5, -0.25, 0.75, 1};
    const auto original = field(x);
    const auto identity = permute(A, index<0, 1, 2>{});
    static_assert(std::is_same_v<typename decltype(identity)::node_type,
                                 PermuteExpr<FieldExpr<&field>, index<0, 1, 2>>>);
    calls = 0;
    check(identity.eval(x).data == original.data, "identity expression must forward the original components");
    check(calls == 1, "identity expression must evaluate its operand once");
    calls = 0;
    const auto values = B.eval(x);
    check(calls == 1, "permutation must evaluate its operand once");
    const auto direct = permute(original, index<2, 0, 1>{});
    const auto restored = permute<1, 2, 0>(B).eval(x);
    static_assert(std::is_same_v<decltype(direct), const Tensor<double, up<c>, up<a>, dn<b>>>);
    check(restored.data == original.data, "inverse permutation must restore the original tensor");
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                check(values(k, i, j) == original(i, j, k) && direct(k, i, j) == original(i, j, k),
                      "axis direction or index variance was reordered incorrectly");

    using O = offset<1, -2, 3, -1>;
    const auto shifted = expand_at<O>(B);
    static_assert(std::is_same_v<typename decltype(shifted)::node_type,
                                 PermuteExpr<FieldExpr<&field, O>, index<2, 0, 1>, O>>);
    const double h = 0.125;
    auto sample = x;
    for (std::size_t axis = 0; axis < 4; ++axis) sample(axis) += h * O::values[axis];
    check(shifted.eval(x, h).data == permute<2, 0, 1>(field(sample)).data,
          "permutation must propagate offsets exactly once");
    check(expand_at<>(shifted).eval(x).data == values.data, "offset reset through permutation");
    calls = 0;
    const auto derivative = partial_field<nu, DiffScheme::Central2>(shifted).eval(x, h);
    check(calls == 8, "partial of permutation must evaluate one operand per sample");
    for (int axis = 0; axis < 4; ++axis)
        for (int k = 0; k < 4; ++k)
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    check(std::abs(derivative(axis, k, i, j) - (axis == 0 ? i + 1 : axis == 2 ? j + 1 : 0)) < 1e-12,
                          "partial must compose with permutation");
    Tensor<double, dn<a>, up<b>> matrix;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) matrix(i, j) = 10 * i + j;
    const auto transpose = permute(matrix, index<1, 0>{});
    static_assert(std::is_same_v<decltype(transpose), const Tensor<double, up<b>, dn<a>>>);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) check(transpose(j, i) == matrix(i, j), "index<1,0> must transpose");
    check(permute(Tensor<double>(3.5), index<>{})() == 3.5, "scalar requires an explicit empty mapping");
}
} // namespace

int main() {
    check_mapping(index<>{});
    check_mapping(index<0>{});
    check_mapping(index<1, 0>{});
    check_mapping(index<0, 1, 2>{});
    check_mapping(index<2, 0, 1>{});
    check_mapping(index<1, 2, 0>{});
    check_mapping(index<3, 1, 0, 2>{});
    check_mapping(index<5, 3, 1, 4, 2, 0>{});
    check_expressions();
    if (failures) return 1;
    std::cout << "complete permutations, transpose, cycles, rank-six copies, expression types, offsets and partials passed\n";
}
