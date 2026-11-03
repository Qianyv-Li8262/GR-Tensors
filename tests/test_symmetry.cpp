#include "tensor_expression.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>

namespace {

int failures = 0;
int calls = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << message << '\n';
    }
}

bool close(double a, double b) {
    return std::isfinite(a) && std::abs(a - b) < 2e-11 * std::max(1.0, std::abs(b));
}

double coefficient(std::size_t flat) {
    return (static_cast<int>((flat * flat + 17 * flat) % 101) - 50) * 0.125;
}

auto source(const coord<mu>& x) {
    ++calls;
    Tensor<double, dn<a>, dn<b>, dn<c>, dn<d>> tensor;
    for (std::size_t flat = 0; flat < tensor.data.size(); ++flat) {
        tensor.data[flat] = coefficient(flat) * (1 + x(0) * x(1)) +
                            (static_cast<int>(flat % 7) - 3) * x(2);
    }
    return tensor;
}

// Independent reference: explicitly enumerate every permutation, including
// repeated coordinates, and divide by k!. No orbit/plan helpers are used here.
template <std::size_t Rank, std::size_t Count>
auto reference(const std::array<double, pow4(Rank)>& input,
               const std::array<std::size_t, Count>& axes, bool alternating) {
    std::array<double, pow4(Rank)> result{};
    for (std::size_t flat = 0; flat < input.size(); ++flat) {
        std::array<std::size_t, Rank> index{};
        auto remaining = flat;
        for (std::size_t axis = Rank; axis > 0; --axis) {
            index[axis - 1] = remaining % 4;
            remaining /= 4;
        }
        std::array<std::size_t, Count> permutation{};
        for (std::size_t i = 0; i < Count; ++i) permutation[i] = i;
        std::size_t count = 0;
        do {
            auto sample = index;
            int sign = 1;
            for (std::size_t i = 0; i < Count; ++i) {
                sample[axes[i]] = index[axes[permutation[i]]];
                for (std::size_t j = i + 1; j < Count; ++j) {
                    if (permutation[i] > permutation[j]) sign = -sign;
                }
            }
            std::size_t offset = 0;
            for (auto component : sample) offset = 4 * offset + component;
            result[flat] += (alternating ? sign : 1) * input[offset];
            ++count;
        } while (std::next_permutation(permutation.begin(), permutation.end()));
        result[flat] /= count;
    }
    return result;
}

template <std::size_t Size>
void equal(const std::array<double, Size>& actual, const std::array<double, Size>& expected,
           const char* message) {
    for (std::size_t i = 0; i < Size; ++i) {
        if (!close(actual[i], expected[i])) {
            check(false, message);
            return;
        }
    }
}

void check_components() {
    coord<mu> x;
    x.data = {0.5, 0.75, -0.25, 1};
    const auto tensor = source(x);
    using SP = sym_plan<4, 2, 0, 1>;
    using AP = asym_plan<4, 2, 0, 1>;
    static_assert(std::is_same_v<SP::positions, std::index_sequence<2, 0, 1>>);
    static_assert(std::is_same_v<sym_analysis<type_list<dn<a>, dn<b>, dn<c>, dn<d>>, 2, 0, 1>::plan, SP>);
    static_assert(std::is_same_v<asym_analysis<type_list<dn<a>, dn<b>, dn<c>, dn<d>>, 2, 0, 1>::plan, AP>);
    const auto expected_sym = reference<4>(tensor.data, std::array<std::size_t, 3>{2, 0, 1}, false);
    const auto expected_asym = reference<4>(tensor.data, std::array<std::size_t, 3>{2, 0, 1}, true);
    const auto symmetric = sym_components<SP>(tensor.data);
    const auto alternating = asym_components<AP>(tensor.data);
    equal(symmetric, expected_sym, "orbit symmetrization differs from full permutation sum");
    equal(alternating, expected_asym, "orbit antisymmetrization differs from signed permutation sum");
    equal(sym<2, 0, 1>(tensor).data, expected_sym, "positional Tensor symmetry interface");
    equal(asym<2, 0, 1>(tensor).data, expected_asym, "positional Tensor antisymmetry interface");
    equal(symm_orbit<dn<c>, dn<a>, dn<b>>(tensor).data, expected_sym, "legacy named symmetry interface");
    equal(asymm_orbit<dn<c>, dn<a>, dn<b>>(tensor).data, expected_asym, "legacy named antisymmetry interface");
    equal(sym_components<SP>(symmetric), symmetric, "symmetry projection must be idempotent");
    equal(asym_components<AP>(alternating), alternating, "antisymmetry projection must be idempotent");
    equal(sym<0, 1, 2>(tensor).data, symmetric, "axis list order must not change symmetrization");
    equal(asym<0, 1, 2>(tensor).data, alternating, "axis list order must not change antisymmetrization");
    const std::array<double, 256> zero{};
    equal(sym_components<SP>(alternating), zero, "symmetry must annihilate an alternating tensor on the same axes");
    equal(asym_components<AP>(symmetric), zero, "antisymmetry must annihilate a symmetric tensor on the same axes");
    equal(asym<0, 1, 2, 3>(tensor).data,
          reference<4>(tensor.data, std::array<std::size_t, 4>{0, 1, 2, 3}, true),
          "four-axis alternating normalization");

    Tensor<double, dn<a>, up<b>, dn<c>> mixed;
    for (std::size_t flat = 0; flat < mixed.data.size(); ++flat) mixed.data[flat] = coefficient(flat);
    equal(sym<0, 2>(mixed).data, reference<3>(mixed.data, std::array<std::size_t, 2>{0, 2}, false),
          "unselected opposite-variance axes must be allowed and stay fixed");

    // Six selected axes in four dimensions have many repeated coordinates.
    // This also exceeds the old per-component template-recursion depth.
    std::array<double, 4096> rank6{};
    for (std::size_t flat = 0; flat < rank6.size(); ++flat) rank6[flat] = coefficient(flat);
    equal(sym_components<sym_plan<6, 0, 1, 2, 3, 4, 5>>(rank6),
          reference<6>(rank6, std::array<std::size_t, 6>{0, 1, 2, 3, 4, 5}, false),
          "six-axis symmetry must normalize repeated-coordinate orbits correctly");
    equal(asym_components<asym_plan<6, 0, 1, 2, 3, 4>>(rank6), std::array<double, 4096>{},
          "alternating five axes in four dimensions must vanish");
}

void check_expressions() {
    calls = 0;
    const auto field = make_field<&source>();
    const auto expression = sym<0, 2>(asym<1, 2>(field));
    const auto reverse = asym<1, 2>(sym<0, 2>(field));
    using F = FieldExpr<&source>;
    using Inner = AsymExpr<F, asym_plan<4, 1, 2>>;
    using Outer = SymExpr<Inner, sym_plan<4, 0, 2>>;
    static_assert(std::is_same_v<typename decltype(expression)::node_type, Outer>);
    static_assert(std::is_same_v<typename Outer::children, type_list<Inner>>);
    static_assert(std::is_same_v<typename decltype(expression)::result_type,
                                 Tensor<double, dn<a>, dn<b>, dn<c>, dn<d>>>);
    check(calls == 0, "constructing symmetry expressions must not evaluate their operand");
    check(expression.formula().find("SymExpr") != std::string_view::npos &&
          expression.formula().find("AsymExpr") != std::string_view::npos &&
          expression.formula().find("sym_plan<4, 0, 2>") != std::string_view::npos,
          "formula must retain symmetry nodes and positional plans");
    check(expression.formula().find("AddExpr") == std::string_view::npos,
          "symmetry must not expand permutations into mathematical addition subtrees");
    const auto renamed = rename_field<dn<mu>, dn<nu>, dn<rho>, dn<sigma>>(expression);
    static_assert(std::is_same_v<typename decltype(renamed)::node_type, Outer>);

    coord<mu> x;
    x.data = {0.5, 0.75, -0.25, 1};
    const auto input = source(x);
    const auto expected = reference<4>(reference<4>(input.data, std::array<std::size_t, 2>{1, 2}, true),
                                       std::array<std::size_t, 2>{0, 2}, false);
    const auto expected_reverse = reference<4>(reference<4>(input.data, std::array<std::size_t, 2>{0, 2}, false),
                                               std::array<std::size_t, 2>{1, 2}, true);
    calls = 0;
    const auto result = expression.eval(x);
    check(calls == 1, "nested symmetry must evaluate the leaf once, then project its values");
    equal(result.data, expected, "nested symmetry must preserve operation order");
    equal(reverse.eval(x).data, expected_reverse, "reversed projections must preserve their own order");
    bool different = false;
    for (std::size_t flat = 0; flat < expected.size(); ++flat) {
        different = different || !close(expected[flat], expected_reverse[flat]);
    }
    check(different, "overlapping symmetric/antisymmetric projections generally do not commute");
    equal(sym<0, 1, 2>(asym<1, 2, 3>(field)).eval(x).data, std::array<double, 256>{},
          "symmetrization and antisymmetrization sharing two axes must annihilate each other");

    using O = offset<1, -2, 3, -1>;
    const double h = 0.125;
    const auto shifted = expand_at<O>(expression);
    const auto direct_shifted = sym<0, 2>(asym<1, 2>(expand_at<O>(field)));
    using ShiftedInner = AsymExpr<FieldExpr<&source, O>, asym_plan<4, 1, 2>, O>;
    static_assert(std::is_same_v<typename decltype(shifted)::node_type,
                                 SymExpr<ShiftedInner, sym_plan<4, 0, 2>, O>>);
    static_assert(std::is_same_v<decltype(shifted), decltype(direct_shifted)>);
    static_assert(std::is_same_v<typename decltype(expand_at<>(shifted))::node_type, Outer>);
    auto sample = x;
    for (std::size_t axis = 0; axis < 4; ++axis) sample(axis) += h * O::values[axis];
    const auto shifted_input = source(sample);
    equal(shifted.eval(x, h).data,
          reference<4>(reference<4>(shifted_input.data, std::array<std::size_t, 2>{1, 2}, true),
                       std::array<std::size_t, 2>{0, 2}, false),
          "symmetry nodes must pass offsets to the leaf exactly once");

    const auto derivative = partial_field<nu, DiffScheme::Central2>(shifted);
    calls = 0;
    const auto differentiated = derivative.eval(x, h);
    check(calls == 8, "derivative of symmetry must evaluate one leaf per stencil sample");
    for (int axis = 0; axis < 4; ++axis) {
        std::array<double, 256> analytic{};
        for (std::size_t flat = 0; flat < analytic.size(); ++flat) {
            analytic[flat] = coefficient(flat) * (axis == 0 ? sample(1) : axis == 1 ? sample(0) : 0) +
                             (axis == 2 ? static_cast<int>(flat % 7) - 3 : 0);
        }
        const auto projected = reference<4>(reference<4>(analytic, std::array<std::size_t, 2>{1, 2}, true),
                                            std::array<std::size_t, 2>{0, 2}, false);
        for (std::size_t flat = 0; flat < projected.size(); ++flat) {
            check(close(differentiated.data[axis * 256 + flat], projected[flat]),
                  "partial must compose with nested symmetry and shifted offsets");
        }
    }
    equal(sym<0, 2>(partial_field<nu, DiffScheme::Central2>(field)).eval(x, h).data,
          reference<5>(partial_field<nu, DiffScheme::Central2>(field).eval(x, h).data,
                       std::array<std::size_t, 2>{0, 2}, false),
          "symmetry may include a derivative axis of its operand");
}

struct e {}; struct f {}; struct g {}; struct h {}; struct i {}; struct j {};
auto high_rank_source(const coord<mu>&) {
    return Tensor<double, dn<a>, dn<b>, dn<c>, dn<d>, dn<e>, dn<f>, dn<g>, dn<h>, dn<i>, dn<j>>{};
}

void check_compact_high_rank_type() {
    const auto A = make_field<&high_rank_source>();
    const auto projected = sym<1, 2, 4, 5, 7, 8>(asym<4, 5, 6>(A));
    using Expected = SymExpr<AsymExpr<FieldExpr<&high_rank_source>, asym_plan<10, 4, 5, 6>>,
                             sym_plan<10, 1, 2, 4, 5, 7, 8>>;
    static_assert(std::is_same_v<typename decltype(projected)::node_type, Expected>);
    static_assert(list_size<typename Expected::children>::value == 1);
    check(projected.formula().size() < 2000, "high-rank projection type must retain two compact nodes");
}

} // namespace

int main() {
    check_components();
    check_expressions();
    check_compact_high_rank_type();
    if (failures) return 1;
    std::cout << "sym/asym plans, permutation references, nesting, offsets, derivatives and compact rank-ten types passed\n";
}
