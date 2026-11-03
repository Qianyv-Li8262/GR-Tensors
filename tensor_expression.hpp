#pragma once

#include "contraction_and_arithmetic.hpp"
#include "num_deriv.hpp"
#include <algorithm>
#include <array>
#include <ratio>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

// Integer multiples of the runtime step h, relative to the same root coordinate.
template <int O0 = 0, int O1 = 0, int O2 = 0, int O3 = 0>
struct offset {
    inline static constexpr std::array<int, 4> values{O0, O1, O2, O3};
};

template <typename Node, typename PointOffset = offset<>>
struct expand;

namespace expression_detail {

template <typename T>
struct tensor_traits;

template <typename Scalar, typename... Indices>
struct tensor_traits<Tensor<Scalar, Indices...>> {
    using scalar_type = Scalar;
    using indices = type_list<Indices...>;
    static constexpr std::size_t rank = sizeof...(Indices);
};

template <typename Function>
struct function_result;

template <typename Result, typename... Args>
struct function_result<Result (*)(Args...)> {
    using type = Result;
};

template <typename Result, typename... Args>
struct function_result<Result (*)(Args...) noexcept> : function_result<Result (*)(Args...)> {};

template <typename Node, typename Indices>
struct indexed_expression;

} // namespace expression_detail

// Function identity is a non-type template argument, not a runtime pointer member.
// Functions are opaque leaves: their bodies are not expanded into this tree.
template <auto Function, typename PointOffset = offset<>>
struct FieldExpr {
    using point_offset = PointOffset;
    using source_type = std::decay_t<typename expression_detail::function_result<decltype(Function)>::type>;
    using scalar_type = typename expression_detail::tensor_traits<source_type>::scalar_type;
    using children = type_list<>;
    static constexpr std::size_t rank = expression_detail::tensor_traits<source_type>::rank;

    template <typename Coordinate>
    auto eval_components(const Coordinate& x, double h = 1e-2) const {
        if constexpr (std::is_same_v<point_offset, offset<>>) {
            return Function(x).data;
        } else {
            auto sample = x;
            for (std::size_t axis = 0; axis < 4; ++axis) {
                if (point_offset::values[axis] != 0) {
                    sample(axis) += h * point_offset::values[axis];
                }
            }
            return Function(sample).data;
        }
    }
};

// The computation tree contains operand nodes, positional plans and point offsets.
// Index letters live in IndexedExpr, outside this type's template arguments.
template <typename L, typename R, typename ContractionPlan, typename PointOffset = offset<>>
struct ContractExpr {
    using point_offset = PointOffset;
    using left_type = L;
    using right_type = R;
    using children = type_list<L, R>;
    using Plan = ContractionPlan;
    using scalar_type = typename L::scalar_type;
    static_assert(std::is_same_v<scalar_type, typename R::scalar_type>,
                  "contraction expression operands must have the same scalar type");
    static_assert(L::rank == Plan::left_rank && R::rank == Plan::right_rank,
                  "contraction expression operand ranks must match the plan");
    static_assert(std::is_same_v<typename L::point_offset, point_offset> &&
                      std::is_same_v<typename R::point_offset, point_offset>,
                  "contraction operands must share the node offset; use expand_at to move the whole tree");
    static constexpr std::size_t rank = Plan::result_rank;

    L left;
    R right;

    template <typename Coordinate>
    auto eval_components(const Coordinate& x, double h = 1e-2) const {
        // Deliberately recursive tree evaluation; repeated subtrees are evaluated again.
        const auto left_values = left.eval_components(x, h);
        const auto right_values = right.eval_components(x, h);
        return contract_components<Plan>(left_values, right_values);
    }
};

template <typename L, typename R, typename Plan, typename PointOffset = offset<>>
struct AddExpr;

template <typename L, typename R, int... Positions, typename PointOffset>
struct AddExpr<L, R, add_plan<Positions...>, PointOffset> {
    using point_offset = PointOffset;
    using scalar_type = typename L::scalar_type;
    using children = type_list<L, R>;
    using Plan = add_plan<Positions...>;
    static constexpr std::size_t rank = L::rank;
    static_assert(rank == R::rank && rank == sizeof...(Positions), "addition expression ranks must match");
    static_assert(std::is_same_v<scalar_type, typename R::scalar_type>, "addition scalar types must match");
    static_assert(std::is_same_v<typename L::point_offset, point_offset> &&
                      std::is_same_v<typename R::point_offset, point_offset>, "addition offsets must match");
    L left;
    R right;

    template <typename Coordinate>
    auto eval_components(const Coordinate& x, double h = 1e-2) const {
        auto result = left.eval_components(x, h);
        const auto rhs = right.eval_components(x, h);
        for (std::size_t flat = 0; flat < result.size(); ++flat) {
            const auto index = unpack_index<rank>(flat);
            const std::array<std::size_t, rank> right_index{index[Positions]...};
            result[flat] += rhs[pack_index(right_index)];
        }
        return result;
    }
};

// Rational constants belong to the type, including the 1/2 in Christoffel's formula.
template <typename Node, typename Ratio, typename PointOffset = offset<>>
struct ScaleExpr {
    using point_offset = PointOffset;
    using scalar_type = typename Node::scalar_type;
    using children = type_list<Node>;
    static constexpr std::size_t rank = Node::rank;
    static_assert(std::is_same_v<typename Node::point_offset, point_offset>, "scale offsets must match");
    Node operand;

    template <typename Coordinate>
    auto eval_components(const Coordinate& x, double h = 1e-2) const {
        auto result = operand.eval_components(x, h);
        for (auto& value : result) {
            value *= static_cast<double>(Ratio::num) / Ratio::den;
        }
        return result;
    }
};

template <typename Node, typename TracePlan, typename PointOffset = offset<>>
struct TraceExpr {
    using Plan = TracePlan;
    using point_offset = PointOffset;
    using scalar_type = typename Node::scalar_type;
    using children = type_list<Node>;
    static constexpr std::size_t rank = Plan::result_rank;
    static_assert(Node::rank == Plan::input_rank, "trace expression rank must match the plan");
    static_assert(std::is_same_v<typename Node::point_offset, point_offset>, "trace offsets must match");
    Node operand;

    template <typename Coordinate>
    auto eval_components(const Coordinate& x, double h = 1e-2) const {
        return trace_components<Plan>(operand.eval_components(x, h));
    }
};

template <typename Node, typename PointOffset = offset<>>
struct InverseExpr {
    using point_offset = PointOffset;
    using scalar_type = typename Node::scalar_type;
    using children = type_list<Node>;
    static constexpr std::size_t rank = 2;
    static_assert(Node::rank == 2, "matrix inverse requires a rank-two field");
    static_assert(std::is_same_v<typename Node::point_offset, point_offset>, "inverse offsets must match");
    Node operand;

    template <typename Coordinate>
    auto eval_components(const Coordinate& x, double h = 1e-2) const {
        const auto input = operand.eval_components(x, h);
        std::array<scalar_type, 32> rows{};
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                rows[8 * i + j] = input[4 * i + j];
                rows[8 * i + 4 + j] = i == j ? 1 : 0;
            }
        }
        for (std::size_t col = 0; col < 4; ++col) {
            std::size_t pivot = col;
            for (std::size_t row = col + 1; row < 4; ++row) {
                if (std::abs(rows[8 * row + col]) > std::abs(rows[8 * pivot + col])) pivot = row;
            }
            assert(rows[8 * pivot + col] != 0 && "metric must be invertible");
            for (std::size_t j = 0; j < 8; ++j) std::swap(rows[8 * col + j], rows[8 * pivot + j]);
            const auto divisor = rows[8 * col + col];
            for (std::size_t j = 0; j < 8; ++j) rows[8 * col + j] /= divisor;
            for (std::size_t row = 0; row < 4; ++row) {
                if (row == col) continue;
                const auto factor = rows[8 * row + col];
                for (std::size_t j = 0; j < 8; ++j) rows[8 * row + j] -= factor * rows[8 * col + j];
            }
        }
        std::array<scalar_type, 16> result{};
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) result[4 * i + j] = rows[8 * i + 4 + j];
        }
        return result;
    }
};

namespace expression_detail {

// Rebase an existing sample P from center From to center To.
template <typename P, typename From, typename To>
using rebase_offset = offset<P::values[0] - From::values[0] + To::values[0],
                             P::values[1] - From::values[1] + To::values[1],
                             P::values[2] - From::values[2] + To::values[2],
                             P::values[3] - From::values[3] + To::values[3]>;

template <typename O, std::size_t Axis, int Distance>
using sample_offset = offset<O::values[0] + (Axis == 0 ? Distance : 0),
                             O::values[1] + (Axis == 1 ? Distance : 0),
                             O::values[2] + (Axis == 2 ? Distance : 0),
                             O::values[3] + (Axis == 3 ? Distance : 0)>;

} // namespace expression_detail

// One direction: explicitly stored sampling subtrees, followed by a weighted sum.
template <DiffScheme Scheme, typename PointOffset, typename... Samples>
struct StencilSumExpr {
    using point_offset = PointOffset;
    using stencil = derivative_stencil<Scheme>;
    using children = type_list<Samples...>;
    using samples_type = std::tuple<Samples...>;
    using first_type = std::tuple_element_t<0, samples_type>;
    using scalar_type = typename first_type::scalar_type;
    static constexpr std::size_t rank = first_type::rank;
    static_assert(sizeof...(Samples) == stencil::offsets.size(), "stencil sample count must match the scheme");
    static_assert(((Samples::rank == rank && std::is_same_v<typename Samples::scalar_type, scalar_type>) && ...),
                  "stencil samples must have matching ranks and scalar types");

    samples_type samples;

    template <typename Coordinate, std::size_t... I>
    auto evaluate(const Coordinate& x, double h, std::index_sequence<I...>) const {
        std::array<scalar_type, pow4(rank)> result{};
        auto accumulate = [&](const auto& sample, auto position) {
            constexpr double weight = static_cast<double>(stencil::weights[decltype(position)::value]) /
                                      stencil::denominator;
            const auto values = sample.eval_components(x, h);
            for (std::size_t j = 0; j < result.size(); ++j) {
                result[j] += weight * values[j];
            }
        };
        (accumulate(std::get<I>(samples), std::integral_constant<std::size_t, I>{}), ...);
        for (auto& value : result) {
            value /= h;
        }
        return result;
    }

    template <typename Coordinate>
    auto eval_components(const Coordinate& x, double h = 1e-2) const {
        return evaluate(x, h, std::index_sequence_for<Samples...>{});
    }
};

// Four directional sums; the new derivative axis is the first result axis.
template <typename D0, typename D1, typename D2, typename D3, typename PointOffset = offset<>>
struct SampledPartialExpr {
    using point_offset = PointOffset;
    using children = type_list<D0, D1, D2, D3>;
    using directions_type = std::tuple<D0, D1, D2, D3>;
    using scalar_type = typename D0::scalar_type;
    static constexpr std::size_t rank = D0::rank + 1;

    directions_type directions;

    template <typename Coordinate, std::size_t... Axis>
    auto evaluate(const Coordinate& x, double h, std::index_sequence<Axis...>) const {
        std::array<scalar_type, pow4(rank)> result{};
        auto copy_direction = [&](auto axis) {
            constexpr std::size_t a = decltype(axis)::value;
            const auto values = std::get<a>(directions).eval_components(x, h);
            std::copy(values.begin(), values.end(), result.begin() + a * values.size());
        };
        (copy_direction(std::integral_constant<std::size_t, Axis>{}), ...);
        return result;
    }

    template <typename Coordinate>
    auto eval_components(const Coordinate& x, double h = 1e-2) const {
        derivative_detail::check_step(x, h);
        return evaluate(x, h, std::make_index_sequence<4>{});
    }
};

namespace expression_detail {
template <typename Node, DiffScheme Scheme>
struct partial_samples;
}

// Mathematical derivative: one operand, no stencil children until evaluation.
template <typename Node, DiffScheme Scheme = DiffScheme::Central8, typename PointOffset = offset<>>
struct PartialExpr {
    using point_offset = PointOffset;
    using children = type_list<Node>;
    using scalar_type = typename Node::scalar_type;
    static constexpr std::size_t rank = Node::rank + 1;
    static constexpr DiffScheme scheme = Scheme;
    static_assert(std::is_same_v<typename Node::point_offset, point_offset>, "derivative offsets must match");
    Node operand;

    template <typename Coordinate>
    auto eval_components(const Coordinate& x, double h = 1e-2) const {
        return expression_detail::partial_samples<Node, Scheme>::exec(operand).eval_components(x, h);
    }
};

template <typename Node, typename... Indices>
struct IndexedExpr {
    using node_type = Node;
    using formula_type = IndexedExpr;
    using point_offset = typename Node::point_offset;
    using scalar_type = typename Node::scalar_type;
    using indices = type_list<Indices...>;
    using result_type = Tensor<scalar_type, Indices...>;
    static constexpr std::size_t rank = Node::rank;
    static_assert(sizeof...(Indices) == rank, "expression indices must match the node rank");
    // Instantiating the tensor also checks valid indices and duplicate letters.
    static_assert(sizeof(result_type) > 0, "expression result must be a valid tensor");

    Node node;

    constexpr std::string_view formula() const {
#if defined(_MSC_VER) && !defined(__clang__)
        return __FUNCSIG__;
#elif defined(__clang__) || defined(__GNUC__)
        return __PRETTY_FUNCTION__;
#else
#error Formula type inspection requires a compiler function-signature macro.
#endif
    }

    template <typename Coordinate>
    result_type eval(const Coordinate& x, double h = 1e-2) const {
        result_type result;
        result.data = node.eval_components(x, h);
        return result;
    }

    template <typename Coordinate>
    result_type operator()(const Coordinate& x, double h = 1e-2) const {
        return eval(x, h);
    }
};

namespace expression_detail {

template <typename Node, typename... Indices>
struct indexed_expression<Node, type_list<Indices...>> {
    using type = IndexedExpr<Node, Indices...>;
};

} // namespace expression_detail

// Expand directly at an absolute offset. Ordinary operations pass that offset
// unchanged to their children; derivative sums preserve their stencil displacements.
template <auto Function, typename OldOffset, typename PointOffset>
struct expand<FieldExpr<Function, OldOffset>, PointOffset> {
    using result = FieldExpr<Function, PointOffset>;

    static result exec(const FieldExpr<Function, OldOffset>&) {
        return {};
    }
};

template <typename L, typename R, typename Plan, typename OldOffset, typename PointOffset>
struct expand<ContractExpr<L, R, Plan, OldOffset>, PointOffset> {
    using result = ContractExpr<typename expand<L, PointOffset>::result,
                                typename expand<R, PointOffset>::result, Plan, PointOffset>;

    static result exec(const ContractExpr<L, R, Plan, OldOffset>& node) {
        return {expand<L, PointOffset>::exec(node.left), expand<R, PointOffset>::exec(node.right)};
    }
};

template <typename Node, typename... Indices, typename PointOffset>
struct expand<IndexedExpr<Node, Indices...>, PointOffset> {
    using result = IndexedExpr<typename expand<Node, PointOffset>::result, Indices...>;

    static result exec(const IndexedExpr<Node, Indices...>& expression) {
        return {expand<Node, PointOffset>::exec(expression.node)};
    }
};

template <typename PointOffset = offset<>, typename Expression>
auto expand_at(const Expression& expression) {
    return expand<Expression, PointOffset>::exec(expression);
}

// Named function entry point, e.g. make_field<&metric>().
// The existing make_field(callable) API remains available for opaque lambda fields.
template <auto Function, typename PointOffset = offset<>>
auto make_field() {
    using node = FieldExpr<Function, PointOffset>;
    using indices = typename expression_detail::tensor_traits<typename node::source_type>::indices;
    using expression = typename expression_detail::indexed_expression<node, indices>::type;
    return expression{node{}};
}

template <typename L, typename... LeftIndices, typename R, typename... RightIndices>
auto operator*(const IndexedExpr<L, LeftIndices...>& left, const IndexedExpr<R, RightIndices...>& right) {
    using analysis = contraction_analysis<type_list<LeftIndices...>, type_list<RightIndices...>>;
    using node = ContractExpr<L, R, typename analysis::plan, typename L::point_offset>;
    using expression = typename expression_detail::indexed_expression<node, typename analysis::free_indices>::type;
    return expression{node{left.node, right.node}};
}

// A rename is an index view: it adds no numerical operation to the inner tree.
template <typename... NewIndices, typename Node, typename... OldIndices>
auto rename_field(const IndexedExpr<Node, OldIndices...>& field) {
    static_assert(sizeof...(NewIndices) == sizeof...(OldIndices), "rename requires same rank");
    static_assert((is_same_variance<OldIndices, NewIndices>::value && ...),
                  "rename may only change index letters, not raise/lower index positions");
    return IndexedExpr<Node, NewIndices...>{field.node};
}

// Keep each stencil displacement when a derivative subtree moves to a new center.
template <DiffScheme Scheme, typename OldOffset, typename... Samples, typename PointOffset>
struct expand<StencilSumExpr<Scheme, OldOffset, Samples...>, PointOffset> {
    template <typename Sample>
    using relocated = expand<Sample, expression_detail::rebase_offset<
        typename Sample::point_offset, OldOffset, PointOffset>>;
    using result = StencilSumExpr<Scheme, PointOffset, typename relocated<Samples>::result...>;

    template <std::size_t... I>
    static result apply(const StencilSumExpr<Scheme, OldOffset, Samples...>& node, std::index_sequence<I...>) {
        return {typename result::samples_type{relocated<Samples>::exec(std::get<I>(node.samples))...}};
    }
    static result exec(const StencilSumExpr<Scheme, OldOffset, Samples...>& node) {
        return apply(node, std::index_sequence_for<Samples...>{});
    }
};

template <typename D0, typename D1, typename D2, typename D3, typename OldOffset, typename PointOffset>
struct expand<SampledPartialExpr<D0, D1, D2, D3, OldOffset>, PointOffset> {
    using result = SampledPartialExpr<typename expand<D0, PointOffset>::result, typename expand<D1, PointOffset>::result,
                               typename expand<D2, PointOffset>::result, typename expand<D3, PointOffset>::result,
                               PointOffset>;
    static result exec(const SampledPartialExpr<D0, D1, D2, D3, OldOffset>& node) {
        return {typename result::directions_type{
            expand<D0, PointOffset>::exec(std::get<0>(node.directions)),
            expand<D1, PointOffset>::exec(std::get<1>(node.directions)),
            expand<D2, PointOffset>::exec(std::get<2>(node.directions)),
            expand<D3, PointOffset>::exec(std::get<3>(node.directions))}};
    }
};

namespace expression_detail {

template <typename Node, DiffScheme Scheme, std::size_t Axis,
          typename Positions = std::make_index_sequence<derivative_stencil<Scheme>::offsets.size()>>
struct make_direction;

template <typename Node, DiffScheme Scheme, std::size_t Axis, std::size_t... I>
struct make_direction<Node, Scheme, Axis, std::index_sequence<I...>> {
    template <std::size_t Position>
    using sample = expand<Node, sample_offset<typename Node::point_offset, Axis,
                                              derivative_stencil<Scheme>::offsets[Position]>>;
    using result = StencilSumExpr<Scheme, typename Node::point_offset, typename sample<I>::result...>;
    static result exec(const Node& node) {
        return {typename result::samples_type{sample<I>::exec(node)...}};
    }
};

template <typename Node, DiffScheme Scheme>
struct partial_samples {
    using D0 = make_direction<Node, Scheme, 0>;
    using D1 = make_direction<Node, Scheme, 1>;
    using D2 = make_direction<Node, Scheme, 2>;
    using D3 = make_direction<Node, Scheme, 3>;
    using result = SampledPartialExpr<typename D0::result, typename D1::result, typename D2::result,
                                      typename D3::result, typename Node::point_offset>;
    static result exec(const Node& node) {
        return {typename result::directions_type{D0::exec(node), D1::exec(node), D2::exec(node), D3::exec(node)}};
    }
};

} // namespace expression_detail

// Record the mathematical derivative. Sampling is deferred until eval(x, h).
template <typename DerivLetter, DiffScheme Scheme = DiffScheme::Central8, typename Node, typename... Indices>
auto partial_field(const IndexedExpr<Node, Indices...>& field) {
    using node = PartialExpr<Node, Scheme, typename Node::point_offset>;
    return IndexedExpr<node, dn<DerivLetter>, Indices...>{node{field.node}};
}

template <typename DerivLetter, DiffScheme Scheme = DiffScheme::Central8, typename Node, typename... Indices>
auto partial(const IndexedExpr<Node, Indices...>& field) {
    return partial_field<DerivLetter, Scheme>(field);
}

template <typename DerivLetter, DiffScheme Scheme = DiffScheme::Central8,
          typename Node, typename... Indices, typename CoordLetter>
auto partial(const IndexedExpr<Node, Indices...>& field, const coord<CoordLetter>& x, double h = 1e-2) {
    return partial_field<DerivLetter, Scheme>(field).eval(x, h);
}


template <typename Node, DiffScheme Scheme, typename OldOffset, typename PointOffset>
struct expand<PartialExpr<Node, Scheme, OldOffset>, PointOffset> {
    using result = PartialExpr<typename expand<Node, PointOffset>::result, Scheme, PointOffset>;
    static result exec(const PartialExpr<Node, Scheme, OldOffset>& node) {
        return {expand<Node, PointOffset>::exec(node.operand)};
    }
};

template <typename Node, typename Ratio, typename OldOffset, typename PointOffset>
struct expand<ScaleExpr<Node, Ratio, OldOffset>, PointOffset> {
    using result = ScaleExpr<typename expand<Node, PointOffset>::result, Ratio, PointOffset>;
    static result exec(const ScaleExpr<Node, Ratio, OldOffset>& node) {
        return {expand<Node, PointOffset>::exec(node.operand)};
    }
};

template <typename L, typename R, typename Plan, typename OldOffset, typename PointOffset>
struct expand<AddExpr<L, R, Plan, OldOffset>, PointOffset> {
    using result = AddExpr<typename expand<L, PointOffset>::result, typename expand<R, PointOffset>::result,
                           Plan, PointOffset>;
    static result exec(const AddExpr<L, R, Plan, OldOffset>& node) {
        return {expand<L, PointOffset>::exec(node.left), expand<R, PointOffset>::exec(node.right)};
    }
};

template <typename Node, typename Plan, typename OldOffset, typename PointOffset>
struct expand<TraceExpr<Node, Plan, OldOffset>, PointOffset> {
    using result = TraceExpr<typename expand<Node, PointOffset>::result, Plan, PointOffset>;
    static result exec(const TraceExpr<Node, Plan, OldOffset>& node) {
        return {expand<Node, PointOffset>::exec(node.operand)};
    }
};

template <typename Node, typename OldOffset, typename PointOffset>
struct expand<InverseExpr<Node, OldOffset>, PointOffset> {
    using result = InverseExpr<typename expand<Node, PointOffset>::result, PointOffset>;
    static result exec(const InverseExpr<Node, OldOffset>& node) {
        return {expand<Node, PointOffset>::exec(node.operand)};
    }
};

template <int Numerator, int Denominator = 1, typename Node, typename... Indices>
auto scale_field(const IndexedExpr<Node, Indices...>& field) {
    using node = ScaleExpr<Node, typename std::ratio<Numerator, Denominator>::type, typename Node::point_offset>;
    return IndexedExpr<node, Indices...>{node{field.node}};
}

template <typename Node, typename... Indices>
auto operator-(const IndexedExpr<Node, Indices...>& field) {
    return scale_field<-1>(field);
}

template <typename L, typename... LeftIndices, typename R, typename... RightIndices>
auto operator+(const IndexedExpr<L, LeftIndices...>& left, const IndexedExpr<R, RightIndices...>& right) {
    using analysis = addition_analysis<type_list<LeftIndices...>, type_list<RightIndices...>>;
    using node = AddExpr<L, R, typename analysis::plan, typename L::point_offset>;
    return IndexedExpr<node, LeftIndices...>{node{left.node, right.node}};
}

template <typename L, typename... LeftIndices, typename R, typename... RightIndices>
auto operator-(const IndexedExpr<L, LeftIndices...>& left, const IndexedExpr<R, RightIndices...>& right) {
    return left + (-right);
}

template <typename LetterA, typename LetterB, typename Node, typename... Indices>
auto trace_field(const IndexedExpr<Node, Indices...>& field) {
    using analysis = trace_analysis<type_list<Indices...>, LetterA, LetterB>;
    using node = TraceExpr<Node, typename analysis::plan, typename Node::point_offset>;
    using result = typename expression_detail::indexed_expression<node, typename analysis::free_indices>::type;
    return result{node{field.node}};
}

template <typename Node, typename First, typename Second>
auto inverse_field(const IndexedExpr<Node, First, Second>& field) {
    static_assert(is_same_variance<First, Second>::value, "metric inverse requires matching index variance");
    using node = InverseExpr<Node, typename Node::point_offset>;
    return IndexedExpr<node, typename opposite<First>::result, typename opposite<Second>::result>{node{field.node}};
}
