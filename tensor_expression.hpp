#pragma once

#include "contraction_and_arithmetic.hpp"
#include <array>
#include <type_traits>

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

template <typename Node, typename... Indices>
struct IndexedExpr {
    using node_type = Node;
    using point_offset = typename Node::point_offset;
    using scalar_type = typename Node::scalar_type;
    using indices = type_list<Indices...>;
    using result_type = Tensor<scalar_type, Indices...>;
    static constexpr std::size_t rank = Node::rank;
    static_assert(sizeof...(Indices) == rank, "expression indices must match the node rank");
    // Instantiating the tensor also checks valid indices and duplicate letters.
    static_assert(sizeof(result_type) > 0, "expression result must be a valid tensor");

    Node node;

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
