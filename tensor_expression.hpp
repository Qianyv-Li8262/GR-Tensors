#pragma once

#include "contraction_and_arithmetic.hpp"
#include <type_traits>

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
template <auto Function>
struct FieldExpr {
    using source_type = std::decay_t<typename expression_detail::function_result<decltype(Function)>::type>;
    using scalar_type = typename expression_detail::tensor_traits<source_type>::scalar_type;
    using children = type_list<>;
    static constexpr std::size_t rank = expression_detail::tensor_traits<source_type>::rank;

    template <typename Coordinate>
    auto eval_components(const Coordinate& x) const {
        return Function(x).data;
    }
};

// The computation tree contains operand nodes and positional plans only.
// Index letters live in IndexedExpr, outside this type's template arguments.
template <typename L, typename R, typename ContractionPlan>
struct ContractExpr {
    using left_type = L;
    using right_type = R;
    using children = type_list<L, R>;
    using Plan = ContractionPlan;
    using scalar_type = typename L::scalar_type;
    static_assert(std::is_same_v<scalar_type, typename R::scalar_type>,
                  "contraction expression operands must have the same scalar type");
    static_assert(L::rank == Plan::left_rank && R::rank == Plan::right_rank,
                  "contraction expression operand ranks must match the plan");
    static constexpr std::size_t rank = Plan::result_rank;

    L left;
    R right;

    template <typename Coordinate>
    auto eval_components(const Coordinate& x) const {
        // Deliberately recursive tree evaluation; repeated subtrees are evaluated again.
        const auto left_values = left.eval_components(x);
        const auto right_values = right.eval_components(x);
        return contract_components<Plan>(left_values, right_values);
    }
};

template <typename Node, typename... Indices>
struct IndexedExpr {
    using node_type = Node;
    using scalar_type = typename Node::scalar_type;
    using indices = type_list<Indices...>;
    using result_type = Tensor<scalar_type, Indices...>;
    static constexpr std::size_t rank = Node::rank;
    static_assert(sizeof...(Indices) == rank, "expression indices must match the node rank");
    // Instantiating the tensor also checks valid indices and duplicate letters.
    static_assert(sizeof(result_type) > 0, "expression result must be a valid tensor");

    Node node;

    template <typename Coordinate>
    result_type eval(const Coordinate& x) const {
        result_type result;
        result.data = node.eval_components(x);
        return result;
    }

    template <typename Coordinate>
    result_type operator()(const Coordinate& x) const {
        return eval(x);
    }
};

namespace expression_detail {

template <typename Node, typename... Indices>
struct indexed_expression<Node, type_list<Indices...>> {
    using type = IndexedExpr<Node, Indices...>;
};

} // namespace expression_detail

// Named function entry point, e.g. make_field<&metric>().
// The existing make_field(callable) API remains available for opaque lambda fields.
template <auto Function>
auto make_field() {
    using node = FieldExpr<Function>;
    using indices = typename expression_detail::tensor_traits<typename node::source_type>::indices;
    using expression = typename expression_detail::indexed_expression<node, indices>::type;
    return expression{node{}};
}

template <typename L, typename... LeftIndices, typename R, typename... RightIndices>
auto operator*(const IndexedExpr<L, LeftIndices...>& left, const IndexedExpr<R, RightIndices...>& right) {
    using analysis = contraction_analysis<type_list<LeftIndices...>, type_list<RightIndices...>>;
    using node = ContractExpr<L, R, typename analysis::plan>;
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
