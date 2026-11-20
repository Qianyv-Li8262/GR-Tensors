#pragma once

#include "tensor_expression.hpp"

namespace dag_detail {

template <typename Scalar, std::size_t Rank, typename Offset>
struct ArrayOperation {
    using scalar_type = Scalar;
    using point_offset = Offset;
    static constexpr std::size_t rank = Rank;
    using result_type = std::array<Scalar, pow4(rank)>;
};

template <std::size_t Begin, typename Function, std::size_t... I>
void component_block(Function& function, std::index_sequence<I...>) {
    (function(std::integral_constant<std::size_t, Begin + I>{}), ...);
}

template <std::size_t Begin, std::size_t End, typename Function>
void component_blocks(Function& function) {
    if constexpr (End - Begin <= 128) {
        component_block<Begin>(function, std::make_index_sequence<End - Begin>{});
    } else {
        constexpr auto middle = Begin + (End - Begin) / 2;
        component_blocks<Begin, middle>(function);
        component_blocks<middle, End>(function);
    }
}

// Numeric kernels depend only on array geometry and scalar type. Keeping them
// outside the operation classes shares component expansion across node IDs,
// sampling offsets and graph Context types.
template <typename LeftLayout, typename RightLayout, typename Scalar>
auto add_layout_components(const std::array<Scalar, LeftLayout::size>& left,
                           const std::array<Scalar, RightLayout::size>& right) {
    static_assert(LeftLayout::size == RightLayout::size, "addition layouts must have equal size");
    std::array<Scalar, LeftLayout::size> result{};
    auto write = [&](auto position) {
        constexpr auto i = decltype(position)::value;
        result[i] = left[LeftLayout::template source<i>] + right[RightLayout::template source<i>];
    };
    component_blocks<0, LeftLayout::size>(write);
    return result;
}

template <typename Layout, typename Scalar>
void copy_layout_components(const std::array<Scalar, Layout::size>& input, Scalar* output) {
    auto write = [&](auto position) {
        constexpr auto i = decltype(position)::value;
        output[i] = input[Layout::template source<i>];
    };
    component_blocks<0, Layout::size>(write);
}

template <typename Ratio, typename Scalar, std::size_t Rank, typename Offset>
struct ScaleOp : ArrayOperation<Scalar, Rank, Offset> {
    template <std::size_t Input, typename Context>
    static auto eval(Context& context) {
        auto result = context.template get<Input>();
        for (auto& value : result) value *= static_cast<double>(Ratio::num) / Ratio::den;
        return result;
    }
};

// Both layouts map this operation's logical output axes to stored input axes.
// Component reads absorb layouts; no permuted input buffers are created.
template <typename LeftLayout, typename RightLayout, typename Scalar, typename Offset>
struct AddOp : ArrayOperation<Scalar, LeftLayout::rank, Offset> {
    template <std::size_t Left, std::size_t Right, typename Context>
    static auto eval(Context& context) {
        decltype(auto) left = context.template get<Left>();
        decltype(auto) right = context.template get<Right>();
        return add_layout_components<LeftLayout, RightLayout>(left, right);
    }
};

template <typename Plan, typename Scalar, typename Offset>
struct TraceOp : ArrayOperation<Scalar, Plan::result_rank, Offset> {
    template <std::size_t Input, typename Context>
    static auto eval(Context& context) {
        return trace_components<Plan>(context.template get<Input>());
    }
};

template <typename Plan, typename Scalar, typename Offset>
struct SymmetryOp : ArrayOperation<Scalar, Plan::result_rank, Offset> {
    template <std::size_t Input, typename Context>
    static auto eval(Context& context) {
        if constexpr (Plan::antisymmetric) {
            return asym_components<Plan>(context.template get<Input>());
        } else {
            return sym_components<Plan>(context.template get<Input>());
        }
    }
};

template <typename Scalar, typename Offset>
struct InverseOp : ArrayOperation<Scalar, 2, Offset> {
    template <std::size_t Input, typename Context>
    static auto eval(Context& context) {
        return expression_detail::inverse_components(context.template get<Input>());
    }
};

// A primitive runtime-step division, not a recursive mathematical derivative.
template <typename Scalar, std::size_t Rank, typename Offset>
struct DivideStepOp : ArrayOperation<Scalar, Rank, Offset> {
    template <std::size_t Input, typename Context>
    static auto eval(Context& context) {
        auto result = context.template get<Input>();
        for (auto& value : result) value /= context.h;
        return result;
    }
};

template <typename Operation>
struct operation_uses_step : std::false_type {};

template <typename Scalar, std::size_t Rank, typename Offset>
struct operation_uses_step<DivideStepOp<Scalar, Rank, Offset>> : std::true_type {};

// Assemble four directional arrays, adding the derivative axis at the front.
template <typename Scalar, typename Offset, typename L0, typename L1, typename L2, typename L3>
struct PackDirectionsOp : ArrayOperation<Scalar, L0::rank + 1, Offset> {
    template <std::size_t D0, std::size_t D1, std::size_t D2, std::size_t D3, typename Context>
    static auto eval(Context& context) {
        typename PackDirectionsOp::result_type result{};
        copy_layout_components<L0>(context.template get<D0>(), result.data());
        copy_layout_components<L1>(context.template get<D1>(), result.data() + L0::size);
        copy_layout_components<L2>(context.template get<D2>(), result.data() + 2 * L0::size);
        copy_layout_components<L3>(context.template get<D3>(), result.data() + 3 * L0::size);
        return result;
    }
};

} // namespace dag_detail
