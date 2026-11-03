#pragma once
#include "tensor_expression.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

enum class DiffScheme { Central2, Central4, Central6, Central8 };

template <typename Letter>
using coord = Tensor<double, up<Letter>>;

// Integer weights with a common denominator; all strategy choices are types.
template <DiffScheme Scheme>
struct derivative_stencil;

template <>
struct derivative_stencil<DiffScheme::Central2> {
    inline static constexpr std::array<int, 2> offsets{-1, 1};
    inline static constexpr std::array<int, 2> weights{-1, 1};
    static constexpr int denominator = 2;
};
template <>
struct derivative_stencil<DiffScheme::Central4> {
    inline static constexpr std::array<int, 4> offsets{-2, -1, 1, 2};
    inline static constexpr std::array<int, 4> weights{1, -8, 8, -1};
    static constexpr int denominator = 12;
};
template <>
struct derivative_stencil<DiffScheme::Central6> {
    inline static constexpr std::array<int, 6> offsets{-3, -2, -1, 1, 2, 3};
    inline static constexpr std::array<int, 6> weights{-1, 9, -45, 45, -9, 1};
    static constexpr int denominator = 60;
};
template <>
struct derivative_stencil<DiffScheme::Central8> {
    inline static constexpr std::array<int, 8> offsets{-4, -3, -2, -1, 1, 2, 3, 4};
    inline static constexpr std::array<int, 8> weights{3, -32, 168, -672, 672, -168, 32, -3};
    static constexpr int denominator = 840;
};

namespace derivative_detail {

template <typename Coordinate>
void check_step(const Coordinate& x, double h) {
    assert(std::isfinite(h) && [&] {
        double scale = 1.0;
        for (double component : x.data) {
            scale = std::max(scale, std::abs(component));
        }
        return std::abs(h) > 32.0 * std::numeric_limits<double>::epsilon() * scale;
    }());
    (void)x;
    (void)h;
}

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

} // namespace derivative_detail

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
struct PartialExpr {
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

// Keep each stencil displacement when a derivative subtree moves to a new center.
template <DiffScheme Scheme, typename OldOffset, typename... Samples, typename PointOffset>
struct expand<StencilSumExpr<Scheme, OldOffset, Samples...>, PointOffset> {
    template <typename Sample>
    using relocated = expand<Sample, derivative_detail::rebase_offset<
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
struct expand<PartialExpr<D0, D1, D2, D3, OldOffset>, PointOffset> {
    using result = PartialExpr<typename expand<D0, PointOffset>::result, typename expand<D1, PointOffset>::result,
                               typename expand<D2, PointOffset>::result, typename expand<D3, PointOffset>::result,
                               PointOffset>;
    static result exec(const PartialExpr<D0, D1, D2, D3, OldOffset>& node) {
        return {typename result::directions_type{
            expand<D0, PointOffset>::exec(std::get<0>(node.directions)),
            expand<D1, PointOffset>::exec(std::get<1>(node.directions)),
            expand<D2, PointOffset>::exec(std::get<2>(node.directions)),
            expand<D3, PointOffset>::exec(std::get<3>(node.directions))}};
    }
};

namespace derivative_detail {

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

} // namespace derivative_detail

// Build the complete sampling tree now; functions are called only by eval(x, h).
template <typename DerivLetter, DiffScheme Scheme = DiffScheme::Central8, typename Node, typename... Indices>
auto partial_field(const IndexedExpr<Node, Indices...>& field) {
    using D0 = derivative_detail::make_direction<Node, Scheme, 0>;
    using D1 = derivative_detail::make_direction<Node, Scheme, 1>;
    using D2 = derivative_detail::make_direction<Node, Scheme, 2>;
    using D3 = derivative_detail::make_direction<Node, Scheme, 3>;
    using node = PartialExpr<typename D0::result, typename D1::result, typename D2::result, typename D3::result,
                             typename Node::point_offset>;
    return IndexedExpr<node, dn<DerivLetter>, Indices...>{node{typename node::directions_type{
        D0::exec(field.node), D1::exec(field.node), D2::exec(field.node), D3::exec(field.node)}}};
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

// Opaque callable fields retain direct evaluation, with compile-time strategy selection.
// Result metadata is obtained in an unevaluated context: no extra center sample.
namespace derivative_detail {

template <typename DerivLetter, DiffScheme Scheme, typename F, typename CoordLetter, std::size_t... I>
auto evaluate_callable(const F& field, const coord<CoordLetter>& x, double h, std::index_sequence<I...>) {
    using source = expression_detail::tensor_traits<std::decay_t<decltype(field.eval(x))>>;
    using indices = typename append_front<dn<DerivLetter>, typename source::indices>::result;
    using result_type = typename construct_tensor_from_list<typename source::scalar_type, indices>::result;
    using stencil = derivative_stencil<Scheme>;
    constexpr std::size_t size = pow4(source::rank);
    result_type result;
    for (std::size_t axis = 0; axis < 4; ++axis) {
        auto accumulate = [&](auto position) {
            constexpr std::size_t i = decltype(position)::value;
            auto sample = x;
            sample(axis) += stencil::offsets[i] * h;
            const auto values = field.eval(sample);
            for (std::size_t j = 0; j < size; ++j) {
                constexpr double weight = static_cast<double>(stencil::weights[i]) / stencil::denominator;
                result.data[axis * size + j] += weight * values.data[j];
            }
        };
        (accumulate(std::integral_constant<std::size_t, I>{}), ...);
        for (std::size_t j = 0; j < size; ++j) {
            result.data[axis * size + j] /= h;
        }
    }
    return result;
}

} // namespace derivative_detail

template <typename DerivLetter, DiffScheme Scheme = DiffScheme::Central8, typename CoordLetter, typename F>
auto partial(const F& field, const coord<CoordLetter>& x, double h = 1e-2) {
    derivative_detail::check_step(x, h);
    return derivative_detail::evaluate_callable<DerivLetter, Scheme>(
        field, x, h, std::make_index_sequence<derivative_stencil<Scheme>::offsets.size()>{});
}

template <typename UpLetter, typename DummyLetter, DiffScheme Scheme = DiffScheme::Central8,
          typename CoordLetter, typename F, typename InvMetricFunc>
auto partial_up(const F& field, const InvMetricFunc& inv_metric, const coord<CoordLetter>& x, double h = 1e-2) {
    auto lower = partial<DummyLetter, Scheme>(field, x, h);
    auto ginv = rename<up<UpLetter>, up<DummyLetter>>(inv_metric.eval(x));
    return ginv * lower;
}
