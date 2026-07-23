#pragma once
#include "contraction_and_arithmetic.hpp"
#include "tensor_class.hpp"
#include <cstdlib>
#include <type_traits>
#include <utility>


enum class DiffScheme {
    Central2,
    Central4,
    Central6,
    Central8,
};

struct StencilPoint {
    int offset;
    double coeff;
};


template <typename Letter>
using coord = Tensor<double, up<Letter>>;



template <typename DerivLetter, typename CoordLetter, typename F, typename datatype, typename... Indices>
auto partial_impl(const F& field, const coord<CoordLetter>& x, double h, DiffScheme scheme,
                  const Tensor<datatype, Indices...>&) {
    Tensor<datatype, dn<DerivLetter>, Indices...> result{};

    constexpr std::size_t orig_size = pow4(sizeof...(Indices));

    auto apply_stencil = [&](const auto& stencil) {
        for (std::size_t a = 0; a < 4; ++a) {
            for (std::size_t j = 0; j < orig_size; ++j) {
                result.data[a * orig_size + j] = 0.0;
            }

            for (const auto& point : stencil) {
                coord<CoordLetter> xs = x;
                xs(a) += point.offset * h;
                auto fs = field(xs);

                for (std::size_t j = 0; j < orig_size; ++j) {
                    result.data[a * orig_size + j] += point.coeff * fs.data[j];
                }
            }

            // Divide by h after applying the dimensionless stencil coefficients.
            for (std::size_t j = 0; j < orig_size; ++j) {
                result.data[a * orig_size + j] /= h;
            }
        }
    };

    switch (scheme) {
    case DiffScheme::Central2: {
        constexpr StencilPoint stencil[] = { {-1, -1.0 / 2.0}, {1, 1.0 / 2.0} };
        apply_stencil(stencil);
        break;
    }
    case DiffScheme::Central4: {
        constexpr StencilPoint stencil[] = {
            {-2, 1.0 / 12.0}, {-1, -2.0 / 3.0}, {1, 2.0 / 3.0}, {2, -1.0 / 12.0},
        };
        apply_stencil(stencil);
        break;
    }
    case DiffScheme::Central6: {
        constexpr StencilPoint stencil[] = {
            {-3, -1.0 / 60.0}, {-2, 3.0 / 20.0}, {-1, -3.0 / 4.0},
            {1, 3.0 / 4.0},    {2, -3.0 / 20.0}, {3, 1.0 / 60.0},
        };
        apply_stencil(stencil);
        break;
    }
    case DiffScheme::Central8: {
        constexpr StencilPoint stencil[] = {
            {-4, 1.0 / 280.0}, {-3, -4.0 / 105.0}, {-2, 1.0 / 5.0},  {-1, -4.0 / 5.0},
            {1, 4.0 / 5.0},    {2, -1.0 / 5.0},    {3, 4.0 / 105.0}, {4, -1.0 / 280.0},
        };
        apply_stencil(stencil);
        break;
    }
    }

    return result;
}



template <typename DerivLetter, typename CoordLetter, typename F>
auto partial(const F& field, const coord<CoordLetter>& x, double h = 1e-2, DiffScheme scheme = DiffScheme::Central8) {
    auto sample = field(x);
    return partial_impl<DerivLetter>(field, x, h, scheme, sample);
}

template <typename DerivLetter, typename CoordLetter, typename F>
auto partial(const F& field, const coord<CoordLetter>& x, DiffScheme scheme) {
    return partial<DerivLetter>(field, x, 1e-2, scheme);
}

template <typename UpLetter, typename DummyLetter, typename CoordLetter, typename F, typename InvMetricFunc>
auto partial_up(const F& field, const InvMetricFunc& inv_metric, const coord<CoordLetter>& x, double h = 1e-2,
                DiffScheme scheme = DiffScheme::Central8) {
    auto lower = partial<DummyLetter>(field, x, h, scheme);
    auto ginv = rename<up<UpLetter>, up<DummyLetter>>(inv_metric(x));

    return ginv * lower;
}

template <typename UpLetter, typename DummyLetter, typename CoordLetter, typename F, typename InvMetricFunc>
auto partial_up(const F& field, const InvMetricFunc& inv_metric, const coord<CoordLetter>& x, DiffScheme scheme) {
    return partial_up<UpLetter, DummyLetter>(field, inv_metric, x, 1e-2, scheme);
}
