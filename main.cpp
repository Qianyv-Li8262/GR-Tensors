#include "tensors_all.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <type_traits>

namespace {

// Geometrized units: G = c = 1.  Coordinates are ordered as (t, r, theta, phi).
constexpr double schwarzschild_mass = 1.0;

template <typename FirstIndex, typename SecondIndex, typename CoordinateLetter>
auto schwarzschild_metric(const coord<CoordinateLetter>& x) {
    static_assert(!index_traits<FirstIndex>::variance::value && !index_traits<SecondIndex>::variance::value,
                  "the Schwarzschild metric must have two lower indices");

    Tensor<double, FirstIndex, SecondIndex> metric{};
    const double r = x(1);
    const double theta = x(2);
    const double lapse_squared = 1.0 - 2.0 * schwarzschild_mass / r;

    metric(0, 0) = -lapse_squared;
    metric(1, 1) = 1.0 / lapse_squared;
    metric(2, 2) = r * r;
    metric(3, 3) = r * r * std::sin(theta) * std::sin(theta);
    return metric;
}

template <typename FirstIndex, typename SecondIndex, typename CoordinateLetter>
auto schwarzschild_inverse_metric(const coord<CoordinateLetter>& x) {
    static_assert(index_traits<FirstIndex>::variance::value && index_traits<SecondIndex>::variance::value,
                  "the inverse Schwarzschild metric must have two upper indices");

    Tensor<double, FirstIndex, SecondIndex> inverse_metric{};
    const double r = x(1);
    const double theta = x(2);
    const double lapse_squared = 1.0 - 2.0 * schwarzschild_mass / r;

    inverse_metric(0, 0) = -1.0 / lapse_squared;
    inverse_metric(1, 1) = lapse_squared;
    inverse_metric(2, 2) = 1.0 / (r * r);
    inverse_metric(3, 3) = 1.0 / (r * r * std::sin(theta) * std::sin(theta));
    return inverse_metric;
}

DEFINE_TENSOR_FIELD2(g, schwarzschild_metric);
DEFINE_TENSOR_FIELD2(g_inverse, schwarzschild_inverse_metric);

// Levi-Civita connection:
// Gamma^Rho_{Mu Nu} = 1/2 g^{Rho Kappa}
//                       (d_Mu g_{Kappa Nu} + d_Nu g_{Kappa Mu} - d_Kappa g_{Mu Nu}).
template <typename Rho, typename Mu, typename Nu, typename CoordinateLetter>
auto christoffel(const coord<CoordinateLetter>& x, double h) {
    const auto derivative_mu = partial<Mu>(g<dn<kappa>, dn<Nu>>, x, h, DiffScheme::Central8);
    const auto derivative_nu = partial<Nu>(g<dn<kappa>, dn<Mu>>, x, h, DiffScheme::Central8);
    const auto derivative_kappa = partial<kappa>(g<dn<Mu>, dn<Nu>>, x, h, DiffScheme::Central8);

    const auto inverse = g_inverse<up<Rho>, up<kappa>>.eval(x);
    return 0.5 * (inverse * (derivative_mu + derivative_nu - derivative_kappa));
}

template <typename Rho, typename Mu, typename Nu>
auto christoffel_field(double h) {
    return make_field([h](const auto& x) {
        return christoffel<Rho, Mu, Nu>(x, h);
    });
}

// Riemann tensor convention:
// R^Rho_{Sigma Mu Nu} = d_Mu Gamma^Rho_{Nu Sigma} - d_Nu Gamma^Rho_{Mu Sigma}
//                       + Gamma^Rho_{Mu Lambda} Gamma^Lambda_{Nu Sigma}
//                       - Gamma^Rho_{Nu Lambda} Gamma^Lambda_{Mu Sigma}.
template <typename Rho, typename Sigma, typename Mu, typename Nu, typename CoordinateLetter>
auto riemann(const coord<CoordinateLetter>& x, double connection_step, double curvature_step) {
    const auto derivative_mu =
        partial<Mu>(christoffel_field<Rho, Nu, Sigma>(connection_step), x, curvature_step, DiffScheme::Central8);
    const auto derivative_nu =
        partial<Nu>(christoffel_field<Rho, Mu, Sigma>(connection_step), x, curvature_step, DiffScheme::Central8);

    const auto quadratic_mu_nu =
        christoffel<Rho, Mu, lambda>(x, connection_step) * christoffel<lambda, Nu, Sigma>(x, connection_step);
    const auto quadratic_nu_mu =
        christoffel<Rho, Nu, lambda>(x, connection_step) * christoffel<lambda, Mu, Sigma>(x, connection_step);

    const auto raw = derivative_mu - derivative_nu + quadratic_mu_nu - quadratic_nu_mu;

    // Addition with a zero tensor puts the free indices in the conventional order.
    const Tensor<double, up<Rho>, dn<Sigma>, dn<Mu>, dn<Nu>> canonical_order{};
    return canonical_order + raw;
}

template <typename CoordinateLetter>
double kretschmann_scalar(const coord<CoordinateLetter>& x, double connection_step, double curvature_step) {
    // First lower the leading index: R_abcd = g_ae R^e_bcd.
    const auto mixed_riemann = riemann<rho, b, c, d>(x, connection_step, curvature_step);
    const auto riemann_lower = g<dn<a>, dn<rho>>.eval(x) * mixed_riemann;

    // Raise all four indices of a second copy.  Each multiplication performs one
    // Einstein contraction; its storage order is immaterial because indices carry names.
    const auto raise_a =
        g_inverse<up<a>, up<mu>>.eval(x) * rename<dn<mu>, dn<nu>, dn<kappa>, dn<sigma>>(riemann_lower);
    const auto raise_b = g_inverse<up<b>, up<nu>>.eval(x) * raise_a;
    const auto raise_c = g_inverse<up<c>, up<kappa>>.eval(x) * raise_b;
    const auto riemann_upper = g_inverse<up<d>, up<sigma>>.eval(x) * raise_c;

    // K = R_abcd R^abcd.  A rank-zero Tensor stores its single scalar at ().
    const auto kretschmann = riemann_lower * riemann_upper;
    return kretschmann();
}

} // namespace

int main() {
    coord<mu> x{};
    x(0) = 0.0;
    x(1) = 10.0;
    x(2) = std::acos(-1.0) / 2.0;
    x(3) = 0.0;

    // The Riemann tensor contains second derivatives of the metric.  Separate step
    // sizes make the two nested eighth-order finite differences easy to tune.
    constexpr double connection_step = 1.0e-3;
    constexpr double curvature_step = 2.0e-3;

    const double numerical = kretschmann_scalar(x, connection_step, curvature_step);
    const double expected = 48.0 * schwarzschild_mass * schwarzschild_mass / std::pow(x(1), 6);
    const double relative_error = std::abs((numerical - expected) / expected);

    std::cout << std::setprecision(15);
    std::cout << "Schwarzschild point (t, r, theta, phi) = (" << x(0) << ", " << x(1) << ", " << x(2) << ", "
              << x(3) << ")\n";
    std::cout << "K = R_abcd R^abcd = " << numerical << '\n';
    std::cout << "48 M^2 / r^6       = " << expected << '\n';
    std::cout << "relative error      = " << relative_error << '\n';
    return relative_error < 1.0e-5 ? 0 : 1;
}
