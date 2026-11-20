#include "tensors_all.hpp"
#include "examples/kretschmann_formula.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
    coord<mu> x{};
    x(1) = 10.0;
    x(2) = std::acos(-1.0) / 2.0;

    // Both derivative levels use the same runtime step. The complete curvature
    // formula remains visible to the expression engine; only the metric is a leaf.
    constexpr double h = 1.0e-2;
    const auto K = kretschmann_formula<DiffScheme::Central8>();
    const double numerical = K.eval(x, h)();
    // Geometrized units, Schwarzschild mass M = 1 (as in the metric leaf).
    const double expected = 48.0 / std::pow(x(1), 6);
    const double relative_error = std::abs((numerical - expected) / expected);

    std::cout << std::setprecision(15);
    std::cout << "Schwarzschild point (t, r, theta, phi) = (" << x(0) << ", " << x(1) << ", " << x(2) << ", "
              << x(3) << ")\n";
    std::cout << "K = R_abcd R^abcd = " << numerical << '\n';
    std::cout << "48 M^2 / r^6       = " << expected << '\n';
    std::cout << "relative error      = " << relative_error << '\n';
    return relative_error < 1.0e-5 ? 0 : 1;
}
