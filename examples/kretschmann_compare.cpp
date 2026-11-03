#include "kretschmann_formula.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>

// Independent analytic data: d[k][i][j] = g_ij,k; dd[k][l][i][j] = g_ij,kl.
// No expression nodes, difference stencils or tensor contraction helpers are used here.
struct MetricJet {
    double g[4][4]{};
    double inverse[4][4]{};
    double d[4][4][4]{};
    double dd[4][4][4][4]{};
};

MetricJet analytic_metric_jet(double r, double theta) {
    MetricJet j;
    const double s = std::sin(theta), c = std::cos(theta);
    const double f = 1 - 2 / r, df = 2 / (r * r), ddf = -4 / (r * r * r);
    j.g[0][0] = -f;
    j.g[1][1] = 1 / f;
    j.g[2][2] = r * r;
    j.g[3][3] = r * r * s * s;
    for (int a = 0; a < 4; ++a) j.inverse[a][a] = 1 / j.g[a][a];
    j.d[1][0][0] = -df;
    j.d[1][1][1] = -df / (f * f);
    j.d[1][2][2] = 2 * r;
    j.d[1][3][3] = 2 * r * s * s;
    j.d[2][3][3] = 2 * r * r * s * c;
    j.dd[1][1][0][0] = -ddf;
    j.dd[1][1][1][1] = 2 * df * df / (f * f * f) - ddf / (f * f);
    j.dd[1][1][2][2] = 2;
    j.dd[1][1][3][3] = 2 * s * s;
    j.dd[1][2][3][3] = j.dd[2][1][3][3] = 4 * r * s * c;
    j.dd[2][2][3][3] = 2 * r * r * (c * c - s * s);
    return j;
}

struct AnalyticCurvature {
    double R[4][4][4][4]{};
    double K = 0;
};

AnalyticCurvature curvature_from_metric_jet(const MetricJet& g) {
    double C[4][4][4]{};
    for (int p = 0; p < 4; ++p) {
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                C[p][a][b] = (g.d[a][p][b] + g.d[b][p][a] - g.d[p][a][b]) / 2;
            }
        }
    }
    AnalyticCurvature result;
    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            for (int c = 0; c < 4; ++c) {
                for (int d = 0; d < 4; ++d) {
                    double value = (g.dd[b][c][a][d] + g.dd[a][d][b][c] -
                                    g.dd[b][d][a][c] - g.dd[a][c][b][d]) / 2;
                    for (int p = 0; p < 4; ++p) {
                        for (int q = 0; q < 4; ++q) {
                            value += g.inverse[p][q] *
                                     (C[p][d][a] * C[q][c][b] - C[p][c][a] * C[q][d][b]);
                        }
                    }
                    result.R[a][b][c][d] = value;
                    // Schwarzschild inverse metric is diagonal, so the full contraction reduces to this sum.
                    result.K += g.inverse[a][a] * g.inverse[b][b] * g.inverse[c][c] * g.inverse[d][d] * value * value;
                }
            }
        }
    }
    return result;
}

int main() {
    const auto K = kretschmann_formula<DiffScheme::Central8>();
    const auto lower = g<a, rho>() * riemann_formula<rho, b, c, d, DiffScheme::Central8>();
    bool passed = metric_calls == 0;
    std::cout << std::scientific << std::setprecision(12);
    std::cout << "r theta h K_type K_jet K_closed K_relerr R_maxerr R_relerr metric_calls_K_plus_R\n";
    const double points[][2] = {{4.0, 0.7}, {6.0, 1.1}, {10.0, std::acos(-1.0) / 2}};
    for (const auto& point : points) {
        coord<mu> x;
        x(1) = point[0];
        x(2) = point[1];
        const auto jet = analytic_metric_jet(x(1), x(2));
        const auto exact = curvature_from_metric_jet(jet);
        const double closed = 48 / std::pow(x(1), 6);
        passed = passed && std::abs(exact.K / closed - 1) < 1e-12;
        const double r = x(1), s2 = std::pow(std::sin(x(2)), 2), f = 1 - 2 / r;
        const double independent_components[] = {-2 / std::pow(r, 3), f / r, f * s2 / r,
                                                   -1 / (r * f), -s2 / (r * f), 2 * r * s2};
        int pair = 0;
        for (int a = 0; a < 4; ++a) {
            for (int b = a + 1; b < 4; ++b) {
                passed = passed && std::abs(exact.R[a][b][a][b] - independent_components[pair++]) < 1e-12;
            }
        }
        for (double h : {0.02, 0.01}) {
            metric_calls = 0;
            const double numerical = K.eval(x, h)();
            const auto numerical_R = lower.eval(x, h);
            double maximum_error = 0, maximum_reference = 0;
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    for (int c = 0; c < 4; ++c) {
                        for (int d = 0; d < 4; ++d) {
                            maximum_error = std::max(maximum_error,
                                std::abs(numerical_R(a, c, d, b) - exact.R[a][b][c][d]));
                            maximum_reference = std::max(maximum_reference, std::abs(exact.R[a][b][c][d]));
                        }
                    }
                }
            }
            const double relative_error = std::abs(numerical / exact.K - 1);
            const double curvature_error = maximum_error / maximum_reference;
            passed = passed && relative_error < 5e-7 && curvature_error < 5e-8;
            std::cout << x(1) << ' ' << x(2) << ' ' << h << ' ' << numerical << ' ' << exact.K << ' '
                      << closed << ' ' << relative_error << ' ' << maximum_error << ' ' << curvature_error
                      << ' ' << metric_calls << '\n';
        }
    }
    return passed ? 0 : 1;
}
