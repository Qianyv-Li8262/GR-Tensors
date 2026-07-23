#include "symm_asymm.hpp"
#include "num_deriv.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>

namespace {

int failures = 0;

void check_close(const std::string& name, double got, double expected, double tol) {
    const double err = std::abs(got - expected);
    if (err <= tol) {
        std::cout << "[PASS] " << name << " got=" << got << " expected=" << expected << " err=" << err << '\n';
    } else {
        ++failures;
        std::cout << "[FAIL] " << name << " got=" << got << " expected=" << expected << " err=" << err
                  << " tol=" << tol << '\n';
    }
}

void check_true(const std::string& name, bool ok) {
    if (ok) {
        std::cout << "[PASS] " << name << '\n';
    } else {
        ++failures;
        std::cout << "[FAIL] " << name << '\n';
    }
}

void test_symm_rank2() {
    Tensor<double, up<mu>, up<nu>> t;
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            t(i, j) = 10.0 * static_cast<double>(i) + static_cast<double>(j);
        }
    }

    auto s = symm<up<mu>, up<nu>>(t);
    check_true("symm rank2 preserves Tensor<double, up<mu>, up<nu>> type",
               std::is_same_v<decltype(s), Tensor<double, up<mu>, up<nu>>>);

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            const double expected = 0.5 * (t(i, j) + t(j, i));
            check_close("symm rank2 component", s(i, j), expected, 1e-12);
        }
    }
}

void test_symm_rank3_selected_pair() {
    Tensor<double, dn<mu>, dn<nu>, up<rho>> t;
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            for (std::size_t k = 0; k < 4; ++k) {
                t(i, j, k) = 100.0 * static_cast<double>(i) + 10.0 * static_cast<double>(j) +
                             static_cast<double>(k);
            }
        }
    }

    auto s = symm<dn<mu>, dn<nu>>(t);
    check_true("symm selected pair preserves untouched index and type",
               std::is_same_v<decltype(s), Tensor<double, dn<mu>, dn<nu>, up<rho>>>);

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            for (std::size_t k = 0; k < 4; ++k) {
                const double expected = 0.5 * (t(i, j, k) + t(j, i, k));
                check_close("symm rank3 selected pair component", s(i, j, k), expected, 1e-12);
            }
        }
    }
}

void test_asymm_rank2() {
    Tensor<double, up<mu>, up<nu>> t;
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            t(i, j) = 10.0 * static_cast<double>(i) + static_cast<double>(j);
        }
    }

    auto a = asymm<up<mu>, up<nu>>(t);
    check_true("asymm rank2 preserves Tensor<double, up<mu>, up<nu>> type",
               std::is_same_v<decltype(a), Tensor<double, up<mu>, up<nu>>>);

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            const double expected = 0.5 * (t(i, j) - t(j, i));
            check_close("asymm rank2 component", a(i, j), expected, 1e-12);
            check_close("asymm rank2 antisymmetry", a(i, j), -a(j, i), 1e-12);
        }
    }
}

void test_asymm_rank3_all_indices() {
    Tensor<double, dn<mu>, dn<nu>, dn<rho>> t;
    t(0, 1, 2) = 6.0;
    t(2, 3, 1) = -3.0;

    auto a = asymm<dn<mu>, dn<nu>, dn<rho>>(t);
    check_true("asymm rank3 preserves type",
               std::is_same_v<decltype(a), Tensor<double, dn<mu>, dn<nu>, dn<rho>>>);

    auto signed_projection = [](std::size_t i, std::size_t j, std::size_t k, std::size_t a0, std::size_t a1,
                                  std::size_t a2, double value) {
        const std::size_t dst[] = {i, j, k};
        const std::size_t src[] = {a0, a1, a2};
        int perm[] = {-1, -1, -1};
        bool used[] = {false, false, false};

        for (int p = 0; p < 3; ++p) {
            for (int q = 0; q < 3; ++q) {
                if (!used[q] && dst[p] == src[q]) {
                    perm[p] = q;
                    used[q] = true;
                    break;
                }
            }
            if (perm[p] < 0) {
                return 0.0;
            }
        }

        int inversions = 0;
        for (int p = 0; p < 3; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                if (perm[p] > perm[q]) {
                    ++inversions;
                }
            }
        }

        const double sign = inversions % 2 == 0 ? 1.0 : -1.0;
        return sign * value / 6.0;
    };

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            for (std::size_t k = 0; k < 4; ++k) {
                const double expected = signed_projection(i, j, k, 0, 1, 2, 6.0) +
                                        signed_projection(i, j, k, 2, 3, 1, -3.0);
                check_close("asymm rank3 component", a(i, j, k), expected, 1e-12);
            }
        }
    }
}

void test_trace_scalar() {
    Tensor<double, up<mu>, dn<nu>> t;
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            t(i, j) = 10.0 * static_cast<double>(i) + static_cast<double>(j);
        }
    }

    auto tr = trace<mu, nu>(t);
    check_true("trace rank2 returns scalar", std::is_same_v<decltype(tr), Tensor<double>>);
    check_close("trace rank2 value", tr.data[0], 66.0, 1e-12);
}

void test_trace_with_free_index() {
    Tensor<double, up<rho>, dn<mu>, up<nu>> t;
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                t(r, i, j) = 100.0 * static_cast<double>(r) + 10.0 * static_cast<double>(i) +
                             static_cast<double>(j);
            }
        }
    }

    auto tr = trace<mu, nu>(t);
    check_true("trace preserves free index", std::is_same_v<decltype(tr), Tensor<double, up<rho>>>);

    for (std::size_t r = 0; r < 4; ++r) {
        double expected = 0.0;
        for (std::size_t i = 0; i < 4; ++i) {
            expected += t(r, i, i);
        }
        check_close("trace free-index component", tr(r), expected, 1e-12);
    }
}

void test_partial_scalar_field() {
    coord<mu> x;
    x(0) = 1.25;
    x(1) = -0.75;
    x(2) = 0.5;
    x(3) = 2.0;

    auto scalar_field = [](const coord<mu>& y) {
        Tensor<double> value;
        value.data[0] = y(0) * y(0) + 3.0 * y(1) - 2.0 * y(2) * y(3) + 5.0;
        return value;
    };

    auto d = partial<nu>(scalar_field, x, 1e-3, DiffScheme::Central8);
    check_true("partial scalar field returns one lower derivative index",
               std::is_same_v<decltype(d), Tensor<double, dn<nu>>>);

    check_close("partial scalar d0", d(0), 2.0 * x(0), 1e-9);
    check_close("partial scalar d1", d(1), 3.0, 1e-9);
    check_close("partial scalar d2", d(2), -2.0 * x(3), 1e-9);
    check_close("partial scalar d3", d(3), -2.0 * x(2), 1e-9);
}

void test_partial_vector_field() {
    coord<mu> x;
    x(0) = 0.25;
    x(1) = -1.5;
    x(2) = 0.75;
    x(3) = 1.25;

    auto vector_field = [](const coord<mu>& y) {
        Tensor<double, up<nu>> value;
        for (std::size_t b = 0; b < 4; ++b) {
            value(b) = static_cast<double>(b + 1) * y(b) * y(b) + 0.5 * y(0);
        }
        return value;
    };

    auto d = partial<rho>(vector_field, x, 1e-3, DiffScheme::Central8);
    check_true("partial vector field returns derivative plus original index",
               std::is_same_v<decltype(d), Tensor<double, dn<rho>, up<nu>>>);

    for (std::size_t a = 0; a < 4; ++a) {
        for (std::size_t b = 0; b < 4; ++b) {
            double expected = 0.0;
            if (a == b) {
                expected += 2.0 * static_cast<double>(b + 1) * x(b);
            }
            if (a == 0) {
                expected += 0.5;
            }
            check_close("partial vector component", d(a, b), expected, 1e-8);
        }
    }
}

void test_partial_up_scalar_field() {
    coord<mu> x;
    x(0) = 1.0;
    x(1) = 2.0;
    x(2) = -1.0;
    x(3) = 0.5;

    auto scalar_field = [](const coord<mu>& y) {
        Tensor<double> value;
        value.data[0] = y(0) + 2.0 * y(1) + 3.0 * y(2) + 4.0 * y(3);
        return value;
    };

    auto inverse_metric = [](const coord<mu>&) {
        Tensor<double, up<sigma>, up<lambda>> ginv;
        for (std::size_t i = 0; i < 4; ++i) {
            for (std::size_t j = 0; j < 4; ++j) {
                ginv(i, j) = i == j ? 1.0 : 0.0;
            }
        }
        ginv(2, 2) = -1.0;
        ginv(3, 3) = -2.0;
        return ginv;
    };

    auto d_up = partial_up<rho, lambda>(scalar_field, inverse_metric, x, 1e-3, DiffScheme::Central8);
    check_true("partial_up scalar field returns one upper derivative index",
               std::is_same_v<decltype(d_up), Tensor<double, up<rho>>>);

    check_close("partial_up scalar d0", d_up(0), 1.0, 1e-9);
    check_close("partial_up scalar d1", d_up(1), 2.0, 1e-9);
    check_close("partial_up scalar d2", d_up(2), -3.0, 1e-9);
    check_close("partial_up scalar d3", d_up(3), -8.0, 1e-9);
}

} // namespace

int main() {
    test_symm_rank2();
    test_symm_rank3_selected_pair();
    test_asymm_rank2();
    test_asymm_rank3_all_indices();
    test_trace_scalar();
    test_trace_with_free_index();
    test_partial_scalar_field();
    test_partial_vector_field();
    test_partial_up_scalar_field();

    if (failures == 0) {
        std::cout << "All tests passed.\n";
        return 0;
    }

    std::cout << failures << " test(s) failed.\n";
    return 1;
}
