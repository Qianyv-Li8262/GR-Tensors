#include "contraction_and_arithmetic.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>

namespace {

using matrix_analysis = contraction_analysis<type_list<up<a>, dn<b>>, type_list<up<b>, dn<c>>>;
using renamed_analysis = contraction_analysis<type_list<up<mu>, dn<nu>>, type_list<up<nu>, dn<rho>>>;
using matrix_plan = contraction_plan<2, 2, contractpair<1, 0>>;
static_assert(std::is_same_v<matrix_analysis::plan, matrix_plan>);
static_assert(std::is_same_v<matrix_analysis::plan, renamed_analysis::plan>);
static_assert(std::is_same_v<matrix_analysis::free_indices, type_list<up<a>, dn<c>>>);
static_assert(!matrix_plan::is_contracted(0, 0) && matrix_plan::is_contracted(1, 0));
static_assert(matrix_plan::is_contracted(0, 1) && !matrix_plan::is_contracted(1, 1));
static_assert(std::is_empty_v<matrix_plan>);

using fourth_rank_analysis =
    contraction_analysis<type_list<up<a>, dn<mu>, up<b>, dn<nu>>, type_list<up<mu>, dn<c>, up<d>, up<nu>>>;
using fourth_rank_plan = contraction_plan<4, 4, contractpair<1, 0>, contractpair<3, 3>>;
static_assert(std::is_same_v<fourth_rank_analysis::plan, fourth_rank_plan>);
static_assert(std::is_same_v<fourth_rank_plan::pairs, type_list<contractpair<1, 0>, contractpair<3, 3>>>);
static_assert(fourth_rank_plan::result_rank == 4);

int failures = 0;
std::mt19937 generator(20261101);

template <typename T>
void fill(T& tensor) {
    std::uniform_real_distribution<double> distribution(-2.0, 2.0);
    for (auto& component : tensor.data) {
        component = distribution(generator);
    }
}

template <typename T>
void check(const char* name, const T& actual, const T& expected) {
    for (std::size_t i = 0; i < actual.data.size(); ++i) {
        const double tolerance = 1e-12 * std::max(1.0, std::abs(expected.data[i]));
        if (!std::isfinite(actual.data[i]) || std::abs(actual.data[i] - expected.data[i]) > tolerance) {
            std::cerr << name << ": mismatch at component " << i << '\n';
            ++failures;
            return;
        }
    }
}

void matrix_product() {
    Tensor<double, up<a>, dn<b>> left;
    Tensor<double, up<b>, dn<c>> right;
    fill(left);
    fill(right);
    Tensor<double, up<a>, dn<c>> expected;
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            for (std::size_t k = 0; k < 4; ++k) {
                expected(i, j) += left(i, k) * right(k, j);
            }
        }
    }
    check("matrix product", left * right, expected);
}

void crossed_multiple_contractions() {
    Tensor<double, up<a>, dn<b>, dn<c>> left;
    Tensor<double, up<c>, dn<d>, up<b>> right;
    fill(left);
    fill(right);
    Tensor<double, up<a>, dn<d>> expected;
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            for (std::size_t k = 0; k < 4; ++k) {
                for (std::size_t l = 0; l < 4; ++l) {
                    expected(i, j) += left(i, k, l) * right(l, j, k);
                }
            }
        }
    }
    using analysis = contraction_analysis<type_list<up<a>, dn<b>, dn<c>>, type_list<up<c>, dn<d>, up<b>>>;
    static_assert(std::is_same_v<analysis::plan, contraction_plan<3, 3, contractpair<1, 2>, contractpair<2, 0>>>);
    check("crossed multiple contractions", left * right, expected);
}

void fourth_rank_product() {
    Tensor<double, up<a>, dn<mu>, up<b>, dn<nu>> left;
    Tensor<double, up<mu>, dn<c>, up<d>, up<nu>> right;
    fill(left);
    fill(right);
    Tensor<double, up<a>, up<b>, dn<c>, up<d>> expected;
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            for (std::size_t k = 0; k < 4; ++k) {
                for (std::size_t l = 0; l < 4; ++l) {
                    for (std::size_t m = 0; m < 4; ++m) {
                        for (std::size_t n = 0; n < 4; ++n) {
                            expected(i, j, k, l) += left(i, m, j, n) * right(m, k, l, n);
                        }
                    }
                }
            }
        }
    }
    check("fourth rank product", left * right, expected);
}

void scalar_result() {
    Tensor<double, up<a>, dn<b>> left;
    Tensor<double, up<b>, dn<a>> right;
    fill(left);
    fill(right);
    Tensor<double> expected;
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            expected() += left(i, j) * right(j, i);
        }
    }
    check("full contraction to scalar", left * right, expected);
}

void outer_product() {
    Tensor<double, up<a>> left;
    Tensor<double, dn<b>> right;
    fill(left);
    fill(right);
    Tensor<double, up<a>, dn<b>> expected;
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            expected(i, j) = left(i) * right(j);
        }
    }
    using analysis = contraction_analysis<type_list<up<a>>, type_list<dn<b>>>;
    static_assert(std::is_same_v<analysis::plan, contraction_plan<1, 1>>);
    check("outer product without pairs", left * right, expected);
}

void scalar_operands() {
    Tensor<double> scalar(2.5);
    Tensor<double, up<a>, dn<b>> tensor;
    fill(tensor);
    auto expected = tensor;
    for (auto& component : expected.data) {
        component *= scalar();
    }
    check("scalar left operand", scalar * tensor, expected);
    check("scalar right operand", tensor * scalar, expected);
    Tensor<double> scalar_expected(6.25);
    check("two scalar operands", scalar * scalar, scalar_expected);
}

void traces() {
    Tensor<double, up<a>, dn<b>, up<c>, dn<d>> input;
    fill(input);
    using analysis = trace_analysis<type_list<up<a>, dn<b>, up<c>, dn<d>>, a, d>;
    using renamed = trace_analysis<type_list<up<mu>, dn<nu>, up<rho>, dn<sigma>>, mu, sigma>;
    static_assert(std::is_same_v<analysis::plan, trace_plan<4, 0, 3>>);
    static_assert(std::is_same_v<analysis::plan, renamed::plan>);
    static_assert(std::is_same_v<analysis::free_indices, type_list<dn<b>, up<c>>>);
    static_assert(std::is_same_v<decltype(trace<a, d>(input)), Tensor<double, dn<b>, up<c>>>);

    Tensor<double, dn<b>, up<c>> outer_expected;
    Tensor<double, up<a>, dn<d>> middle_expected;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                outer_expected(i, j) += input(k, i, j, k);
                middle_expected(i, j) += input(i, k, k, j);
            }
        }
    }
    check("trace outer axes", trace<a, d>(input), outer_expected);
    check("trace reversed axes", trace<d, a>(input), outer_expected);
    check("trace middle axes", trace<b, c>(input), middle_expected);

    Tensor<double> double_expected;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double_expected() += input(i, j, j, i);
        }
    }
    check("successive traces to scalar", trace<b, c>(trace<a, d>(input)), double_expected);
}

} // namespace

int main() {
    for (int trial = 0; trial < 8; ++trial) {
        matrix_product();
        crossed_multiple_contractions();
        fourth_rank_product();
        scalar_result();
        outer_product();
        scalar_operands();
        traces();
    }
    if (failures != 0) {
        return 1;
    }
    std::cout << "contraction checks passed (8 deterministic random trials per case)\n";
}
