// This translation unit deliberately includes no Tensor or named-index header.
#include "contraction_executor.hpp"

#include <array>
#include <iostream>

int main() {
    using plan = contraction_plan<2, 2, contractpair<1, 0>>;
    std::array<int, 16> left{};
    std::array<int, 16> right{};
    for (std::size_t i = 0; i < 16; ++i) {
        left[i] = static_cast<int>(i) - 5;
        right[i] = 11 - static_cast<int>(i);
    }
    const auto result = contract_components<plan>(left, right);
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            int expected = 0;
            for (std::size_t inner = 0; inner < 4; ++inner) {
                expected += left[4 * row + inner] * right[4 * inner + column];
            }
            if (result[4 * row + column] != expected) {
                return 1;
            }
        }
    }
    // Unequal operand ranks and interleaved free axes exercise output decoding.
    // Reordering pairs must preserve the integer result and output axis order.
    using crossed_plan = contraction_plan<4, 3, contractpair<1, 2>, contractpair<3, 0>>;
    using reversed_plan = contraction_plan<4, 3, contractpair<3, 0>, contractpair<1, 2>>;
    std::array<int, 256> crossed_left{};
    std::array<int, 64> crossed_right{};
    for (std::size_t i = 0; i < crossed_left.size(); ++i) {
        crossed_left[i] = static_cast<int>(i % 11) - 5;
    }
    for (std::size_t i = 0; i < crossed_right.size(); ++i) {
        crossed_right[i] = static_cast<int>(i % 7) - 3;
    }
    const auto crossed = contract_components<crossed_plan>(crossed_left, crossed_right);
    const auto reversed = contract_components<reversed_plan>(crossed_left, crossed_right);
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            for (std::size_t k = 0; k < 4; ++k) {
                int expected = 0;
                for (std::size_t u = 0; u < 4; ++u) {
                    for (std::size_t v = 0; v < 4; ++v) {
                        expected += crossed_left[((i * 4 + u) * 4 + j) * 4 + v] *
                                    crossed_right[(v * 4 + k) * 4 + u];
                    }
                }
                const auto output = (i * 4 + j) * 4 + k;
                if (crossed[output] != expected || reversed[output] != expected) {
                    return 1;
                }
            }
        }
    }
    // Trace also executes with no Tensor or index-letter dependency.
    const auto traced = trace_components<trace_plan<4, 1, 3>>(crossed_left);
    const auto reversed_trace = trace_components<trace_plan<4, 3, 1>>(crossed_left);
    const auto vector_trace = trace_components<trace_plan<3, 0, 2>>(crossed_right);
    for (std::size_t i = 0; i < 4; ++i) {
        int vector_expected = 0;
        for (std::size_t j = 0; j < 4; ++j) {
            int trace_expected = 0;
            vector_expected += crossed_right[(j * 4 + i) * 4 + j];
            for (std::size_t k = 0; k < 4; ++k) {
                trace_expected += crossed_left[((i * 4 + k) * 4 + j) * 4 + k];
            }
            if (traced[4 * i + j] != trace_expected || reversed_trace[4 * i + j] != trace_expected) {
                return 1;
            }
        }
        if (vector_trace[i] != vector_expected) {
            return 1;
        }
    }
    int scalar_expected = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        scalar_expected += left[5 * i];
    }
    if (trace_components<trace_plan<2, 0, 1>>(left)[0] != scalar_expected) {
        return 1;
    }
    std::cout << "contraction and trace executor checks passed without named-index dependencies\n";
}
