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
    std::cout << "positional executor checks passed without named-index dependencies\n";
}
