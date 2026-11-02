#include "contraction_and_arithmetic.hpp"

int main() {
    Tensor<double, up<a>> left;
    Tensor<double, up<a>> right;
    auto invalid = left * right;
    return static_cast<int>(invalid.data.size());
}
