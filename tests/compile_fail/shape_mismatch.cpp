#include "contraction_executor.hpp"

int main() {
    std::array<double, 4> left{};
    std::array<double, 16> right{};
    auto invalid = contract_components<contraction_plan<2, 2, contractpair<1, 0>>>(left, right);
    return static_cast<int>(invalid.size());
}
