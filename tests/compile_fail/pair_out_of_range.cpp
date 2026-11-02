#include "contraction_plan.hpp"

using invalid_plan = contraction_plan<2, 2, contractpair<2, 0>>;
int main() {
    return static_cast<int>(invalid_plan::result_rank);
}
