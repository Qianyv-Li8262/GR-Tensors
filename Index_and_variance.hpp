#pragma once
#include <type_traits>
struct mu {};
struct nu {};
struct rho {};
struct sigma {};
struct kappa {};
struct lambda {};
struct a {};
struct b {};
struct c {};
struct d {};

template <typename T>
struct up {};
template <typename T>
struct dn {};

template <typename T>
struct index_traits {
    using is_index = std::false_type;
};
template <typename Letter>
struct index_traits<up<Letter>> {
    using letter = Letter;
    using variance = std::true_type;
    using is_index = std::true_type;
}; // up is true, down is false
template <typename Letter>
struct index_traits<dn<Letter>> {
    using letter = Letter;
    using variance = std::false_type;
    using is_index = std::true_type;
}; // up is true, down is false

// is_same_variance<A, B>::value
// 判断两个指标是否同为上指标或同为下指标。
// 示例：
//   same_variance<up<mu>, up<nu>>::value == true
//   same_variance<up<mu>, dn<mu>>::value == false
template <typename A, typename B>
struct is_same_variance : std::false_type {};
template <typename A, typename B>
struct is_same_variance<up<A>, up<B>> : std::true_type {};
template <typename A, typename B>
struct is_same_variance<dn<A>, dn<B>> : std::true_type {};

template <typename T>
struct opposite{};
template <typename letter>
struct opposite<up<letter>>{
    using result = dn<letter>;
};
template <typename letter>
struct opposite<dn<letter>>{
    using result = up<letter>;
};