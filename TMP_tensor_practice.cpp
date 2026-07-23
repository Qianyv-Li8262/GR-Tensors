#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <type_traits>
#include <utility>




// ============================================================

// Contents:

//Finished
// 1. Index tags and variance wrappers
//    指标字母、上指标 up<T>、下指标 dn<T> 

//Finished
// 2. Compile-time type_list utilities
//    编译期类型列表基础工具

//Finished
// 3. Index type traits
//    指标合法性、取字母、升降类型比较

//Finished
// 4. Tensor storage utilities
//    pow4、linear_index、pack/unpack

//Finished
// 5. Tensor core class

//Finished
// 6. Tensor index-list transformations
//    tensor_from_list、letters_from_list、find_index_type 等

//Finished
// 7. Einstein contraction type deduction
//    contract<...> 推导乘法结果指标

//Finished
// 8. Tensor multiplication implementation
//    contract_loop/free_loop/operator*

//Finished
// 9. Scalar multiplication

//Finished
// 10. trace 求迹

//Finished
// 11. rename / reorder / operator+ / operator-

//Finished
// 12. Numerical partial derivatives

//Finished
// 13. Example metric functions

//Finished
// 14. Symmetrization

//Finished
// 15. Antisymmetrization

//Finished
// 16. Field wrapper and field operations

// 17. Main















// ============================================================
// 1. Index tags and variance wrappers
// ============================================================
//
// 这里定义“抽象指标”的名字，例如 mu、nu、rho。
// 它们本身不含数值，只是类型标签，用在 Tensor 的模板参数里。
//
// up<T> 表示上指标 T，例如 up<mu> 对应 μ。
// dn<T> 表示下指标 T，例如 dn<mu> 对应 μ。
//
// 示例：
//   Tensor<double, up<mu>, dn<nu>> A;
// 表示一个 rank-2 张量 A^μ_ν。
// ============================================================

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















// ============================================================
// 2. Compile-time type_list utilities
// ============================================================
//
// type_list<Ts...> 是一个只存在于编译期的类型列表。
// 本程序大量用它来保存“指标列表”，例如：
//
//   type_list<up<mu>, dn<nu>, dn<rho>>
//
// 它不产生运行期对象，只用于模板元编程。
// ============================================================

template <typename... Ts>
struct type_list {};


// push_back<List, T>::type
// 把类型 T 插入 type_list 的末尾。
// 示例：
//   push_back<type_list<A, B>, C>::type == type_list<A, B, C>
template <typename List, typename T>
struct push_back;
template <typename... Ts, typename T>
struct push_back<type_list<Ts...>, T> {
    using type = type_list<Ts..., T>;
};


// push_front<List, T>::type
// 把类型 T 插入 type_list 的开头。
// 示例：
//   push_front<type_list<B, C>, A>::type == type_list<A, B, C>
template <typename List, typename T>
struct push_front;
template <typename... Ts, typename T>
struct push_front<type_list<Ts...>, T> {
    using type = type_list<T, Ts...>;
};


// concat_lists<L1, L2>::type
// 拼接两个 type_list。
// 示例：
//   concat_lists<type_list<A, B>, type_list<C>>::type
//   == type_list<A, B, C>
template <typename L1, typename L2>
struct concat_lists;
template <typename... As, typename... Bs>
struct concat_lists<type_list<As...>, type_list<Bs...>> {
    using type = type_list<As..., Bs...>;
};


// list_size<List>::value
// 返回 type_list 的长度。
template <typename List>
struct list_size;
template <typename... Ts>
struct list_size<type_list<Ts...>> : std::integral_constant<std::size_t, sizeof...(Ts)> {};


// contains<T, Ts...>::value
// 判断类型包 Ts... 中是否包含 T。
template <typename T, typename... Ts>
struct contains : std::false_type {};
template <typename T, typename U, typename... Ts>
struct contains<T, U, Ts...> : std::conditional_t<std::is_same_v<T, U>, std::true_type, contains<T, Ts...>> {};


// contains_in_list<T, List>::value
// 判断 type_list 中是否包含类型 T。
template <typename T, typename List>
struct contains_in_list;
template <typename T, typename... Ts>
struct contains_in_list<T, type_list<Ts...>> : contains<T, Ts...> {};


// remove_first<Target, Ts...>::type
// 从类型包 Ts... 中删除第一个 Target，返回 type_list。
//
// 示例：
//   remove_first<B, A, B, C, B>::type == type_list<A, C, B>
template <typename Target, typename... Ts>
struct remove_first;
template <typename Target>
struct remove_first<Target> {
    using type = type_list<>;
};
template <typename Target, typename Head, typename... Tail>
struct remove_first<Target, Head, Tail...> {
    using type = std::conditional_t<std::is_same_v<Target, Head>, type_list<Tail...>,
                                    typename push_front<typename remove_first<Target, Tail...>::type, Head>::type>;
};


// remove_first_in_list<Target, List>::type
// remove_first 的 type_list 版本。
template <typename Target, typename List>
struct remove_first_in_list;
template <typename Target, typename... Ts>
struct remove_first_in_list<Target, type_list<Ts...>> {
    using type = typename remove_first<Target, Ts...>::type;
};


// list1_minus_list2<List1, List2>::result
// 从 List1 中依次删除 List2 中出现的元素。
// 注意：这里只删除“第一个匹配项”，所以它适合处理可能含重复项的指标列表。
//
// 用途：
//   在张量乘法中，result_indices 是自由指标；
//   all_indices - result_indices 得到参与缩并的指标出现项。
template <typename List1, typename List2>
struct list1_minus_list2 {};
template <typename... List1>
struct list1_minus_list2<type_list<List1...>, type_list<>> {
    using result = type_list<List1...>;
};
template <typename... List1, typename Head, typename... Tail>
struct list1_minus_list2<type_list<List1...>, type_list<Head, Tail...>> {
    using result = typename list1_minus_list2<typename remove_first_in_list<Head, type_list<List1...>>::type,
                                              type_list<Tail...>>::result;
};


// unique_list<List>::type
// 对 type_list 去重，保留每个类型的一份。
//
// 用途：
//   缩并指标会出现一上一下两次，例如 up<mu>, dn<mu>。
//   提取字母后可能得到 type_list<mu, mu>，需要 unique 成 type_list<mu>。
template <typename List>
struct unique_list;
template <>
struct unique_list<type_list<>> {
    using type = type_list<>;
};
template <typename Head, typename... Tail>
struct unique_list<type_list<Head, Tail...>> {
private:
    using tail_unique = typename unique_list<type_list<Tail...>>::type;

public:
    using type = std::conditional_t<contains_in_list<Head, tail_unique>::value, tail_unique,
                                    typename push_front<tail_unique, Head>::type>;
};


// nth_type<N, List>::type
// 取 type_list 中第 N 个类型。
template <std::size_t N, typename List>
struct nth_type;
template <std::size_t N, typename Head, typename... Tail>
struct nth_type<N, type_list<Head, Tail...>> {
    using type = typename nth_type<N - 1, type_list<Tail...>>::type;
};
template <typename Head, typename... Tail>
struct nth_type<0, type_list<Head, Tail...>> {
    using type = Head;
};















// ============================================================
// 3. Index type traits
// ============================================================
//
// 这些 helper 用于识别 up<T>/dn<T>，取出里面的字母 T，
// 以及判断两个指标是否同为上指标或同为下指标。
// ============================================================


// get_letter<Index>::letter
// 从 up<T> 或 dn<T> 中取出字母 T。
// 示例：
//   get_letter<up<mu>>::letter == mu
template <typename T>
struct get_letter {};
template <typename T>
struct get_letter<up<T>> {
    using letter = T;
};
template <typename T>
struct get_letter<dn<T>> {
    using letter = T;
};

// same_variance<A, B>::value
// 判断两个指标是否同为上指标或同为下指标。
// 示例：
//   same_variance<up<mu>, up<nu>>::value == true
//   same_variance<up<mu>, dn<mu>>::value == false
template <typename A, typename B>
struct same_variance : std::false_type {};
template <typename A, typename B>
struct same_variance<up<A>, up<B>> : std::true_type {};
template <typename A, typename B>
struct same_variance<dn<A>, dn<B>> : std::true_type {};


// is_index<T>::value
// 判断 T 是否为合法指标类型 up<X> 或 dn<X>。
template <typename T>
struct is_index : std::false_type {};
template <typename T>
struct is_index<up<T>> : std::true_type {};
template <typename T>
struct is_index<dn<T>> : std::true_type {};


// has_duplicate<Ts...>::value
// 判断类型包中是否有重复类型。
// 在 Tensor 中用于检查指标字母不能重复，例如不能写：
//   Tensor<double, up<mu>, dn<mu>>
// 因为单个张量内部不允许重复字母；缩并应通过乘法或 trace 完成。
template <typename... Ts>
struct has_duplicate : std::false_type {};
template <typename T, typename... Rest>
struct has_duplicate<T, Rest...>
    : std::conditional_t<contains<T, Rest...>::value, std::true_type, has_duplicate<Rest...>> {};


// contains_compatible_index<Target, List>::value
// 判断 List 中是否存在和 Target “同字母且同升降类型”的指标。
//
// 用途：
//   reorder 和加减法要求两个张量具有相同自由指标集合，
//   但顺序可以不同。
template <typename Target, typename List>
struct contains_compatible_index;
template <typename Target, typename... Ts>
struct contains_compatible_index<Target, type_list<Ts...>>
    : std::bool_constant<((std::is_same_v<typename get_letter<Target>::letter, typename get_letter<Ts>::letter> &&
                           same_variance<Target, Ts>::value) ||
                          ...)> {};


// is_reorder_compatible<NewList, OldList>::value
// 判断 OldList 是否可以重排成 NewList。
// 要求：
//   1. 字母相同；
//   2. 上/下指标位置相同；
//   3. 顺序可以不同。
template <typename NewList, typename OldList>
struct is_reorder_compatible;

template <typename... NewIndices, typename OldList>
struct is_reorder_compatible<type_list<NewIndices...>, OldList>
    : std::bool_constant<(contains_compatible_index<NewIndices, OldList>::value && ...)> {};















// ============================================================
// 4. Tensor storage utilities
// ============================================================
//
// 本程序固定维数为 4，因此 rank-r 张量有 4^r 个分量。
// data 使用一维 std::array 存储，多重指标通过 4 进制方式线性化。
// ============================================================


// factorial(n)
// 编译期/运行期均可用的小阶乘函数。
// 用于 symm/antisymm 的归一化系数 1/n!。
constexpr std::size_t factorial(std::size_t n) {
    std::size_t r = 1;
    for (std::size_t i = 2; i <= n; ++i) {
        r *= i;
    }
    return r;
}

// pow4(rank)
// 返回 4^rank。
// 程序默认四维时空。
constexpr std::size_t pow4(std::size_t rank) {
    std::size_t r = 1;
    for (std::size_t i = 0; i < rank; ++i) {
        r *= 4;
    }
    return r;
}


// linear_index(i, j, k, ...)
// 把多重指标转换为 data[] 的线性下标。
//
// 示例：rank=3 时
//   linear_index(i,j,k) == ((i * 4) + j) * 4 + k
template <typename... Ints>
constexpr std::size_t linear_index(Ints... idxs) {
    std::array<std::size_t, sizeof...(Ints)> arr{static_cast<std::size_t>(idxs)...};
    std::size_t result = 0;
    for (std::size_t i = 0; i < arr.size(); ++i) {
        result = result * 4 + arr[i];
    }
    return result;
}


// unpack_index<Rank>(lin)
// 把线性下标 lin 还原成 rank 个 0..3 的指标。
//
// 用途：
//   reorder 时枚举新张量的线性下标，再映射回旧张量下标。
template <std::size_t Rank>
std::array<std::size_t, Rank> unpack_index(std::size_t lin) {
    std::array<std::size_t, Rank> idx{};

    for (std::size_t rev = 0; rev < Rank; ++rev) {
        std::size_t d = Rank - 1 - rev;
        idx[d] = lin % 4;
        lin /= 4;
    }   

    return idx;
}


// pack_index(idx)
// unpack_index 的逆操作。
template <std::size_t Rank>
std::size_t pack_index(const std::array<std::size_t, Rank>& idx) {
    std::size_t lin = 0;

    for (std::size_t i = 0; i < Rank; ++i) {
        lin = lin * 4 + idx[i];
    }

    return lin;
}















// ============================================================
// 5. Tensor core class
// ============================================================
//
// Tensor<datatype, Indices...>
//
// datatype:
//   分量类型，例如 double。
//
// Indices...:
//   指标类型，例如 up<mu>, dn<nu>。
//
// 示例：
//   Tensor<double, up<mu>> V;              // V^μ
//   Tensor<double, dn<mu>, dn<nu>> g;      // g_{μν}
//   Tensor<double> scalar;                 // 标量，rank = 0
//
// 访问分量：
//   V(0) = 1.0;
//   g(1, 2) = 3.0;
//
// 注意：
//   当前维度固定为 4，因此每个指标取值 0,1,2,3。
// ============================================================
template <typename datatype, typename... Indices>
struct Tensor {
    // 所有指标都必须是 up<T> 或 dn<T>。
    static_assert((is_index<Indices>::value && ...), "Tensor indices must all be up<T> or dn<T>");

    // 同一个 Tensor 类型内部不允许重复字母。
    // 例如 Tensor<double, up<mu>, dn<mu>> 不允许。
    // 如果需要缩并，请使用 trace 或张量乘法。
    static_assert(!has_duplicate<typename get_letter<Indices>::letter...>::value, "Tensor indices must not repeat");

    static constexpr std::size_t rank = sizeof...(Indices);

    // 一维连续存储，共 4^rank 个分量。
    std::array<datatype, pow4(rank)> data{};

    Tensor() = default;

    // 用同一个初值填满所有分量。
    explicit Tensor(datatype init_value) {
        data.fill(init_value);
    }

    // 非 const 分量访问。
    template <typename... Ints>
    datatype& operator()(Ints... idxs) {
        static_assert(sizeof...(Ints) == rank, "number of indices must match tensor rank");
        return data[linear_index(idxs...)];
    }

    // const 分量访问。
    template <typename... Ints>
    const datatype& operator()(Ints... idxs) const {
        static_assert(sizeof...(Ints) == rank, "number of indices must match tensor rank");
        return data[linear_index(idxs...)];
    }
};















// ============================================================
// 6.Tensor index-list transformations
// ============================================================
template <typename datatype, typename List>
struct tensor_from_list;
template <typename datatype, typename... Indices>
struct tensor_from_list<datatype, type_list<Indices...>> {
    using type = Tensor<datatype, Indices...>;
};


// letters_from_list<type_list<up<mu>, dn<nu>>>::type
// 得到 type_list<mu, nu>。
template <typename List>
struct letters_from_list;
template <typename... Ts>
struct letters_from_list<type_list<Ts...>> {
    using type = type_list<typename get_letter<Ts>::letter...>;
};


// find_indices<N, Letter, IndexList>::value
// 在指标列表中查找某个字母的位置。
// IndexList 中元素是 up<T>/dn<T>。
//
// 示例：
//   find_indices<0, nu, type_list<up<mu>, dn<nu>>>::value == 1
template <int N, typename A, typename B>
struct find_indices;
template <int N, typename letter, typename head, typename... list>
struct find_indices<N, letter, type_list<head, list...>>
    : std::conditional_t<std::is_same_v<letter, typename get_letter<head>::letter>, std::integral_constant<int, N>,
                         find_indices<N + 1, letter, type_list<list...>>> {};


// find_index_type<Letter, IndexList>::type
// 在指标列表中查找字母 Letter 对应的完整指标类型。
// 示例：
//   find_index_type<mu, type_list<up<mu>, dn<nu>>>::type == up<mu>
template <typename Letter, typename List>
struct find_index_type;
template <typename Letter, typename Head, typename... Tail>
struct find_index_type<Letter, type_list<Head, Tail...>>
    : std::conditional_t<std::is_same_v<Letter, typename get_letter<Head>::letter>, std::type_identity<Head>,
                         find_index_type<Letter, type_list<Tail...>>> {};


// find_in_letters<N, Letter, LetterList>::value
// 在“纯字母列表”中查找 Letter 的位置。
// 与 find_indices 不同，这里的 List 是 type_list<mu, nu, ...>，
// 而不是 type_list<up<mu>, dn<nu>, ...>。
template <int N, typename Letter, typename List>
struct find_in_letters;
template <int N, typename Letter, typename Head, typename... Tail>
struct find_in_letters<N, Letter, type_list<Head, Tail...>>
    : std::conditional_t<std::is_same_v<Letter, Head>, std::integral_constant<int, N>,
                         find_in_letters<N + 1, Letter, type_list<Tail...>>> {};















// ============================================================
// 7. Einstein contraction type deduction
// ============================================================
//
// contract<type_list<>, Indices...>::result
//
// 给定两个张量乘法时拼接后的指标列表：
//   LIndices..., RIndices...
//
// 按 Einstein 约定：
//   如果同一个字母出现一上一下，则这对指标被缩并，
//   不出现在结果张量中。
//
//   如果同一个字母出现两次但同为上指标或同为下指标，
//   则是非法表达式。
//
// 示例：
//   Tensor<double, up<mu>, dn<nu>> A;
//   Tensor<double, up<nu>, dn<rho>> B;
//
//   A * B 的拼接指标是：
//     up<mu>, dn<nu>, up<nu>, dn<rho>
//
//   dn<nu> 与 up<nu> 缩并，结果指标为：
//     up<mu>, dn<rho>
//
//   所以 A * B 返回 Tensor<double, up<mu>, dn<rho>>。
template <typename Accum, typename List>
struct contract_from_list;


// chk_contract<T, U, Ts...>::type
//
// 从 U, Ts... 中查找是否存在和 T 同字母的指标。
//
// 返回值含义：
//   std::true_type:
//     没找到同字母指标，T 是自由指标，应保留。
//
//   std::false_type:
//     找到了同字母且同升降类型的指标，例如 up<mu> 和 up<mu>，非法。
//
//   某个指标类型 X:
//     找到了同字母且异升降类型的指标，例如 up<mu> 和 dn<mu>，
//     T 与 X 构成一对缩并指标，都应从结果中去掉。
template <typename T, typename... Ts>
struct chk_contract {
    using type = std::true_type;
};
template <typename T, typename U, typename... Ts>
struct chk_contract<T, U, Ts...> {
    using type = std::conditional_t<std::is_same_v<typename get_letter<T>::letter, typename get_letter<U>::letter>,
                                    std::conditional_t<std::is_same_v<T, U>, std::false_type, U>,
                                    typename chk_contract<T, Ts...>::type>;
};


// contract<Accum, Ts...>::result
//
// Accum:
//   当前已经确定要保留的自由指标列表。
//
// Ts...:
//   尚未处理的指标列表。
//
// 递归逻辑：
//   每次取第一个指标 T，向后寻找同字母指标。
//   - 若没有找到：T 是自由指标，加入 Accum。
//   - 若找到同字母同升降：static_assert 报错。
//   - 若找到同字母异升降：T 与匹配项缩并，二者都不加入 Accum。
template <typename... Ts>
struct contract {};
template <typename... AccumTs>
struct contract<type_list<AccumTs...>> {
    using result = type_list<AccumTs...>;
};
template <typename... AccumTs, typename T>
struct contract<type_list<AccumTs...>, T> {
    using result = type_list<AccumTs..., T>;
};
template <typename... accum, typename T, typename U, typename... Ts>
struct contract<type_list<accum...>, T, U, Ts...> {
private:
    using intermediate = typename chk_contract<T, U, Ts...>::type;
    static_assert(!std::is_same_v<intermediate, std::false_type>,
                  "Invalid contraction due to duplicate upper/lower indices");
    using next_accum =
        std::conditional_t<std::is_same_v<intermediate, std::true_type>, type_list<accum..., T>, type_list<accum...>>;

public:
    using result = std::conditional_t<
        std::is_same_v<intermediate, std::true_type>, typename contract<next_accum, U, Ts...>::result,
        typename contract_from_list<next_accum, typename remove_first<intermediate, U, Ts...>::type>::result>;
    /*
    递归说明：

    intermediate 有三种情况：

    1. std::true_type
       表示 T 在剩余指标里没有同字母伙伴。
       因此 T 是自由指标，加入 next_accum，然后继续处理 U, Ts...

    2. std::false_type
       表示 T 找到了同字母且同升降的指标。
       例如 up<mu> 与 up<mu>。
       这是非法缩并，触发 static_assert。

    3. 某个具体指标类型 intermediate
       表示 T 找到了同字母且异升降的指标。
       例如 T = up<mu>, intermediate = dn<mu>。
       这两者构成一对缩并指标，因此都不加入结果。
       接着从剩余列表中删除 intermediate，继续递归。
    */
};
// TODO: refactor contract{} due to overcomplicated logic


template <typename Accum, typename... Rest>
struct contract_from_list<Accum, type_list<Rest...>> {
    using result = typename contract<Accum, Rest...>::result;
};















// ============================================================
// 8. Tensor multiplication with Einstein contraction
// ============================================================
//
// operator*(Tensor L, Tensor R)
//
// 实现 Einstein 求和约定：
//   - 重复出现的一上一下同字母指标自动缩并。
//   - 未缩并指标作为结果张量的自由指标保留。
//   - 同字母同升降重复会在编译期报错。
//
// 示例：
//   Tensor<double, up<mu>, dn<nu>> A;
//   Tensor<double, up<nu>, dn<rho>> B;
//   auto C = A * B;
//   // C 的类型为 Tensor<double, up<mu>, dn<rho>>
// ============================================================


// contract_loop<ContractLetters, AllLetters>
//
// 内层缩并循环。
// ContractLetters 是需要求和的缩并字母列表。
// 对每个缩并字母生成一层 for(i=0;i<4;++i)。
//
// 当所有缩并变量都赋值后，执行一次：
//   sum += L(...) * R(...);
template <typename ContractLetters, typename AllLetters>
struct contract_loop;
// 终止：所有缩并变量都定好了，做一次乘加
template <typename AllLetters>
struct contract_loop<type_list<>, AllLetters> {
    template <typename datatype, typename... LIdx, typename... RIdx, std::size_t N>
    static void exec(const Tensor<datatype, LIdx...>& L, const Tensor<datatype, RIdx...>& R,
                     std::array<std::size_t, N>& vals, datatype& sum) {
        sum += L(vals[find_in_letters<0, typename get_letter<LIdx>::letter, AllLetters>::value]...) *
               R(vals[find_in_letters<0, typename get_letter<RIdx>::letter, AllLetters>::value]...);
    }
};
// 递归：剥掉一个缩并字母，生成一层 for
template <typename Head, typename... Tail, typename AllLetters>
struct contract_loop<type_list<Head, Tail...>, AllLetters> {
    template <typename datatype, typename... LIdx, typename... RIdx, std::size_t N>
    static void exec(const Tensor<datatype, LIdx...>& L, const Tensor<datatype, RIdx...>& R,
                     std::array<std::size_t, N>& vals, datatype& sum) {
        constexpr std::size_t pos = find_in_letters<0, Head, AllLetters>::value;
        for (std::size_t i = 0; i < 4; ++i) {
            vals[pos] = i;
            contract_loop<type_list<Tail...>, AllLetters>::exec(L, R, vals, sum);
        }
    }
};


// free_loop<FreeLetters, AllLetters, ContractLetters>
//
// 外层自由指标循环。
// FreeLetters 是结果张量的自由字母列表。
// 对每个自由字母生成一层 for(i=0;i<4;++i)。
//
// 当自由指标全部赋值后，调用 contract_loop 完成缩并求和，
// 并把结果写入 C(...)。
template <typename FreeLetters, typename AllLetters, typename ContractLetters>
struct free_loop;
// 终止：所有自由变量都定好了，执行缩并并写回结果
template <typename AllLetters, typename ContractLetters>
struct free_loop<type_list<>, AllLetters, ContractLetters> {
    template <typename datatype, typename... LIdx, typename... RIdx, typename... ResIdx, std::size_t N>
    static void exec(const Tensor<datatype, LIdx...>& L, const Tensor<datatype, RIdx...>& R,
                     Tensor<datatype, ResIdx...>& C, std::array<std::size_t, N>& vals) {
        datatype sum = 0;
        contract_loop<ContractLetters, AllLetters>::exec(L, R, vals, sum);
        C(vals[find_in_letters<0, typename get_letter<ResIdx>::letter, AllLetters>::value]...) = sum;
    }
};
// 递归：剥掉一个自由字母，生成一层 for
template <typename Head, typename... Tail, typename AllLetters, typename ContractLetters>
struct free_loop<type_list<Head, Tail...>, AllLetters, ContractLetters> {
    template <typename datatype, typename... LIdx, typename... RIdx, typename... ResIdx, std::size_t N>
    static void exec(const Tensor<datatype, LIdx...>& L, const Tensor<datatype, RIdx...>& R,
                     Tensor<datatype, ResIdx...>& C, std::array<std::size_t, N>& vals) {
        constexpr std::size_t pos = find_in_letters<0, Head, AllLetters>::value;
        for (std::size_t i = 0; i < 4; ++i) {
            vals[pos] = i;
            free_loop<type_list<Tail...>, AllLetters, ContractLetters>::exec(L, R, C, vals);
        }
    }
};


template <typename datatype, typename... LIndices, typename... RIndices>
auto operator*(const Tensor<datatype, LIndices...>& L, const Tensor<datatype, RIndices...>& R) {
    // 1. 根据 Einstein 约定推导结果指标。
    //    会自动移除一上一下的重复字母，并检查非法重复。
    using result_indices = typename contract<type_list<>, LIndices..., RIndices...>::result;
    using result_tensor = typename tensor_from_list<datatype, result_indices>::type;

    // 2. 自由字母就是结果张量中的字母。
    using free_letters = typename letters_from_list<result_indices>::type;

    // 3. 计算缩并字母。
    //    all_indices 包含左右张量所有指标；
    //    去掉 result_indices 后，剩下的就是被缩并掉的指标出现项。
    using all_indices = type_list<LIndices..., RIndices...>;
    using contracted_occ = typename list1_minus_list2<all_indices, result_indices>::result;
    using contracted_raw = typename letters_from_list<contracted_occ>::type;
    using contract_letters = typename unique_list<contracted_raw>::type;

    // 4. 统一变量表：
    //    vals[] 中同时保存自由指标值和缩并指标值。
    //    all_letters = [自由字母..., 缩并字母...]
    using all_letters = typename concat_lists<free_letters, contract_letters>::type;
    constexpr std::size_t num_all = list_size<all_letters>::value;

    // 5. 执行嵌套循环。
    result_tensor C{};
    std::array<std::size_t, num_all> vals{};
    free_loop<free_letters, all_letters, contract_letters>::exec(L, R, C, vals);

    return C;
}















// ============================================================
// 9. Scalar multiplication
// ============================================================

template <typename datatype, typename... Indices>
auto operator*(datatype scalar, const Tensor<datatype, Indices...>& T) {
    Tensor<datatype, Indices...> C;
    for (std::size_t i = 0; i < C.data.size(); ++i)
        C.data[i] = scalar * T.data[i];
    return C;
}


// Tensor * scalar
// 转发到 scalar * Tensor。
template <typename datatype, typename... Indices>
auto operator*(const Tensor<datatype, Indices...>& T, datatype scalar) {
    return scalar * T;
}















// ============================================================
// 10. Trace / self contraction
// ============================================================
//
// trace<LetterA, LetterB>(T)
//
// 对同一个张量内部的两个指标做缩并。
// LetterA 与 LetterB 必须一上一下，否则编译期报错。
//
// 示例：
//   Tensor<double, up<mu>, dn<nu>> A;
//   auto tr = trace<mu, nu>(A);
//   // tr 是 Tensor<double> 标量，值为 sum_i A^i_i。
//
// 又如：
//   Tensor<double, up<rho>, dn<mu>, dn<nu>> Gamma;
//   auto V = trace<rho, mu>(Gamma);
//   // 对 rho 和 mu 缩并，结果保留 dn<nu>。


// trace_loop<FreeLetters, AllLetters, LetterA, LetterB>
//
// 先枚举所有自由指标；
// 当自由指标固定后，对 LetterA 和 LetterB 同步取 i=0..3，
// 计算 sum_i T(... i ... i ...)，写入结果。
template <typename FreeLetters, typename AllLetters, typename LetterA, typename LetterB>
struct trace_loop;
// 自由指标循环终止：开始对 LetterA 和 LetterB 做自缩并
template <typename AllLetters, typename LetterA, typename LetterB>
struct trace_loop<type_list<>, AllLetters, LetterA, LetterB> {
    template <typename datatype, typename... Indices, typename... ResIdx, std::size_t N>
    static void exec(const Tensor<datatype, Indices...>& T, Tensor<datatype, ResIdx...>& result,
                     std::array<std::size_t, N>& vals) {
        constexpr std::size_t posA = find_in_letters<0, LetterA, AllLetters>::value;
        constexpr std::size_t posB = find_in_letters<0, LetterB, AllLetters>::value;

        datatype sum{};

        for (std::size_t i = 0; i < 4; ++i) {
            vals[posA] = i;
            vals[posB] = i;

            sum += T(vals[find_in_letters<0, typename get_letter<Indices>::letter, AllLetters>::value]...);
        }

        result(vals[find_in_letters<0, typename get_letter<ResIdx>::letter, AllLetters>::value]...) = sum;
    }
};
// 自由指标递归：每剥一个自由字母，生成一层 for
template <typename Head, typename... Tail, typename AllLetters, typename LetterA, typename LetterB>
struct trace_loop<type_list<Head, Tail...>, AllLetters, LetterA, LetterB> {
    template <typename datatype, typename... Indices, typename... ResIdx, std::size_t N>
    static void exec(const Tensor<datatype, Indices...>& T, Tensor<datatype, ResIdx...>& result,
                     std::array<std::size_t, N>& vals) {
        constexpr std::size_t pos = find_in_letters<0, Head, AllLetters>::value;

        for (std::size_t i = 0; i < 4; ++i) {
            vals[pos] = i;
            trace_loop<type_list<Tail...>, AllLetters, LetterA, LetterB>::exec(T, result, vals);
        }
    }
};


template <typename LetterA, typename LetterB, typename datatype, typename... Indices>
auto trace(const Tensor<datatype, Indices...>& T) {
    using idx_list = type_list<Indices...>;

    using idxA = typename find_index_type<LetterA, idx_list>::type;
    using idxB = typename find_index_type<LetterB, idx_list>::type;
    static_assert(!same_variance<idxA, idxB>::value, "Invalid contraction due to duplicate upper/lower indices");

    using step1 = typename remove_first_in_list<idxA, idx_list>::type;
    using result_indices = typename remove_first_in_list<idxB, step1>::type;

    using result_tensor = typename tensor_from_list<datatype, result_indices>::type;

    using free_letters = typename letters_from_list<result_indices>::type;
    using all_letters = typename concat_lists<free_letters, type_list<LetterA, LetterB>>::type;

    constexpr std::size_t num_all = list_size<all_letters>::value;

    result_tensor result{};
    std::array<std::size_t, num_all> vals{};

    trace_loop<free_letters, all_letters, LetterA, LetterB>::exec(T, result, vals);

    return result;
}















// ============================================================
// 11. Rename, reorder, addition and subtraction
// ============================================================

template <typename OldList, typename... NewIndices, std::size_t Rank, std::size_t... Is>
void fill_old_index_from_new_index_impl(const std::array<std::size_t, Rank>& new_idx,
                                        std::array<std::size_t, Rank>& old_idx, std::index_sequence<Is...>) {
    ((old_idx[find_indices<0, typename get_letter<NewIndices>::letter, OldList>::value] = new_idx[Is]), ...);
}

template <typename OldList, typename... NewIndices, std::size_t Rank>
void fill_old_index_from_new_index(const std::array<std::size_t, Rank>& new_idx,
                                   std::array<std::size_t, Rank>& old_idx) {
    fill_old_index_from_new_index_impl<OldList, NewIndices...>(new_idx, old_idx, std::make_index_sequence<Rank>{});
}

// rename<NewIndices...>(T)
//
// 只改变指标“名字”，不改变数据排列。
// 要求：
//   - rank 相同；
//   - 每个位置的升降类型不变；
//   - 只允许 up<X> -> up<Y> 或 dn<X> -> dn<Y>。
//
// 示例：
//   Tensor<double, up<mu>, dn<nu>> A;
//   auto B = rename<up<rho>, dn<sigma>>(A);
//   // B 的数据和 A 完全相同，类型变为 Tensor<double, up<rho>, dn<sigma>>。
//
// 注意：
//   rename 不会交换轴！
//   如果要交换指标顺序，请使用 reorder。
template <typename... NewIndices, typename datatype, typename... OldIndices>
auto rename(const Tensor<datatype, OldIndices...>& T) {
    static_assert(sizeof...(NewIndices) == sizeof...(OldIndices), "rename requires same rank");
    static_assert((is_index<NewIndices>::value && ...), "rename target indices must all be up<T> or dn<T>");
    static_assert((same_variance<OldIndices, NewIndices>::value && ...),
                  "rename may only change index letters, not raise/lower index positions");

    Tensor<datatype, NewIndices...> result;
    result.data = T.data;
    return result;
}


// reorder<NewIndices...>(T)
//
// 按指标名字重排张量轴。
// 要求 NewIndices 与 OldIndices 具有相同字母和相同升降类型，但顺序可以不同。
//
// 示例：
//   Tensor<double, up<mu>, dn<nu>> A;
//   auto B = reorder<dn<nu>, up<mu>>(A);
//   // B_{ν}^{μ} 的数据满足 B(j,i) = A(i,j)。
//
// 和 rename 的区别：
//   rename 只改类型名，不动 data。
//   reorder 会真的重排 data。
template <typename... NewIndices, typename datatype, typename... OldIndices>
auto reorder(const Tensor<datatype, OldIndices...>& T) {
    static_assert(sizeof...(NewIndices) == sizeof...(OldIndices), "reorder requires same rank");
    static_assert((is_index<NewIndices>::value && ...), "reorder target indices must all be up<T> or dn<T>");

    using old_list = type_list<OldIndices...>;
    using new_list = type_list<NewIndices...>;

    static_assert(is_reorder_compatible<new_list, old_list>::value,
                  "reorder requires same index letters with same variance");

    constexpr std::size_t rank = sizeof...(OldIndices);

    Tensor<datatype, NewIndices...> result{};

    constexpr std::size_t total_size = pow4(rank);

    for (std::size_t lin_new = 0; lin_new < total_size; ++lin_new) {
        auto new_idx = unpack_index<rank>(lin_new);

        std::array<std::size_t, rank> old_idx{};

        fill_old_index_from_new_index<old_list, NewIndices...>(new_idx, old_idx);

        std::size_t lin_old = pack_index<rank>(old_idx);

        result.data[lin_new] = T.data[lin_old];
    }

    return result;
}


template <typename datatype, typename... LIndices, typename... RIndices>
auto operator+(const Tensor<datatype, LIndices...>& A, const Tensor<datatype, RIndices...>& B) {
    static_assert(sizeof...(LIndices) == sizeof...(RIndices), "operator+ requires tensors with same rank");

    using left_list = type_list<LIndices...>;
    using right_list = type_list<RIndices...>;

    static_assert(is_reorder_compatible<left_list, right_list>::value,
                  "operator+ requires same free indices with same variance");

    auto B_reordered = reorder<LIndices...>(B);

    Tensor<datatype, LIndices...> C;

    for (std::size_t i = 0; i < C.data.size(); ++i) {
        C.data[i] = A.data[i] + B_reordered.data[i];
    }

    return C;
}


template <typename datatype, typename... LIndices, typename... RIndices>
auto operator-(const Tensor<datatype, LIndices...>& A, const Tensor<datatype, RIndices...>& B) {
    static_assert(sizeof...(LIndices) == sizeof...(RIndices), "operator- requires tensors with same rank");

    using left_list = type_list<LIndices...>;
    using right_list = type_list<RIndices...>;

    static_assert(is_reorder_compatible<left_list, right_list>::value,
                  "operator- requires same free indices with same variance");

    auto B_reordered = reorder<LIndices...>(B);

    Tensor<datatype, LIndices...> C;

    for (std::size_t i = 0; i < C.data.size(); ++i) {
        C.data[i] = A.data[i] - B_reordered.data[i];
    }

    return C;
}















// ============================================================
// 12. Numerical partial derivatives
// ============================================================
//
// coord<Letter> 表示四维坐标 x^Letter。
// 实际类型是 Tensor<double, up<Letter>>。
//
// partial<DerivLetter>(field, x, h)
// 对张量场 field 在点 x 处做数值偏导，返回多一个下指标的张量：
//
//   field(x) : Tensor<T, Indices...>
//   partial<rho>(field, x) : Tensor<T, dn<rho>, Indices...>
//
// 这里使用 8 阶中心差分。
// ============================================================
template <typename Letter>
using coord = Tensor<double, up<Letter>>;


// partial_impl
//
// 内部实现函数。
// sample 参数只用于推导 field(x) 的返回张量类型。
//
// 对每个坐标方向 a = 0..3：
//   result[a, ...] = ∂_a field(...)
//
// 使用 8 阶中心差分：
//   f'(x) ≈ Σ c_k f(x + k h) / h, k = ±1, ±2, ±3, ±4。
template <typename DerivLetter, typename CoordLetter, typename F, typename datatype, typename... Indices>
auto partial_impl(const F& field, const coord<CoordLetter>& x, double h, const Tensor<datatype, Indices...>&) {
    Tensor<datatype, dn<DerivLetter>, Indices...> result{};

    constexpr std::size_t orig_size = pow4(sizeof...(Indices));

    // 8阶中心差分系数 (derivative=1, accuracy=8)
    // stencil: -4, -3, -2, -1, +1, +2, +3, +4
    // coeffs:  1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280
    // 注意: f(x+0) 的系数为 0，所以不需要计算

    struct StencilPoint {
        int offset;
        double coeff;
    };

    constexpr StencilPoint stencil[] = {
        {-4, 1.0 / 280.0}, {-3, -4.0 / 105.0}, {-2, 1.0 / 5.0},  {-1, -4.0 / 5.0},
        {1, 4.0 / 5.0},    {2, -1.0 / 5.0},    {3, 4.0 / 105.0}, {4, -1.0 / 280.0},
    };

    for (std::size_t a = 0; a < 4; ++a) {
        // 先清零这一片
        for (std::size_t j = 0; j < orig_size; ++j) {
            result.data[a * orig_size + j] = 0.0;
        }

        for (const auto& [offset, coeff] : stencil) {
            coord<CoordLetter> xs = x;
            xs(a) += offset * h;
            auto fs = field(xs);

            for (std::size_t j = 0; j < orig_size; ++j) {
                result.data[a * orig_size + j] += coeff * fs.data[j];
            }
        }

        // 除以 h
        for (std::size_t j = 0; j < orig_size; ++j) {
            result.data[a * orig_size + j] /= h;
        }
    }

    return result;
}


// partial<DerivLetter>(field, x, h)
//
// 对 field 求偏导。
// 返回值类型自动从 field(x) 推导。
//
// 示例：
//   auto dg = partial<rho>(g, x);
//   // 如果 g(x) 是 g_{μν}，则 dg 是 ∂_ρ g_{μν}，
//   // 类型 Tensor<double, dn<rho>, dn<mu>, dn<nu>>。
template <typename DerivLetter, typename CoordLetter, typename F>
auto partial(const F& field, const coord<CoordLetter>& x, double h = 1e-2) {
    auto sample = field(x);
    return partial_impl<DerivLetter>(field, x, h, sample);
}


// partial_up<UpLetter, DummyLetter>(field, inv_metric, x, h)
//
// 先计算 lower = ∂_{DummyLetter} field，
// 再用逆度规把导数指标升上去。
//
// 注意：
//   这里 UpLetter 目前没有直接参与类型推导，实际返回指标来自 inv_metric * lower。
//   如果以后要严格指定上指标名字，可以再加 rename。
template <typename UpLetter, typename DummyLetter, typename CoordLetter, typename F, typename InvMetricFunc>
auto partial_up(const F& field, const InvMetricFunc& inv_metric, const coord<CoordLetter>& x, double h = 1e-2) {
    auto lower = partial<DummyLetter>(field, x, h);
    auto ginv = rename<up<UpLetter>, up<DummyLetter>>(inv_metric(x));

    return ginv * lower;
}















// ============================================================
// 13. Metric functions
// ============================================================


template <typename a, typename b, typename letter, typename datatype>
Tensor<datatype, dn<a>, dn<b>> metric(const Tensor<datatype, up<letter>>& x) {
    Tensor<datatype, dn<a>, dn<b>> g; // 默认全0
    double r = x(1);
    double theta = x(2);
    double rs = 2.0; // 2GM/c², 取 M=G=c=1
    double f = 1.0 - rs / r;
    g(0, 0) = -f;
    g(1, 1) = 1.0 / f;
    g(2, 2) = r * r;
    g(3, 3) = r * r * std::sin(theta) * std::sin(theta);
    return g;
}

template <typename A, typename B, typename letter, typename datatype>
Tensor<datatype, up<A>, up<B>> inv_metric(const Tensor<datatype, up<letter>>& x) {
    Tensor<datatype, up<A>, up<B>> ginv{};

    double r = x(1);
    double th = x(2);
    double rs = 2.0;

    double f = 1.0 - rs / r;

    ginv(0, 0) = -1.0 / f;
    ginv(1, 1) = f;
    ginv(2, 2) = 1.0 / (r * r);
    ginv(3, 3) = 1.0 / (r * r * std::sin(th) * std::sin(th));

    return ginv;
}















// ============================================================
// 14. Symmetrization
// ============================================================
//
// symm_all(T)
// 对 T 的所有指标做完全对称化：
//
//   symm_all(T) = 1/n! * Σ_{π∈S_n} T_{π(...)}
//
// symm<Letters...>(T)
// 只对指定字母对应的指标做对称化，其余指标保持原位。
//
// 示例：
//   Tensor<double, dn<mu>, dn<nu>, dn<rho>> T;
//   auto S1 = symm_all(T);        // 对 μνρ 全对称
//   auto S2 = symm<mu, nu>(T);    // 只对 μν 对称
// ============================================================


// calc_symm_tree / loop_symm_tree
//
// 编译期生成所有指标排列。
// 对每一种排列，使用 rename 得到对应项，然后全部相加。
// symm_all_unnormalized 不除以 n!；symm_all 会除以 n!。
template <typename list1, typename list2>
struct calc_symm_tree {};
template <typename... peeled_idxs>
struct calc_symm_tree<type_list<peeled_idxs...>, type_list<>> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        return rename<peeled_idxs...>(T);
    }
};

template <typename a, typename b, typename c>
struct loop_symm_tree {};
template <typename... peeled_idxs, typename... idxs, typename choice>
struct loop_symm_tree<type_list<peeled_idxs...>, type_list<idxs...>, type_list<choice>> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        using new_peeled = typename push_back<type_list<peeled_idxs...>, choice>::type;
        using new_idxs = typename remove_first_in_list<choice, type_list<idxs...>>::type;
        auto S = calc_symm_tree<new_peeled, new_idxs>::exec(T);
        return S;
    }
};
template <typename... peeled_idxs, typename... idxs, typename choice, typename... choice_list>
struct loop_symm_tree<type_list<peeled_idxs...>, type_list<idxs...>, type_list<choice, choice_list...>> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        using new_peeled = typename push_back<type_list<peeled_idxs...>, choice>::type;
        using new_idxs = typename remove_first_in_list<choice, type_list<idxs...>>::type;
        auto S = calc_symm_tree<new_peeled, new_idxs>::exec(T);
        auto U = loop_symm_tree<type_list<peeled_idxs...>, type_list<idxs...>, type_list<choice_list...>>::exec(T);
        return S + U;
    }
};

template <typename... Peeled, typename... Rest>
struct calc_symm_tree<type_list<Peeled...>, type_list<Rest...>> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        return loop_symm_tree<type_list<Peeled...>, type_list<Rest...>, type_list<Rest...>>::exec(T);
    }
};


template <typename Index, typename LetterList>
struct index_letter_in_letters;

template <typename Index, typename... Letters>
struct index_letter_in_letters<Index, type_list<Letters...>>
    : std::bool_constant<(std::is_same_v<typename get_letter<Index>::letter, Letters> || ...)> {};

template <typename OrigList, typename LetterList>
struct indices_from_letters;

template <typename OrigList, typename... Letters>
struct indices_from_letters<OrigList, type_list<Letters...>> {
    using type = type_list<typename find_index_type<Letters, OrigList>::type...>;
};

template <typename NewList>
struct rename_from_list;

template <typename... NewIndices>
struct rename_from_list<type_list<NewIndices...>> {
    template <typename datatype, typename... OldIndices>
    static auto exec(const Tensor<datatype, OldIndices...>& T) {
        return rename<NewIndices...>(T);
    }
};

template <bool Selected, typename Head, typename SymIndices, typename PermutedSymIndices>
struct embed_one_index;

// 没被选中：保持原指标不变
template <typename Head, typename SymIndices, typename PermutedSymIndices>
struct embed_one_index<false, Head, SymIndices, PermutedSymIndices> {
    using type = Head;
};

// 被选中：在 SymIndices 中找到 Head 的字母位置，然后取 PermutedSymIndices 对应位置
template <typename Head, typename SymIndices, typename PermutedSymIndices>
struct embed_one_index<true, Head, SymIndices, PermutedSymIndices> {
private:
    using head_letter = typename get_letter<Head>::letter;

    static constexpr std::size_t pos = find_indices<0, head_letter, SymIndices>::value;

public:
    using type = typename nth_type<pos, PermutedSymIndices>::type;
};

template <typename OrigList, typename SymLetters, typename SymIndices, typename PermutedSymIndices>
struct embed_permutation;

template <typename SymLetters, typename SymIndices, typename PermutedSymIndices>
struct embed_permutation<type_list<>, SymLetters, SymIndices, PermutedSymIndices> {
    using type = type_list<>;
};

template <typename Head, typename... Tail, typename SymLetters, typename SymIndices, typename PermutedSymIndices>
struct embed_permutation<type_list<Head, Tail...>, SymLetters, SymIndices, PermutedSymIndices> {
private:
    static constexpr bool selected = index_letter_in_letters<Head, SymLetters>::value;

    using new_head = typename embed_one_index<selected, Head, SymIndices, PermutedSymIndices>::type;

    using tail_result =
        typename embed_permutation<type_list<Tail...>, SymLetters, SymIndices, PermutedSymIndices>::type;

public:
    using type = typename push_front<tail_result, new_head>::type;
};

template <typename OrigList, typename SymLetters, typename SymIndices, typename Peeled, typename Rest>
struct calc_partial_symm_tree;

template <typename OrigList, typename SymLetters, typename SymIndices, typename Peeled, typename Rest, typename Choices>
struct loop_partial_symm_tree;

template <typename OrigList, typename SymLetters, typename SymIndices, typename... Peeled>
struct calc_partial_symm_tree<OrigList, SymLetters, SymIndices, type_list<Peeled...>, type_list<>> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        using permuted_sym_indices = type_list<Peeled...>;

        using full_new_indices =
            typename embed_permutation<OrigList, SymLetters, SymIndices, permuted_sym_indices>::type;

        return rename_from_list<full_new_indices>::exec(T);
    }
};

template <typename OrigList, typename SymLetters, typename SymIndices, typename... Peeled, typename... Rest,
          typename Choice>
struct loop_partial_symm_tree<OrigList, SymLetters, SymIndices, type_list<Peeled...>, type_list<Rest...>,
                              type_list<Choice>> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        using new_peeled = typename push_back<type_list<Peeled...>, Choice>::type;

        using new_rest = typename remove_first_in_list<Choice, type_list<Rest...>>::type;

        return calc_partial_symm_tree<OrigList, SymLetters, SymIndices, new_peeled, new_rest>::exec(T);
    }
};

template <typename OrigList, typename SymLetters, typename SymIndices, typename... Peeled, typename... Rest,
          typename Choice, typename... ChoiceList>
struct loop_partial_symm_tree<OrigList, SymLetters, SymIndices, type_list<Peeled...>, type_list<Rest...>,
                              type_list<Choice, ChoiceList...>> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        using new_peeled = typename push_back<type_list<Peeled...>, Choice>::type;

        using new_rest = typename remove_first_in_list<Choice, type_list<Rest...>>::type;

        auto S = calc_partial_symm_tree<OrigList, SymLetters, SymIndices, new_peeled, new_rest>::exec(T);

        auto U = loop_partial_symm_tree<OrigList, SymLetters, SymIndices, type_list<Peeled...>, type_list<Rest...>,
                                        type_list<ChoiceList...>>::exec(T);

        return S + U;
    }
};

template <typename OrigList, typename SymLetters, typename SymIndices, typename... Peeled, typename... Rest>
struct calc_partial_symm_tree<OrigList, SymLetters, SymIndices, type_list<Peeled...>, type_list<Rest...>> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        return loop_partial_symm_tree<OrigList, SymLetters, SymIndices, type_list<Peeled...>, type_list<Rest...>,
                                      type_list<Rest...>>::exec(T);
    }
};


// symm_all_unnormalized(T)
//
// 返回所有指标排列项的和，不带 1/n!。
template <typename datatype, typename... Idx>
auto symm_all_unnormalized(const Tensor<datatype, Idx...>& T) {
    return calc_symm_tree<type_list<>, type_list<Idx...>>::exec(T);
}


// symm_all(T)
//
// 对所有指标完全对称化，带 1/n! 归一化。
template <typename datatype, typename... Idx>
auto symm_all(const Tensor<datatype, Idx...>& T) {
    auto S = symm_all_unnormalized(T);
    return (1.0 / static_cast<datatype>(factorial(sizeof...(Idx)))) * S;
}


// symm_unnormalized<Letters...>(T)
//
// 只对指定 Letters 对应的指标做未归一化对称化。
//
// 示例：
//   symm_unnormalized<mu, nu>(T)
// 返回 T_{μν...} + T_{νμ...}
template <typename... SymLetters, typename datatype, typename... Idx>
auto symm_unnormalized(const Tensor<datatype, Idx...>& T) {
    static_assert(sizeof...(SymLetters) > 0, "symm_unnormalized requires at least one letter");

    using orig_list = type_list<Idx...>;
    using sym_letters = type_list<SymLetters...>;

    using sym_indices = typename indices_from_letters<orig_list, sym_letters>::type;

    return calc_partial_symm_tree<orig_list, sym_letters, sym_indices, type_list<>, sym_indices>::exec(T);
}


// symm<Letters...>(T)
//
// 只对指定 Letters 对应的指标做归一化对称化。
//
// 示例：
//   symm<mu, nu>(T) = 1/2 * (T_{μν...} + T_{νμ...})
template <typename... SymLetters, typename datatype, typename... Idx>
auto symm(const Tensor<datatype, Idx...>& T) {
    auto S = symm_unnormalized<SymLetters...>(T);

    return (1.0 / static_cast<datatype>(factorial(sizeof...(SymLetters)))) * S;
}















// ============================================================
// 15. Antisymmetrization
// ============================================================
//
// antisymm_all(T)
// 对所有指标做完全反对称化：
//
//   antisymm_all(T) = 1/n! * Σ_{π∈S_n} sign(π) T_{π(...)}
//
// antisymm<Letters...>(T)
// 只对指定字母对应的指标做反对称化。
//
// 示例：
//   Tensor<double, dn<mu>, dn<nu>> F;
//   auto A = antisymm<mu, nu>(F);
//   // A_{μν} = 1/2 (F_{μν} - F_{νμ})
// ============================================================


// calc_partial_asymm_tree / loop_partial_asymm_tree
//
// 类似对称化树，但每个排列项带有置换符号 sign。
// 当选择某个 Choice 时，它在 Rest 中的位置 pos 决定符号是否翻转：
//   pos 为偶数：符号不变；
//   pos 为奇数：符号翻转。

template <std::size_t N, typename List>
struct asymm_nth_type;

template <std::size_t N, typename Head, typename... Tail>
struct asymm_nth_type<N, type_list<Head, Tail...>> {
    using type = typename asymm_nth_type<N - 1, type_list<Tail...>>::type;
};

template <typename Head, typename... Tail>
struct asymm_nth_type<0, type_list<Head, Tail...>> {
    using type = Head;
};

template <int N, typename Target, typename List>
struct asymm_find_type_pos;

template <int N, typename Target, typename Head, typename... Tail>
struct asymm_find_type_pos<N, Target, type_list<Head, Tail...>>
    : std::conditional_t<std::is_same_v<Target, Head>, std::integral_constant<int, N>,
                         asymm_find_type_pos<N + 1, Target, type_list<Tail...>>> {};

template <typename Index, typename LetterList>
struct asymm_index_letter_in_letters;

template <typename Index, typename... Letters>
struct asymm_index_letter_in_letters<Index, type_list<Letters...>>
    : std::bool_constant<(std::is_same_v<typename get_letter<Index>::letter, Letters> || ...)> {};

template <typename OrigList, typename LetterList>
struct asymm_indices_from_letters;

template <typename OrigList, typename... Letters>
struct asymm_indices_from_letters<OrigList, type_list<Letters...>> {
    using type = type_list<typename find_index_type<Letters, OrigList>::type...>;
};

template <typename NewList>
struct asymm_rename_from_list;

template <typename... NewIndices>
struct asymm_rename_from_list<type_list<NewIndices...>> {
    template <typename datatype, typename... OldIndices>
    static auto exec(const Tensor<datatype, OldIndices...>& T) {
        return rename<NewIndices...>(T);
    }
};

template <bool Selected, typename Head, typename SymIndices, typename PermutedSymIndices>
struct asymm_embed_one_index;

template <typename Head, typename SymIndices, typename PermutedSymIndices>
struct asymm_embed_one_index<false, Head, SymIndices, PermutedSymIndices> {
    using type = Head;
};

template <typename Head, typename SymIndices, typename PermutedSymIndices>
struct asymm_embed_one_index<true, Head, SymIndices, PermutedSymIndices> {
private:
    using head_letter = typename get_letter<Head>::letter;

    static constexpr std::size_t pos = find_indices<0, head_letter, SymIndices>::value;

public:
    using type = typename asymm_nth_type<pos, PermutedSymIndices>::type;
};

template <typename OrigList, typename SymLetters, typename SymIndices, typename PermutedSymIndices>
struct asymm_embed_permutation;

template <typename SymLetters, typename SymIndices, typename PermutedSymIndices>
struct asymm_embed_permutation<type_list<>, SymLetters, SymIndices, PermutedSymIndices> {
    using type = type_list<>;
};

template <typename Head, typename... Tail, typename SymLetters, typename SymIndices, typename PermutedSymIndices>
struct asymm_embed_permutation<type_list<Head, Tail...>, SymLetters, SymIndices, PermutedSymIndices> {
private:
    static constexpr bool selected = asymm_index_letter_in_letters<Head, SymLetters>::value;

    using new_head = typename asymm_embed_one_index<selected, Head, SymIndices, PermutedSymIndices>::type;

    using tail_result =
        typename asymm_embed_permutation<type_list<Tail...>, SymLetters, SymIndices, PermutedSymIndices>::type;

public:
    using type = typename push_front<tail_result, new_head>::type;
};

template <int Sign>
struct asymm_apply_sign;

template <>
struct asymm_apply_sign<1> {
    template <typename datatype, typename... Indices>
    static auto exec(const Tensor<datatype, Indices...>& T) {
        return T;
    }
};

template <>
struct asymm_apply_sign<-1> {
    template <typename datatype, typename... Indices>
    static auto exec(const Tensor<datatype, Indices...>& T) {
        return datatype(-1) * T;
    }
};

template <typename OrigList, typename SymLetters, typename SymIndices, typename Peeled, typename Rest, int Sign>
struct calc_partial_asymm_tree;

template <typename OrigList, typename SymLetters, typename SymIndices, typename Peeled, typename Rest, typename Choices,
          int Sign>
struct loop_partial_asymm_tree;

template <typename OrigList, typename SymLetters, typename SymIndices, typename... Peeled, int Sign>
struct calc_partial_asymm_tree<OrigList, SymLetters, SymIndices, type_list<Peeled...>, type_list<>, Sign> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        using permuted_sym_indices = type_list<Peeled...>;

        using full_new_indices =
            typename asymm_embed_permutation<OrigList, SymLetters, SymIndices, permuted_sym_indices>::type;

        auto term = asymm_rename_from_list<full_new_indices>::exec(T);

        return asymm_apply_sign<Sign>::exec(term);
    }
};

template <typename OrigList, typename SymLetters, typename SymIndices, typename... Peeled, typename... Rest,
          typename Choice, int Sign>
struct loop_partial_asymm_tree<OrigList, SymLetters, SymIndices, type_list<Peeled...>, type_list<Rest...>,
                               type_list<Choice>, Sign> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        using rest_list = type_list<Rest...>;

        static constexpr int pos = asymm_find_type_pos<0, Choice, rest_list>::value;

        static constexpr int new_sign = ((pos % 2) == 0) ? Sign : -Sign;

        using new_peeled = typename push_back<type_list<Peeled...>, Choice>::type;

        using new_rest = typename remove_first_in_list<Choice, rest_list>::type;

        return calc_partial_asymm_tree<OrigList, SymLetters, SymIndices, new_peeled, new_rest, new_sign>::exec(T);
    }
};

template <typename OrigList, typename SymLetters, typename SymIndices, typename... Peeled, typename... Rest,
          typename Choice, typename... ChoiceList, int Sign>
struct loop_partial_asymm_tree<OrigList, SymLetters, SymIndices, type_list<Peeled...>, type_list<Rest...>,
                               type_list<Choice, ChoiceList...>, Sign> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        using rest_list = type_list<Rest...>;

        static constexpr int pos = asymm_find_type_pos<0, Choice, rest_list>::value;

        static constexpr int new_sign = ((pos % 2) == 0) ? Sign : -Sign;

        using new_peeled = typename push_back<type_list<Peeled...>, Choice>::type;

        using new_rest = typename remove_first_in_list<Choice, rest_list>::type;

        auto S = calc_partial_asymm_tree<OrigList, SymLetters, SymIndices, new_peeled, new_rest, new_sign>::exec(T);

        auto U = loop_partial_asymm_tree<OrigList, SymLetters, SymIndices, type_list<Peeled...>, rest_list,
                                         type_list<ChoiceList...>, Sign>::exec(T);

        return S + U;
    }
};

template <typename OrigList, typename SymLetters, typename SymIndices, typename... Peeled, typename... Rest, int Sign>
struct calc_partial_asymm_tree<OrigList, SymLetters, SymIndices, type_list<Peeled...>, type_list<Rest...>, Sign> {
    template <typename datatype, typename... Idx>
    static auto exec(const Tensor<datatype, Idx...>& T) {
        using rest_list = type_list<Rest...>;

        return loop_partial_asymm_tree<OrigList, SymLetters, SymIndices, type_list<Peeled...>, rest_list, rest_list,
                                       Sign>::exec(T);
    }
};


// antisymm_unnormalized<Letters...>(T)
//
// 指定指标的未归一化反对称化。
template <typename... AsymmLetters, typename datatype, typename... Idx>
auto antisymm_unnormalized(const Tensor<datatype, Idx...>& T) {
    static_assert(sizeof...(AsymmLetters) > 0, "antisymm_unnormalized requires at least one letter");

    using orig_list = type_list<Idx...>;
    using asymm_letters = type_list<AsymmLetters...>;

    using asymm_indices = typename asymm_indices_from_letters<orig_list, asymm_letters>::type;

    return calc_partial_asymm_tree<orig_list, asymm_letters, asymm_indices, type_list<>, asymm_indices, 1>::exec(T);
}


// antisymm<Letters...>(T)
//
// 指定指标的归一化反对称化。
template <typename... AsymmLetters, typename datatype, typename... Idx>
auto antisymm(const Tensor<datatype, Idx...>& T) {
    auto A = antisymm_unnormalized<AsymmLetters...>(T);

    return (datatype(1) / static_cast<datatype>(factorial(sizeof...(AsymmLetters)))) * A;
}


// antisymm_all_unnormalized(T)
// 对所有指标做未归一化反对称化。
template <typename datatype, typename... Idx>
auto antisymm_all_unnormalized(const Tensor<datatype, Idx...>& T) {
    return antisymm_unnormalized<typename get_letter<Idx>::letter...>(T);
}


// antisymm_all(T)
// 对所有指标做归一化反对称化。
template <typename datatype, typename... Idx>
auto antisymm_all(const Tensor<datatype, Idx...>& T) {
    return antisymm<typename get_letter<Idx>::letter...>(T);
}















// ============================================================
// 16. Field wrapper and field operations
// ============================================================
//
// Field<F>
//
// 用来包装“张量场”：也就是一个可调用对象，输入坐标 x，输出 Tensor。
//
// 示例：
//   using X = coord<mu>;
//
//   auto g = make_field([](const X& x) {
//       return metric<mu, nu>(x);
//   });
//
//   X x{};
//   auto gx = g(x);  // gx 是 Tensor<double, dn<mu>, dn<nu>>
//
// Field 的好处是可以重载 + - *，从而用接近数学公式的写法组合张量场。


// Field
// 保存一个可调用对象 func，并转发 operator()。
template <typename F>
struct Field {
    F func;

    Field(F f) : func(std::move(f)) {
    }

    // 让 Field 可以像函数一样调用
    template <typename... Args>
    auto operator()(Args&&... args) const {
        return func(std::forward<Args>(args)...);
    }
};
template <typename F>
Field(F) -> Field<F>;


// make_field(f)
// 辅助函数：把 lambda 或函数对象包装成 Field。
//
// 示例：
//   auto g = make_field([](const X& x) { return metric<mu,nu>(x); });
template <typename F>
auto make_field(F&& f) {
    return Field<std::decay_t<F>>{std::forward<F>(f)};
}

// ==================== Field 运算符重载 ====================

// Field + Field
template <typename F1, typename F2>
auto operator+(const Field<F1>& a, const Field<F2>& b) {
    return make_field([a, b](const auto& x) {
        return a(x) + b(x);
    });
}

// Field - Field
template <typename F1, typename F2>
auto operator-(const Field<F1>& a, const Field<F2>& b) {
    return make_field([a, b](const auto& x) {
        return a(x) - b(x);
    });
}

// Field * Field（张量缩并）
template <typename F1, typename F2>
auto operator*(const Field<F1>& a, const Field<F2>& b) {
    return make_field([a, b](const auto& x) {
        return a(x) * b(x);
    });
}

// scalar * Field
template <typename F>
auto operator*(double scalar, const Field<F>& a) {
    return make_field([scalar, a](const auto& x) {
        return scalar * a(x);
    });
}

// Field * scalar
template <typename F>
auto operator*(const Field<F>& a, double scalar) {
    return scalar * a;
}

// ==================== Field 版工具函数 ====================

template <typename... NewIndices, typename F>
auto rename_field(const Field<F>& field) {
    return make_field([field](const auto& x) {
        return rename<NewIndices...>(field(x));
    });
}

template <typename... NewIndices, typename F>
auto reorder_field(const Field<F>& field) {
    return make_field([field](const auto& x) {
        return reorder<NewIndices...>(field(x));
    });
}

// partial 返回 Field
template <typename DerivLetter, typename F>
auto partial_field(const Field<F>& field, double h = 1e-2) {
    return make_field([field, h](const auto& x) {
        return partial<DerivLetter>(field, x, h);
    });
}

// trace 的 Field 版
template <typename LetterA, typename LetterB, typename F>
auto trace_field(const Field<F>& field) {
    return make_field([field](const auto& x) {
        return trace<LetterA, LetterB>(field(x));
    });
}

template <typename F>
auto symm_all_field(const Field<F>& field) {
    return make_field([field](const auto& x) {
        return symm_all(field(x));
    });
}

template <typename... SymLetters, typename F>
auto symm_field(const Field<F>& field) {
    return make_field([field](const auto& x) {
        return symm<SymLetters...>(field(x));
    });
}

template <typename... SymLetters, typename F>
auto symm_unnormalized_field(const Field<F>& field) {
    return make_field([field](const auto& x) {
        return symm_unnormalized<SymLetters...>(field(x));
    });
}

template <typename... AsymmLetters, typename F>
auto antisymm_unnormalized_field(const Field<F>& field) {
    return make_field([field](const auto& x) {
        return antisymm_unnormalized<AsymmLetters...>(field(x));
    });
}

template <typename... AsymmLetters, typename F>
auto antisymm_field(const Field<F>& field) {
    return make_field([field](const auto& x) {
        return antisymm<AsymmLetters...>(field(x));
    });
}
template <typename F>
auto antisymm_all_unnormalized_field(const Field<F>& field) {
    return make_field([field](const auto& x) {
        return antisymm_all_unnormalized(field(x));
    });
}
template <typename F>
auto antisymm_all_field(const Field<F>& field) {
    return make_field([field](const auto& x) {
        return antisymm_all(field(x));
    });
}












constexpr size_t get_C_a_b(size_t a, size_t b) {
    size_t result = 1;
    size_t denom = factorial(b);
    for (size_t i = 0; i < b; ++i) {
        result *= (a - i);
    }
    result /= denom;
    return denom;
}

template <size_t rank>
constexpr size_t expand_index_form(std::array<int,rank> indicies){
    if (rank==0 || rank==4){
        return 0;
    }else if (rank==1){
        return indicies[0];
    }
}





int main() {

    return 0;
}