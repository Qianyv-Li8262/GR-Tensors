# TMP Einstein Tensor

一个用于爱因斯坦求和张量计算的 C++17 原型库。项目以模板元编程（TMP）将**指标字母**和**上下指标位置**编码进 `Tensor` 的类型，让不合法的张量运算尽可能在编译期暴露。

当前实现固定在四维指标空间（每个指标取 `0..3`）。

## 已有内容

- `Tensor<Scalar, Indices...>`：使用连续 `std::array` 存储 `4^rank` 个分量。
- `up<Letter>` / `dn<Letter>`：在类型层表示上、下指标；内置 `mu`、`nu`、`rho`、`sigma` 等指标字母。
- 张量乘法：同名且一上一下的指标自动缩并；同变异性缩并会触发编译期错误。
- 加减法、数乘、指标改名（`rename`）和显式迹（`trace`）。加减法允许指标排列不同，但要求指标集合一致。
- 指定指标位置的对称化 `symm_orbit` 与反对称化 `asymm_orbit`。
- `Field` 包装器，用于把坐标到张量的映射组合成场表达式。
- `make_field<&function>()`：以具名函数作为类型化叶子，用 `*` 构造缩并表达式递归树。
- 数值偏导 `partial`：支持二、四、六、八阶中心差分；默认八阶。`partial_up` 使用逆度规将导数指标抬升。

`diff_forms.hpp` 目前只是占位头文件，尚未提供微分形式运算。

## 示例：Schwarzschild 曲率不变量

[`main.cpp`](main.cpp) 定义了 Schwarzschild 度规及其逆度规，并按以下过程数值计算

```text
g_ab  ->  Gamma^rho_{mu nu}  ->  R^rho_{sigma mu nu}  ->  K = R_abcd R^abcd
```

程序在 `(t, r, theta, phi) = (0, 10, pi/2, 0)` 处，将数值结果与解析式

```text
K = 48 M^2 / r^6
```

比较；只有相对误差小于 `1e-5` 时才以成功状态退出。连接和曲率的嵌套数值微分分别使用独立步长，便于调参。

## 构建和运行

需要支持 C++17 的编译器。仓库没有构建系统，直接编译示例即可：

```bash
g++ -std=c++17 -O2 main.cpp -o main
./main
```

在 Windows 上可将输出文件命名为 `main.exe` 并运行 `./main.exe`。库代码均为头文件；在自己的程序中包含 `tensors_all.hpp` 即可。

## 最小用法

```cpp
#include "tensors_all.hpp"

Tensor<double, up<mu>, dn<nu>> A{};
Tensor<double, up<nu>, dn<rho>> B{};

// nu 自动缩并，C 的类型为 Tensor<double, up<mu>, dn<rho>>。
auto C = A * B;
```

指标的字母用于匹配，存储位置并不决定数学指标的顺序；需要改变字母而不改变上下指标位置时使用 `rename`。抬降指标不是 `rename` 的职责，应通过度规与张量乘法完成。

## 缩并的两层接口

指标分析与数值执行分离。配对、上下指标检查和输出轴顺序在编译期确定；计划直接保存多个 `contractpair<L, R>`，不再构造中间槽位映射。

第一层根据指标类型生成计划，轴编号从零开始：

```cpp
using Analysis = contraction_analysis<
    type_list<up<a>, dn<mu>, up<b>, dn<nu>>,
    type_list<up<mu>, dn<c>, up<d>, up<nu>>>;

using Plan = Analysis::plan;
static_assert(std::is_same_v<Plan,
    contraction_plan<4, 4, contractpair<1, 0>, contractpair<3, 3>>>);
static_assert(std::is_same_v<Analysis::free_indices,
    type_list<up<a>, up<b>, dn<c>, up<d>>>);
```

`contractpair<L, R>` 表示左张量第 `L` 轴与右张量第 `R` 轴缩并。输出按“左侧自由轴、右侧自由轴”的顺序排列，各自保持原有轴顺序；求和顺序按 pair 的声明顺序。第一层按左侧指标出现顺序生成 pair，因此保持原实现的输出布局与求和顺序。

执行时用普通循环枚举输出分量并填入自由轴坐标，然后递归消耗 pair：每层给 `indexleft[Pair::left]` 和 `indexright[Pair::right]` 赋同一个 `0..3` 坐标；pair 为空时累加左右分量的乘积。`Plan::pairs` 保留独立的轴对类型，便于后续对轴对集合设计无序 hash。

第二层只接收分量数组，计划作为模板实参传入，无需计划对象，也不再分析指标字母：

```cpp
using Plan = contraction_plan<2, 2, contractpair<1, 0>>;
std::array<double, 16> left{};
std::array<double, 16> right{};
auto result = contract_components<Plan>(left, right); // std::array<double, 16>
```

执行层仅依赖 `contraction_executor.hpp`。直接构造位置计划时，调用方负责张量的数学含义；轴越界、重复使用某一输入轴、分量数组大小不匹配均会在编译期被拒绝。按字母使用 `A * B` 时，上下指标兼容性由第一层检查。空 pair 包表示外积；完全缩并返回只有一个分量的数组。当前仍固定每轴四维。

这层分离已经用于下面的缩并表达式树；当前尚未实现表达式 DAG 或场求值缓存。

## 类型化的缩并表达式树

具名函数通过非类型模板参数进入叶子类型。函数须返回 `Tensor`，并能接收传给 `.eval(x)` 的坐标；函数声明在 `main` 之外。不同函数即使函数指针类型相同，也会产生不同的 `FieldExpr<&function>` 类型。

```cpp
Tensor<double, dn<a>, dn<b>> metric(const coord<mu>& x);
Tensor<double, up<a>, up<b>> inverse_metric(const coord<mu>& x);

auto g = make_field<&metric>();
auto inverse = make_field<&inverse_metric>();
auto inverse_bc = rename_field<up<b>, up<c>>(inverse);
auto g_cd = rename_field<dn<c>, dn<d>>(g);
auto inverse_dmu = rename_field<up<d>, up<mu>>(inverse);

// 建树时不调用函数、不计算分量；每次 * 在编译期分析指标并确定本层 Plan。
auto expression = g * inverse_bc * g_cd * inverse_dmu;
using Tree = decltype(expression)::node_type;
using Plan = Tree::Plan; // contraction_plan<2, 2, contractpair<1, 0>>
using Result = decltype(expression)::result_type; // Tensor<double, dn<a>, up<mu>>

coord<mu> x{};
x(1) = 10.0;
x(2) = 1.1;
auto result = expression.eval(x); // 数值计算在这里发生。
```

外层 `IndexedExpr<Node, Indices...>` 负责指标字母；内层 `ContractExpr<L, R, Plan>` 只包含左右子树和轴对计划。节点通过 `left_type`、`right_type`、`children`、`Plan`、`scalar_type`、`rank` 暴露元数据。`rename_field` 保持存储轴顺序与上下指标位置，只换外层字母，内层节点类型不变。因此不同字母生成相同轴对计划时，对应计算子树可以拥有相同类型。

树按 C++ 的括号和左结合规则保留原始结构；例如上面的四项连乘为：

```cpp
using P = contraction_plan<2, 2, contractpair<1, 0>>;
using G = FieldExpr<&metric>;
using I = FieldExpr<&inverse_metric>;
using Tree = ContractExpr<ContractExpr<ContractExpr<G, I, P>, G, P>, I, P>;
```

空 pair 包也通过同一节点表示外积；零阶张量场与其他张量场的乘法也是缩并节点。完全缩并得到零阶 `Tensor`，使用 `result()` 取标量。具名函数本身是不可展开的叶子，`eval` 从根递归计算子节点，相同子树出现两次就计算两次。节点按值保存子节点，表达式可安全保存由临时表达式构成的子树。

这一接口目前支持场之间的 `*`、指标改名和求值；加减法、直接数值常量的乘法及求导尚未编码为树节点。原有 `make_field(callable)`、`Tensor * Tensor` 接口继续使用原来的行为；两种场接口暂不混合组合。

[`examples/metric_expression.cpp`](examples/metric_expression.cpp) 提供可直接运行的 Schwarzschild 四项连乘，并打印结果类型、完整展开的树类型和外层表达式类型：

```bash
clang++ -std=c++17 -O2 -I . examples/metric_expression.cpp -o metric_expression
./metric_expression
```

示例的类型输出使用 Clang/GCC 的 `__PRETTY_FUNCTION__` 或 MSVC 的 `__FUNCSIG__` 扩展，仅用于展示；缩并表达式本身不依赖类型字符串。

## 验证

需要 Python 3 和 `clang++` 或 `g++`：

```bash
python tests/run_tests.py
# 也可以显式指定编译器路径：
python tests/run_tests.py --compiler clang++
```

检查包含矩阵乘法、交错多轴缩并、含 `contractpair<3, 3>` 的四阶张量、完全缩并、外积及零阶输入的独立分量循环对照；另有仅依赖执行层的数组测试和 Schwarzschild 曲率示例。表达式测试检查编译期树类型、改名后的子树类型相同、建树不求值、非对角度规连乘、外积、零阶场乘法、全缩并，以及重复子树的调用次数。非法同变异性缩并、改名改变上下位置、越界轴、左右重复轴及数组大小错误必须编译失败，并检查对应诊断。

## 文件导览

- `tensor_class.hpp`：张量容器、索引打包和分量访问。
- `Index_and_variance.hpp`、`type_lists.hpp`：指标和编译期类型列表工具。
- `contraction_analysis.hpp`：指标字母分析与编译期缩并计划生成。
- `contraction_plan.hpp`：`contractpair<L, R>`、轴对列表、阶数及轴合法性检查。
- `contraction_executor.hpp`：仅依据位置计划执行分量缩并。
- `tensor_expression.hpp`：具名函数叶子、缩并递归树及外层指标视图。
- `contraction_and_arithmetic.hpp`：连接缩并的两层接口，以及算术、改名与迹。
- `symm_asymm.hpp`：对称化与反对称化。
- `field_wrapper.hpp`、`num_deriv.hpp`：张量场封装与有限差分。
- `main.cpp`：Schwarzschild/Kretschmann 端到端示例。
