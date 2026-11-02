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

指标分析与数值执行已经分离。配对、上下指标检查、输出轴顺序和位置映射全部在编译期确定，运行时仅执行分量求和。

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

第二层只接收分量数组，计划作为模板实参传入，无需计划对象，也不再分析指标字母：

```cpp
using Plan = contraction_plan<2, 2, contractpair<1, 0>>;
std::array<double, 16> left{};
std::array<double, 16> right{};
auto result = contract_components<Plan>(left, right); // std::array<double, 16>
```

执行层仅依赖 `contraction_executor.hpp`。直接构造位置计划时，调用方负责张量的数学含义；轴越界、重复使用某一输入轴、分量数组大小不匹配均会在编译期被拒绝。按字母使用 `A * B` 时，上下指标兼容性由第一层检查。空 pair 包表示外积；完全缩并返回只有一个分量的数组。当前仍固定每轴四维。

这层分离提供了后续表达式图执行器可使用的入口；当前尚未实现表达式 DAG 或场求值缓存。

## 验证

需要 Python 3 和 `clang++` 或 `g++`：

```bash
python tests/run_tests.py
# 也可以显式指定编译器路径：
python tests/run_tests.py --compiler clang++
```

检查包含矩阵乘法、交错多轴缩并、含 `contractpair<3, 3>` 的四阶张量、完全缩并、外积及零阶输入的独立分量循环对照；另有仅依赖执行层的数组测试和 Schwarzschild 曲率示例。非法同变异性缩并、越界轴、左右重复轴及数组大小错误必须编译失败，并检查对应诊断。

## 文件导览

- `tensor_class.hpp`：张量容器、索引打包和分量访问。
- `Index_and_variance.hpp`、`type_lists.hpp`：指标和编译期类型列表工具。
- `contraction_analysis.hpp`：指标字母分析与编译期缩并计划生成。
- `contraction_plan.hpp`：`contractpair<L, R>`、位置计划及编译期轴映射。
- `contraction_executor.hpp`：仅依据位置计划执行分量缩并。
- `contraction_and_arithmetic.hpp`：连接缩并的两层接口，以及算术、改名与迹。
- `symm_asymm.hpp`：对称化与反对称化。
- `field_wrapper.hpp`、`num_deriv.hpp`：张量场封装与有限差分。
- `main.cpp`：Schwarzschild/Kretschmann 端到端示例。
