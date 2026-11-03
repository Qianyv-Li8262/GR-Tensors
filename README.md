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
- 数值偏导 `partial`：编译期选择二、四、六、八阶中心差分，默认八阶；类型化表达式的偏导直接展开成带 offset 的采样子树。`partial_up` 使用逆度规将导数指标抬升。

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

## 缩并与取迹的两层接口

缩并和取迹的指标分析与数值执行分离。指标层负责字母匹配、上下指标检查和结果指标；执行层只接收位置计划与分量数组，不依赖 `Tensor` 或指标字母。相关计划和执行逻辑放在现有的 `contraction_*` 头文件中；加法保持 `contraction_and_arithmetic.hpp` 中的原实现。

三种计划生成的外层入口统一通过 `::plan` 获取计划、`::free_indices` 获取结果指标，调用方不需要准备递归的空列表：

```cpp
using Add = addition_analysis<LeftIndices, RightIndices>;
using Contract = contraction_analysis<LeftIndices, RightIndices>;
using Trace = trace_analysis<Indices, LetterA, LetterB>;
```

`addition_analysis` 与加法原有的 `analysis_plus_impl`、`tensorplus_impl` 一起保留在 `contraction_and_arithmetic.hpp` 中；索引兼容性检查由外层入口负责。

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

取迹用 `trace_plan<输入阶数, 轴A, 轴B>` 表示同一输入中需要取相同坐标并求和的两根轴，自由轴保留原存储顺序：

```cpp
using Analysis = trace_analysis<type_list<up<a>, dn<b>, up<c>, dn<d>>, a, d>;
static_assert(std::is_same_v<Analysis::plan, trace_plan<4, 0, 3>>);
static_assert(std::is_same_v<Analysis::free_indices, type_list<dn<b>, up<c>>>);

std::array<double, 256> input{};
auto result = trace_components<Analysis::plan>(input); // std::array<double, 16>
```

原来的 `trace<a, d>(tensor)` 接口在指标层生成上述计划再调用执行层；二阶张量取迹返回零阶 `Tensor`。缺失字母、相同字母或同为上/下指标会在分析层报错；直接指定位置时，执行前检查轴越界、两轴相同和数组大小不匹配。

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

外层 `IndexedExpr<Node, Indices...>` 负责指标字母；内层 `ContractExpr<L, R, Plan, PointOffset>` 包含左右子树、轴对计划和采样位置。节点通过 `left_type`、`right_type`、`children`、`Plan`、`scalar_type`、`rank`、`point_offset` 暴露元数据。`rename_field` 保持存储轴顺序、上下指标位置和 offset，只换外层字母，内层节点类型不变。因此不同字母生成相同轴对计划和 offset 时，对应计算子树可以拥有相同类型。

树按 C++ 的括号和左结合规则保留原始结构；例如上面的四项连乘为：

```cpp
using P = contraction_plan<2, 2, contractpair<1, 0>>;
using G = FieldExpr<&metric>;
using I = FieldExpr<&inverse_metric>;
using Tree = ContractExpr<ContractExpr<ContractExpr<G, I, P>, G, P>, I, P>;
```

空 pair 包也通过同一节点表示外积；零阶张量场与其他张量场的乘法也是缩并节点。完全缩并得到零阶 `Tensor`，使用 `result()` 取标量。具名函数本身是不可展开的叶子，`eval` 从根递归计算子节点，相同子树出现两次就计算两次。节点按值保存子节点，表达式可安全保存由临时表达式构成的子树。

`offset<O0, O1, O2, O3>` 是四个编译期 `int`，省略时 `offset<>` 等同于 `offset<0, 0, 0, 0>`。`FieldExpr<Function, PointOffset>` 和 `ContractExpr<L, R, Plan, PointOffset>` 的位置参数均默认为零；外层 `IndexedExpr::point_offset` 直接取自内部节点。offset 不增加运行时数据成员。

offset 的四个值表示相对同一个根坐标 `x` 的整数步数。求值时传入公共步长 `h`（默认 `1e-2`），每个叶子在 `sample(axis) = x(axis) + h * offset[axis]` 处调用函数。普通运算节点传递原始 `x` 和 `h`，因此嵌套多层也只在叶子平移一次。

```cpp
// 直接创建带位置的叶子。
auto shifted_metric = make_field<&metric, offset<1, -2, 0, 0>>();

// 也可以从根向下，在指定位置直接生成整棵树的类型与对象。
using O = offset<1, -2, 0, 0>;
using ShiftedTree = typename expand<decltype(expression)::node_type, O>::result;
auto shifted_expression = expand_at<O>(expression);
auto shifted_result = shifted_expression.eval(x, 0.01);
```

`expand<Node, O>::result` 由根向下确定节点位置：普通运算传递相同位置，偏导分支叠加各自的差分偏移。`expand_at<O>(expression)` 同时创建对应对象，保留指标和轴对计划，建树时不调用函数。`O` 是根节点相对 `x` 的绝对中心位置，重复展开会替换中心并保留各差分分支的相对偏移，`expand_at<>(expression)` 恢复零中心。普通缩并要求左右操作数与父节点具有相同 offset；不同采样位置会产生不同节点类型。

这一接口目前支持场之间的 `*`、指标改名、偏导和求值；一般加减法及直接数值常量的乘法尚未编码为树节点。原有 `make_field(callable)`、`Tensor * Tensor` 接口继续使用原来的行为；两种场接口暂不混合组合。

### 编译期偏导展开

包含 `num_deriv.hpp` 或 `tensors_all.hpp` 后，将差分策略写入模板参数：

```cpp
// 建树时不求值。每个方向生成 8 个 expression 的采样子树，共 32 个。
auto derivative = partial_field<rho, DiffScheme::Central8>(expression);
auto values = derivative.eval(x, 0.01);

// 嵌套偏导直接继续展开；结果最前面依次是 sigma、rho 两根导数轴。
auto second = partial_field<sigma, DiffScheme::Central2>(derivative);
auto second_values = second.eval(x, 0.01);

// partial(expr) 也返回树；带坐标的形式直接求值。
auto same_tree = partial<rho, DiffScheme::Central8>(expression);
auto same_values = partial<rho, DiffScheme::Central8>(expression, x, 0.01);
```

`derivative_stencil<Scheme>` 在编译期确定采样位置和权重：二、四、六、八阶策略每个方向分别有 2、4、6、8 个点。`PartialExpr` 包含四个方向的 `StencilSumExpr`；每个加权和节点通过 `children` 暴露该方向所有带 offset 的原表达式子树，按固定权重求和后除以 `h`。原表达式如果是缩并，每个采样分支都会包含完整的缩并子树。

嵌套偏导在编译期合并整数偏移，例如同一方向的外层 `+1` 与内层 `-1` 生成零 offset 的相同叶子类型。运行时仍分别执行各采样路径，尚未做缓存或子树合并。连续求导保持原差分算子的复合：两次二阶中心一阶差分得到 `(f(x+2h)-2f(x)+f(x-2h))/(4h*h)`。类型化表达式树内部共用 `.eval(x, h)` 传入的步长。

原有 callable `Field` 的数值接口也改为编译期策略：`partial<mu, DiffScheme::Central4>(field, x, h)`、`partial_field<mu, DiffScheme::Central4>(field, h)`，以及 `partial_up<mu, nu, DiffScheme::Central4>(field, inverse, x, h)`。这一路的函数体仍是叶子，结果类型由未求值的表达式推导，不再额外调用一次中心点函数。运行时 `switch` 和接收 `DiffScheme` 实参的重载已移除。

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

检查包含矩阵乘法、交错多轴缩并、含 `contractpair<3, 3>` 的四阶张量、完全缩并、外积及零阶输入的独立分量循环对照；另有仅依赖执行层的数组测试和 Schwarzschild 曲率示例。表达式测试检查编译期树类型、改名后的子树类型相同、建树不求值、非对角度规连乘、外积、零阶场乘法、全缩并，以及重复子树的调用次数。offset 测试覆盖四维正负偏移、默认零位置、向叶子传播、整树重新展开、改名保留位置和多层树只平移一次。偏导测试覆盖四种策略的多项式解析导数、完整采样坐标与调用次数、嵌套/混合偏导、对含偏导的乘积继续求导，以及 callable 接口和导数指标抬升。非法同变异性缩并、改名改变上下位置、越界轴、左右重复轴、数组大小错误、缩并节点位置不一致、重复导数指标和运行时差分策略必须编译失败，并检查对应诊断。

## 文件导览

- `tensor_class.hpp`：张量容器、索引打包和分量访问。
- `Index_and_variance.hpp`、`type_lists.hpp`：指标和编译期类型列表工具。
- `contraction_analysis.hpp`：缩并、取迹的指标分析与位置计划生成。
- `contraction_plan.hpp`：`contraction_plan`、`trace_plan` 及位置合法性检查；缩并保留独立的 `contractpair<L, R>`。
- `contraction_executor.hpp`：仅依据位置计划执行数组的缩并和取迹。
- `tensor_expression.hpp`：具名函数叶子、缩并递归树、外层指标视图及编译期 offset 展开。
- `contraction_and_arithmetic.hpp`：连接缩并的两层接口，以及算术、改名与迹。
- `symm_asymm.hpp`：对称化与反对称化。
- `field_wrapper.hpp`：callable 张量场封装。
- `num_deriv.hpp`：编译期差分策略、偏导采样树、偏导 offset 展开与数值求值接口。
- `main.cpp`：Schwarzschild/Kretschmann 端到端示例。
