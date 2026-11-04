# TMP Einstein Tensor

一个用于爱因斯坦求和张量计算的 C++17 原型库。项目以模板元编程（TMP）将**指标字母**和**上下指标位置**编码进 `Tensor` 的类型，让不合法的张量运算尽可能在编译期暴露。

当前实现固定在四维指标空间（每个指标取 `0..3`）。

## 已有内容

- `Tensor<Scalar, Indices...>`：使用连续 `std::array` 存储 `4^rank` 个分量。
- `up<Letter>` / `dn<Letter>`：在类型层表示上、下指标；内置 `mu`、`nu`、`rho`、`sigma` 等指标字母。
- 张量乘法：同名且一上一下的指标自动缩并；同变异性缩并会触发编译期错误。
- 加减法、数乘、指标改名（`rename`）和显式迹（`trace`）。加减法允许指标排列不同，但要求指标集合一致。
- 指定零起始轴号的归一化对称化 `sym<...>`、反对称化 `asym<...>`，支持 Tensor 与类型化表达式；原有按指标类型选择的 `symm_orbit`、`asymm_orbit` 继续可用。
- `Field` 包装器，用于把坐标到张量的映射组合成场表达式。
- `make_field<&function>()`：以具名函数作为类型化叶子，用 `*` 构造缩并表达式递归树。
- 数值偏导 `partial`：编译期选择二、四、六、八阶中心差分，默认八阶；数学类型保留未展开的偏导节点，数值求值沿用采样执行。`partial_up` 使用逆度规将导数指标抬升。

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

这层分离用于下面的缩并表达式树，也用于 `compile` 生成的表达式 DAG 与单次求值缓存。

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

`expand<Node, O>::result` 由根向下确定节点位置：数学节点传递相同位置，保留偏导操作本身；已有采样执行节点则保留各自的差分位移。`expand_at<O>(expression)` 同时创建对应对象，建树时不调用函数。`O` 是根节点相对 `x` 的绝对中心位置，`expand_at<>(expression)` 恢复零中心。普通缩并要求左右操作数与父节点具有相同 offset；不同采样位置会产生不同节点类型。

这一接口支持场之间的 `+`、`-`、`*`、指标改名、取迹、度规求逆、偏导和求值。`scale_field<1, 2>(expr)` 将有理系数编码为 `std::ratio<1, 2>`，避免 C++17 浮点模板参数的限制。加法的数学节点使用原有 `addition_analysis` 生成的轴置换计划；`contraction_and_arithmetic.hpp` 中的 Tensor 加法保持原实现。原有 callable `Field` 接口继续可用，两种场接口暂不混合组合。

### 完整轴置换

`index<Axes...>` 显式给出完整的零起始轴对应关系：输出第 `k` 根轴取自输入第 `Axes[k]` 根轴。必须将输入的每根轴恰好列一次，包含保持原位的轴；缺失、重复或越界都在编译期拒绝，不会自动补齐。

```cpp
auto B = permute(A, index<1, 0>{}); // [a,b] -> [b,a]，B(j,i) = A(i,j)
auto C = permute(A3, index<2, 0, 1>{}); // [a,b,c] -> [c,a,b]，C(k,i,j) = A3(i,j,k)
auto D = permute<0, 2, 1>(A3); // 等价简写，仍须给全三根轴。
auto unchanged = permute(A3, index<0, 1, 2>{}); // 恒等映射也完整写出。
auto scalar = permute(Tensor<double>(3.0), index<>{}); // 零阶的完整映射为空。

// 后端直接重排已经算出的数组，不需要重新求值计算树。
auto reordered = permute_components<index<1, 0>>(std::array<double, 16>{});
```

`permute` 同时支持 Tensor 和类型化表达式。Tensor 的指标名称及上下位置随轴一起移动；表达式增加一个 `PermuteExpr<Node, index<...>, Offset>` 节点，保留完整计划、单个子节点以及 offset，也能继续组合偏导。`index<...>::is_identity` 在编译期识别完整恒等映射：表达式直接转交操作数的求值结果，保留公式节点但跳过分量重排。Tensor 和裸数组接口仍按值返回独立结果。一般置换将输入复制到独立输出数组，因此即使某分量的源/目标位置相同也必须写入，不能跳过造成输出未初始化；不提供原地覆盖接口。源/目标分量位置均在编译期计算，生成固定槽位的复制操作，运行时不查映射表或转换多维下标。数组复制实现位于 `permutation.hpp`；DAG 执行路径将显式置换保留为轴映射元数据，见下文。

### 对称化与反对称化

`sym<Axes...>(A)` 和 `asym<Axes...>(A)` 按零起始的存储轴位置选择指标，未选中的轴保持固定。选择的指标须至少两个、互不重复、位于输入范围内，并且上下位置一致。结果的指标字母、轴顺序和阶数不变；两者都使用归一化投影，即全排列求和后除以 `k!`，反对称化额外乘以排列符号。

```cpp
// A 可以是 Tensor，也可以是 make_field<&function>() 得到的表达式。
auto symmetric = sym<0, 2>(A);
auto alternating = asym<1, 2>(A);
auto nested = sym<0, 2>(asym<1, 2>(A)); // 先反对称，再对称。

// 十阶输入上选择六个对称轴和三个反对称轴，也只增加两个数学节点。
auto projected = sym<1, 2, 4, 5, 7, 8>(asym<4, 5, 6>(A10));
std::cout << projected.formula() << '\n';
```

例如最后一式的内部类型形如 `SymExpr<AsymExpr<Leaf, asym_plan<10, 4, 5, 6>>, sym_plan<10, 1, 2, 4, 5, 7, 8>>`，不会把各排列展开成加法子树。每个节点的 `children` 只包含一个操作数，位置 plan 不包含指标字母；`rename_field` 保留内部类型，`expand_at` 将 offset 继续传到叶子，也可以在任意一层组合偏导和其他运算。

上述十阶例子的两组轴共同包含 `4,5`，因此这两个完整投影的复合在数学上为零。这里仍忠实保留两个操作节点。一般交叠的对称、反对称投影需要保留调用顺序，不能假定交换。

与此前运算一样，入口自行生成 plan，不需要手动准备初始列表：

```cpp
using Analysis = sym_analysis<type_list<dn<a>, up<b>, dn<c>>, 0, 2>;
using Plan = Analysis::plan; // sym_plan<3, 0, 2>
auto result = sym_components<Plan>(std::array<double, 64>{});

using Alternating = asym_analysis<type_list<dn<a>, dn<b>, dn<c>>, 0, 2>;
auto other = asym_components<Alternating::plan>(std::array<double, 64>{});
```

计划、指标分析和数组执行都位于原有 `symm_asymm.hpp`，数学节点和表达式入口位于 `tensor_expression.hpp`。执行沿用轨道归并：累加到排序后的代表，归一化后广播。每层只求值一次操作数，运行时不枚举 `k!` 个排列；四维中反对称五个或更多轴直接得到零分量。代表槽位、排列符号和归一化因子均由 plan 在编译期预计算，再由 `if constexpr` 和固定大小的参数包生成固定源/目标槽位的数值操作。浮点类型的归一化权重也在编译期计算，求值时乘以常数。运行时不再查映射表、拆下标、排序或判断排列符号，只执行结果初始化、固定位置的加减、归一化和广播。编译期计算按块进行、块遍历采用平衡递归，避免旧实现每个分量增加一层模板递归；只构造或查看数学类型时不会生成这些执行步骤。实际指令中的内联、向量化和临时数组消除仍由编译优化决定。

### 查看 K 的数学类型

`IndexedExpr::formula_type` 就是完整的数学表达式类型。`expr.formula()` 直接返回编译器的 `__PRETTY_FUNCTION__` 或 `__FUNCSIG__`；不会调用函数、求值或展开偏导采样，也没有额外的公式打印器。

[`examples/kretschmann_formula.cpp`](examples/kretschmann_formula.cpp) 在 `main()` 中现场创建度规叶子、逆度规、偏导、Christoffel、Riemann 和 K。度规函数本身定义在外部，表达式对象与所有组合均在 `main()` 内：

```cpp
auto g = make_field<&metric>();
auto inverse = inverse_field(g);
auto pg = partial_field<mu, DiffScheme::Central8>(g);
// 同一 main() 中继续组合 Gamma、pGamma、R_lower、R_upper，完整代码见示例。
auto K = R_lower * R_upper;
using Formula = decltype(K)::formula_type;
std::cout << K.formula() << '\n';
```

逆度规由 `InverseExpr<FieldExpr<&metric>>` 表示；Gamma、R 的函数负责组合表达式，所以它们会内联进入 K 的类型，只有度规函数是叶子。`PartialExpr<Operand, Scheme, Offset>` 保留原操作数，`children` 只有一棵子树。K 的构造和类型检查均不生成采样分支。

```bash
clang++ -std=c++17 -O0 -I . examples/kretschmann_formula.cpp -o kretschmann_formula
./kretschmann_formula         # 类型统计
./kretschmann_formula --type  # 编译器吐出的完整 K.formula()
```

当前数学树有 191 个节点位置、21 种不同节点类型、54 处度规叶子、40 处未展开偏导，最大偏导嵌套为 2，采样节点为 0。Clang 给出的完整签名长 11,507 个字符。不同节点类型数是对这棵数学树的类型统计，不代表已做 CSE，也不是编译器所有模板实例化的总数。此示例只检查数学类型，不对 K 做数值求值或解析对照。

[`examples/kretschmann_formula.hpp`](examples/kretschmann_formula.hpp) 还保留可复用的构造函数供独立数值对照使用；编译期断言检查它与 `main()` 内现场构造的类型完全相同。另一个独立程序 [`examples/kretschmann_compare.cpp`](examples/kretschmann_compare.cpp) 使用该数学类型做数值求值，与手推的度规零、一、二阶导数表达式对照；推导全文见 [`examples/kretschmann_manual.md`](examples/kretschmann_manual.md)。它同时检查全部 256 个 Riemann 分量，以及 `K=48M^2/r^6`：

```bash
clang++ -std=c++17 -O0 -I . examples/kretschmann_compare.cpp -o kretschmann_compare
./kretschmann_compare
```

在三组坐标、两种步长的六组检查中，K 相对解析参考的最大误差约为 `3.56e-10`。这是独立解析参考与原有有限差分执行的数值验证；数学类型里的偏导仍未展开，该示例尚未切换到 DAG 执行路径。

### Bach 张量的四层偏导示例

[`examples/bach_tensor.cpp`](examples/bach_tensor.cpp) 在 `main()` 中从一个度规叶子组合 `Gamma -> Riemann -> Ricci/Weyl -> 两层协变导数 -> Bach`，每层协变导数均显式包含全部联络项，直接使用现有表达式操作。数学树保留四层偏导，具有 12,160 个节点位置；完整编译器类型见 [`outputs/bach_formula_type_central2.txt`](outputs/bach_formula_type_central2.txt)。没有修改库执行器或加入 CSE。

示例默认使用编译期 `Central2`，编译时增加 `-DBACH_CENTRAL4` 可比较四阶中心差分。四种背景包括具有非零解析 Bach 的常曲率曲面乘积及其共形变换，以及两个 Bach 为零的背景；每种背景扫描四组步长，核对全部 16 个 Bach 和 256 个 Weyl 分量。公式、独立解析推导与复现命令见 [`examples/bach_manual.md`](examples/bach_manual.md)，数值结果见 [`outputs/bach_comparison.md`](outputs/bach_comparison.md)。

### 数学偏导与现有采样执行

包含 `tensor_expression.hpp` 或 `tensors_all.hpp` 后，将差分策略写入模板参数：

```cpp
// 建树时保留一个数学偏导节点，不生成采样子树。
auto derivative = partial_field<rho, DiffScheme::Central8>(expression);
std::cout << derivative.formula() << '\n';
// 数值求值仍可使用原有的差分采样执行。
auto values = derivative.eval(x, 0.01);

// 数学树保留嵌套偏导；结果最前面依次是 sigma、rho 两根导数轴。
auto second = partial_field<sigma, DiffScheme::Central2>(derivative);
auto second_values = second.eval(x, 0.01);

// partial(expr) 也返回树；带坐标的形式直接求值。
auto same_tree = partial<rho, DiffScheme::Central8>(expression);
auto same_values = partial<rho, DiffScheme::Central8>(expression, x, 0.01);
```

`derivative_stencil<Scheme>` 在编译期确定采样位置和权重：二、四、六、八阶策略每个方向分别有 2、4、6、8 个点。数学节点 `PartialExpr` 与执行节点 `SampledPartialExpr` 分离；数学树直接调用 `.eval()` 时使用原有的四方向 `StencilSumExpr` 采样执行，加权求和后除以 `h`，这条直接执行路径不做 CSE。另一条 `compile` 路径在建图时降级偏导，并复用相同位置的采样，见下文。

嵌套偏导在编译期合并整数偏移，例如同一方向的外层 `+1` 与内层 `-1` 生成零 offset 的相同叶子类型。数学树的直接执行器仍分别执行各采样路径，不做缓存或子树合并。连续求导保持原差分算子的复合：两次二阶中心一阶差分得到 `(f(x+2h)-2f(x)+f(x-2h))/(4h*h)`。类型化表达式树内部共用 `.eval(x, h)` 传入的步长。

原有 callable `Field` 的数值接口也改为编译期策略：`partial<mu, DiffScheme::Central4>(field, x, h)`、`partial_field<mu, DiffScheme::Central4>(field, h)`，以及 `partial_up<mu, nu, DiffScheme::Central4>(field, inverse, x, h)`。这一路的函数体仍是叶子，结果类型由未求值的表达式推导，不再额外调用一次中心点函数。运行时 `switch` 和接收 `DiffScheme` 实参的重载已移除。

[`examples/metric_expression.cpp`](examples/metric_expression.cpp) 提供可直接运行的 Schwarzschild 四项连乘，并打印结果类型、完整展开的树类型和外层表达式类型：

```bash
clang++ -std=c++17 -O2 -I . examples/metric_expression.cpp -o metric_expression
./metric_expression
```

示例的类型输出使用 Clang/GCC 的 `__PRETTY_FUNCTION__` 或 MSVC 的 `__FUNCSIG__` 扩展，仅用于展示；缩并表达式本身不依赖类型字符串。

### 编译期表达式 DAG

`expression_graph.hpp` 提供第二条执行路径，支持场叶子、缩并（含外积和标量乘法）、有理数缩放、加减、取迹、求逆、对称/反对称投影、置换以及四种中心差分偏导。原有数学树与递归执行器保持各自的接口：

```cpp
#include "expression_graph.hpp"

auto F = make_field<&f>(); // f、g 返回二阶张量，指标为 up<a>, dn<b>
auto G = make_field<&g>();
auto S = F * rename_field<up<b>, dn<a>>(G); // 完全缩并为标量场
auto A = S * S;

auto graph = compile(A);
auto direct = A.eval(x, h);
auto shared = graph.eval(x, h); // 返回与 direct 相同的 Tensor 类型

static_assert(decltype(graph)::node_count == 4);
static_assert(decltype(graph)::cache_count == 1);
static_assert(decltype(graph)::use_count<2> == 2);
```

`compile` 解开 `IndexedExpr`，在递归前先查询整棵当前子树，未命中才后序编译孩子，生成 `ExprGraph<Output, DAGNode<Id, Operation, Inputs...>...>`。输出描述保留原来的指标及根节点的轴映射；孩子先于父节点分配 ID。图类型包含全部操作、位置计划和依赖，图对象不保留原表达式或指向它的指针，建图不会调用场函数。

建图使用不可变的编译期 `TypeHashTable`：类型签名的 FNV-1a 哈希选择桶，桶内通过完整类型相等判断解决碰撞。表内有两类键：`SourceKey<规范子树类型>` 映射到完整 `NodeRef<Id, Layout>`；`NodeKey<操作, 孩子ID...>` 映射到节点 ID。前者在递归之前查询，不依赖孩子 ID；后者在递归之后查询。`compile<1>(A)` 可强制所有键进入同一桶，仍得到相同的图类型。哈希与 ID 只用于当前编译和图内部，不作为跨编译器的持久标识。

入口按左右原始子树的编译器类型签名（`__PRETTY_FUNCTION__` / `__FUNCSIG__`）字典序选择方向，交换时同步转置缩并轴对，并记录自由轴顺序变化。它只规范化当前一层，不递归规范化孩子；命中后通过独立的模板特化直接返回已有引用，不实例化孩子的编译路径。未命中则继续编译，结束时登记规范子树到图存储的轴映射。原样重复及当前层左右相反的子树均可提前命中；更深层的等价变形仍由后序查重处理。长类型签名的生成和比较仍有编译成本。

递归后的缩并再把操作类型和孩子 ID 组成键查重，此时孩子按 ID 升序排列；相同 ID 时比较缩并轴对以确定方向。交换左右时同步转置轴对，并把原输出顺序记录为 `index<...>` 映射。父节点把孩子映射直接吸收到自己的缩并轴号中，不创建中间置换数组；只有根节点输出顺序与数学表达式不同时，才调用 `permute_components` 恢复顺序。这里不排序求和轴对、不改变缩并的结合方式，也不识别任意代数等价关系。

使用次数按去重后的 DAG 边数统计，同一父节点两次读取同一孩子计为两次；返回根结果计为一次。运行时沿依赖递归求值，仅为使用次数大于一的节点生成 `tuple<optional<结果数组>...>` 缓存，单次使用结果通过临时值交给父节点。缓存每次 `eval` 重新创建，不跨坐标保留；第一版共享缓存持续到本次求值结束，尚未按最后一次使用提前释放。

偏导在编译时通过现有采样位置模板降为场采样、有理常数缩放、加法、除以运行时步长和四方向拼装。执行图不保留黑盒的偏导递归调用；嵌套偏导的整数位移在类型里相加，相同场函数在相同绝对 offset 的采样进入同一个图节点。仅 offset 相同不足以合并不同的函数或运算。不同求值调用仍使用各自的新缓存。

例如 `partial_field<rho, DiffScheme::Central2>(partial_field<nu, DiffScheme::Central2>(f))`的四维 Hessian，直接执行有 64 条采样路径；DAG 只含 33 个不同采样位置，中心 `offset<0,0,0,0>` 只求值一次。同方向分量仍是两个中心一阶差分的复合 `(f(x+2h)-2f(x)+f(x-2h))/(4h*h)`，未改写成另一个二阶差分公式。

显式置换只组合引用的轴映射，不创建运算节点。加法按布局直接读取分量，缩放保持布局；取迹和投影把布局吸收到轴计划中。对称/反对称始终是单输入投影节点，不展开成排列分支。求逆在二阶输入的存储上计算，并保留转置布局，原执行器和图执行器共用数组求逆内核。

场函数应当在相同输入下返回相同结果。缩并规范化当前限定算术标量；交换乘法操作数、单独物化差分的加权项，以及转置布局下的求逆可能影响浮点舍入细节，对拍采用数值容差。图上的复用是按结构和位置做的 CSE，不进行任意代数化简或浮点运算重结合。未知节点仍给出编译期诊断。

### K 的图执行对拍、bench 与节点图

```powershell
python tools/bench_k.py --schemes central2 central8
python tools/bench_k.py --schemes central2 --trace --build-only # 可选：带开销的编译剖析
python tools/summarize_k_bench.py
```

`examples/kretschmann_graph_bench.cpp` 使用同一个 K 表达式比较递归执行与 `compile(K).eval()`，在三组坐标、两种步长下对照独立解析值 `48/r^6`。bench 在预热后计七轮中位耗时，使用相同编译选项和防止整次求值被提升到循环外的包装；同时记录度规调用次数。编译脚本默认记录不带剖析的整份构建耗时；可选 `--trace` 从 Clang 时间追踪提取根 `compile(K)` 实例化的包含耗时，不叠加内部事件。时间追踪本身会增加编译开销，两种口径分开记录。

报告保存为 `outputs/k_benchmark_report.md`，原始数据保存为 `k_bench_*.json`、`k_build_*.json`。图数据 `k_structure_*.json` 从实际 C++ 类型导出，分别包含数学树、保留偏导的数学类型 DAG，以及偏导降级后的执行 DAG。`tools/render_k_graph.py --output <HTML片段路径>` 生成可缩放的圆圈节点图，并在 `outputs` 保存对应 SVG；执行树由实际执行 DAG 沿每条路径分别展开，可逐层检查。数学树与展开执行树的节点数使用不同口径，图中分别标明。

哈希桶和节点列表使用纯类型列表；哈希插入仅实例化被选中的桶。使用次数和缓存槽位通过一次编译期数组遍历确定，缓存类型仅拼接共享结果，避免为元数据实例化运行时 tuple 拼接的模板设施。这些调整不改变图节点和数值执行顺序。

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
- `tensor_expression.hpp`：数学表达式节点、编译器类型查看接口、偏导采样执行和 offset 传递。
- `expression_graph.hpp`：表达式 DAG 大类型、编译期哈希表、带轴映射的规范化、偏导降级、`compile` 和选择性共享缓存求值。
- `expression_graph_operations.hpp`：图执行的数组运算、单输入投影、步长除法和方向拼装原语。
- `contraction_and_arithmetic.hpp`：连接缩并的两层接口，以及算术、改名与迹。
- `symm_asymm.hpp`：对称化、反对称化的位置计划、指标分析、数组执行与 Tensor 入口。
- `permutation.hpp`：完整零起始轴置换计划、固定槽位复制执行器与 Tensor 入口。
- `field_wrapper.hpp`：callable 张量场封装。
- `num_deriv.hpp`：编译期差分策略、步长检查和 callable 数值偏导接口。
- `main.cpp`：Schwarzschild/Kretschmann 端到端示例。
