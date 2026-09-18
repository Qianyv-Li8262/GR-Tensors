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

## 文件导览

- `tensor_class.hpp`：张量容器、索引打包和分量访问。
- `Index_and_variance.hpp`、`type_lists.hpp`：指标和编译期类型列表工具。
- `contraction_and_arithmetic.hpp`：缩并、算术、改名与迹。
- `symm_asymm.hpp`：对称化与反对称化。
- `field_wrapper.hpp`、`num_deriv.hpp`：张量场封装与有限差分。
- `main.cpp`：Schwarzschild/Kretschmann 端到端示例。
