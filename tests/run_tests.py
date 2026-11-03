"""Build/run C++17 checks and verify that invalid contractions fail to compile."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default=os.environ.get("CXX"))
    parser.add_argument("--build-dir", type=Path, default=root / "tests" / ".build")
    args = parser.parse_args()
    compiler = args.compiler or shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        parser.error("provide --compiler with a clang++ or g++ executable")
    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    flags = [compiler, "-std=c++17", "-Wall", "-Wextra", "-pedantic", "-I", str(root)]
    suffix = ".exe" if os.name == "nt" else ""

    for relative_path in ["tests/test_contraction.cpp", "tests/test_positional_contraction.cpp",
                          "tests/test_expression.cpp", "examples/metric_expression.cpp", "main.cpp"]:
        executable = build_dir / (Path(relative_path).stem + suffix)
        built = subprocess.run(flags + ["-O2", str(root / relative_path), "-o", str(executable)],
                               capture_output=True, text=True)
        if built.returncode:
            print(built.stdout + built.stderr, file=sys.stderr)
            return 1
        ran = subprocess.run([str(executable)], capture_output=True, text=True)
        if ran.returncode:
            print(ran.stdout + ran.stderr, file=sys.stderr)
            return 1
        print(f"PASS {relative_path}: {ran.stdout.strip()}")

    failures = {
        "expression_duplicate_indices.cpp": "Tensor indices must not repeat",
        "expression_same_variance.cpp": "Invalid contraction due to same variance contraction",
        "expression_rename_variance.cpp": "rename may only change index letters",
        "same_variance.cpp": "Invalid contraction due to same variance contraction",
        "pair_out_of_range.cpp": "contraction pair axis is out of range",
        "repeated_left_axis.cpp": "an operand axis must occur in at most one contraction pair",
        "repeated_right_axis.cpp": "an operand axis must occur in at most one contraction pair",
        "shape_mismatch.cpp": "left component count does not match contraction plan",
    }
    for filename, diagnostic in failures.items():
        source = root / "tests" / "compile_fail" / filename
        built = subprocess.run(flags + ["-fsyntax-only", str(source)], capture_output=True, text=True)
        if built.returncode == 0 or diagnostic not in built.stdout + built.stderr:
            print(f"FAIL expected compile-time diagnostic for {filename}\n{built.stdout}{built.stderr}",
                  file=sys.stderr)
            return 1
        print(f"PASS compile-time rejection: {filename}")

    # Compile snippets from stdin so new plan checks need no extra source files.
    invalid_plans = {
        "trace axis out of range": (
            "return trace_plan<2, 0, 2>::result_size;", "trace plan axis is out of range"),
        "trace same axis": (
            "return trace_plan<2, 0, 0>::result_size;", "trace requires two distinct axes"),
        "trace shape mismatch": (
            "trace_components<trace_plan<2, 0, 1>>(std::array<double, 4>{});",
            "input component count does not match trace plan"),
        "trace same variance": (
            "trace<a, b>(Tensor<double, up<a>, up<b>>{});", "trace requires opposite index variance"),
        "trace missing letter": (
            "trace<a, c>(Tensor<double, up<a>, dn<b>>{});", "trace index letter must exist in the tensor"),
        "trace repeated letter": (
            "trace<a, a>(Tensor<double, up<a>, dn<b>>{});", "trace requires two distinct index letters"),
        "expression offset mismatch": (
            "Tensor<double> sample(const Tensor<double, up<mu>>&); "
            "auto result = make_field<&sample>() * make_field<&sample, offset<1, 0, 0, 0>>();",
            "contraction operands must share the node offset"),
        "runtime derivative strategy": (
            "Tensor<double> sample(const coord<mu>&); "
            "DiffScheme scheme = DiffScheme::Central2; "
            "auto derivative = partial_field<mu, scheme>(make_field<&sample>());",
            "no matching function"),
        "derivative repeated index": (
            "Tensor<double, up<mu>> sample(const coord<nu>&); "
            "auto derivative = partial_field<mu>(make_field<&sample>());",
            "Tensor indices must not repeat"),
    }
    for name, (body, diagnostic) in invalid_plans.items():
        source = '#include "field_wrapper.hpp"\nint main() { ' + body + ' }\n'
        built = subprocess.run(flags + ["-x", "c++", "-fsyntax-only", "-"], input=source,
                               capture_output=True, text=True)
        if built.returncode == 0 or diagnostic not in built.stdout + built.stderr:
            print(f"FAIL expected compile-time diagnostic for {name}\n{built.stdout}{built.stderr}",
                  file=sys.stderr)
            return 1
        print(f"PASS compile-time rejection: {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
