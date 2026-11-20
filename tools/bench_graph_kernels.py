"""Compare evaluator build time and runtime using already exported K DAGs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--schemes', type=int, nargs='+', choices=(2, 4), default=[2, 4])
    args = parser.parse_args()
    if args.repeats < 1 or not args.label.replace('_', '').replace('-', '').isalnum():
        parser.error('positive repeats and an alphanumeric label are required')
    root = Path(__file__).resolve().parent.parent
    folder = root/'outputs'/'kernel_compile'/args.label
    folder.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which('clang++')
    if not compiler:
        parser.error('clang++ is required')
    original = (root/'examples/kretschmann_graph_bench.cpp').read_text()
    report = dict(compiler=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
                  operations_sha256=hashlib.sha256((root/'expression_graph_operations.hpp').read_bytes()).hexdigest(),
                  schemes={}, note='Saved DAG; full benchmark/export driver codegen and linking included. No graph construction or tracing. Runtime benchmarks run after all builds.')
    for scheme in args.schemes:
        header = root/f'outputs/compile_profile/central{scheme}/saved_graph.hpp'
        if not header.exists():
            parser.error(f'exported graph missing: {header}')
        source = original.replace('const auto graph = compile(K);', 'const SavedGraph graph{};')
        source = source.replace('using Graph = decltype(compile(K));', 'using Graph = SavedGraph;')
        # The original driver targets Central2/8. For Central4 use the steps
        # already validated by profile_compile.py, retaining its error threshold.
        if scheme == 4:
            source = source.replace('order == 2 ? 0.002 : 0.02', 'order == 2 ? 0.002 : 0.01')
        source = '#include "examples/kretschmann_formula.hpp"\n'+f'#include "{header.as_posix()}"\n'+source
        src, exe = folder/f'k{scheme}.cpp', folder/f'k{scheme}.exe'
        src.write_text(source, encoding='utf-8')
        command = [compiler, '-std=c++17', '-O2', '-DNDEBUG', '-fconstexpr-steps=1000000000',
                   '-ftemplate-depth=16384', '-fbracket-depth=16384', '-ferror-limit=1',
                   '-ftemplate-backtrace-limit=1', '-I', str(root), '-I', str(root/'examples'),
                   f'-DK_BENCH_SCHEME={(scheme-2)//2}', str(src), '-o', str(exe)]
        seconds = []
        for i in range(args.repeats):
            start = time.perf_counter()
            result = subprocess.run(command, capture_output=True)
            seconds.append(time.perf_counter()-start)
            (folder/f'k{scheme}.log').write_bytes(result.stdout+result.stderr)
            if result.returncode:
                raise SystemExit(f'compile failed; see {folder}/k{scheme}.log')
            print(f'{args.label} K{scheme} build {i+1}: {seconds[-1]:.3f}s', flush=True)
        report['schemes'][str(scheme)] = dict(command=command, seconds=seconds,
            median_seconds=statistics.median(seconds), executable_bytes=exe.stat().st_size,
            graph_sha256=hashlib.sha256(header.read_bytes()).hexdigest())
        (folder/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for scheme in args.schemes:
        path = folder/f'k{scheme}_results.json'
        subprocess.run([str(folder/f'k{scheme}.exe'), str(path)], check=True)
        result = json.loads(path.read_text())
        assert result['passed'] and len(result['checks']) == 6
        report['schemes'][str(scheme)]['result'] = result
    (folder/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved {folder}/report.json', flush=True)


if __name__ == '__main__':
    main()
