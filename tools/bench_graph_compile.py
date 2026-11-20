"""Time graph-only C++ builds, without instantiating eval, and fingerprint the DAG type."""

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
    parser.add_argument('--graph-header', default='expression_graph.hpp')
    parser.add_argument('--scheme', type=int, choices=(2, 4, 6, 8), default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    root = Path(__file__).resolve().parent.parent
    compiler = shutil.which('clang++')
    if not compiler:
        parser.error('clang++ is required')
    build = root/'tests'/'.build'
    build.mkdir(parents=True, exist_ok=True)
    header = (root/args.graph_header).resolve()
    source = build/'graph_compile_probe.cpp'
    executable = build/'graph_compile_probe.exe'
    source.write_text('#include "examples/kretschmann_formula.hpp"\n'
        f'#include "{header.as_posix()}"\n#include <iostream>\n'
        'int main() {\n'
        f'  auto expression = kretschmann_formula<DiffScheme::Central{args.scheme}>();\n'
        '  using Graph = decltype(compile(expression));\n'
        '  std::cout << Graph::node_count << " " << Graph::cache_count << "\\n"\n'
        '            << dag_detail::type_name<Graph>();\n}\n', encoding='utf-8')
    command = [compiler, '-std=c++17', '-O0', '-DNDEBUG', '-fconstexpr-steps=1000000000',
               '-ftemplate-depth=16384', '-fbracket-depth=16384', '-ferror-limit=1',
               '-ftemplate-backtrace-limit=1', '-I', str(root), str(source), '-o', str(executable)]
    durations = []
    for repeat in range(args.repeats):
        start = time.perf_counter()
        result = subprocess.run(command, capture_output=True)
        durations.append(time.perf_counter()-start)
        if result.returncode:
            log = args.output.with_suffix('.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_bytes(result.stdout+result.stderr)
            raise SystemExit(f'build failed; see {log}')
        print(f'Central{args.scheme} graph-only build {repeat+1}: {durations[-1]:.3f}s', flush=True)
    metadata, signature = subprocess.check_output([str(executable)]).split(b'\n', 1)
    nodes, cached = map(int, metadata.split())
    report = dict(scheme=args.scheme, nodes=nodes, cached_nodes=cached,
                  graph_type_sha256=hashlib.sha256(signature).hexdigest(),
                  header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
                  compiler=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
                  command=command, seconds=durations, median_seconds=statistics.median(durations),
                  includes_eval=False, includes_small_driver_codegen_and_link=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('nodes', 'cached_nodes', 'graph_type_sha256', 'median_seconds')}))


if __name__ == '__main__':
    main()
