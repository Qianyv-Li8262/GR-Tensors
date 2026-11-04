"""Compile, cross-check, benchmark and export the actual K expression graph."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schemes', nargs='+', choices=['central2', 'central4', 'central6', 'central8'],
                        default=['central2', 'central8'])
    parser.add_argument('--compiler', default=shutil.which('clang++'))
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--trace', action='store_true', help='Enable Clang time tracing (adds compile overhead)')
    args = parser.parse_args()
    if not args.compiler:
        parser.error('clang++ is required')
    root = Path(__file__).resolve().parent.parent
    build = root / 'tests' / '.build'
    output = root / 'outputs'
    build.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    flags = ['-std=c++17', '-O2', '-DNDEBUG', '-fconstexpr-steps=100000000',
             '-ftemplate-depth=8192', '-fbracket-depth=8192', '-ferror-limit=1',
             '-ftemplate-backtrace-limit=1', '-I', str(root)]
    compiler_version = subprocess.check_output([args.compiler, '--version'], text=True).splitlines()[0]
    for scheme in args.schemes:
        number = (int(scheme.removeprefix('central')) - 2) // 2
        exe = build / ('k_graph_' + scheme + ('.exe' if os.name == 'nt' else ''))
        result_path = output / ('k_bench_' + scheme + '.json')
        graph_path = output / ('k_structure_' + scheme + '.json')
        metadata_path = output / ('k_build_' + scheme + ('_trace' if args.trace else '') + '.json')
        trace_path = output / ('k_trace_' + scheme + '.json')
        trace_flags = [f'-ftime-trace={trace_path}', '-ftime-trace-granularity=1000000'] if args.trace else []
        command = [args.compiler, *flags, *trace_flags, f'-DK_BENCH_SCHEME={number}',
                   str(root / 'examples' / 'kretschmann_graph_bench.cpp'), '-o', str(exe)]
        if not args.skip_build:
            print(f'BUILD {scheme}: {compiler_version}', flush=True)
            start = time.perf_counter()
            process = subprocess.run(command, capture_output=True, text=True)
            elapsed = time.perf_counter() - start
            metadata = {'compiler': compiler_version, 'platform': platform.platform(),
                        'processor': platform.processor(), 'command': command, 'compile_seconds': elapsed,
                        'returncode': process.returncode, 'includes_graph_export': True,
                        'trace_enabled': args.trace}
            if args.trace and process.returncode == 0 and trace_path.exists():
                trace = json.loads(trace_path.read_text(encoding='utf-8'))['traceEvents']
                graph_events = [e for e in trace if e.get('name') == 'InstantiateFunction' and
                                e.get('args', {}).get('detail', '').startswith('compile<')]
                if graph_events:
                    metadata['compile_k_instantiation_seconds'] = max(e['dur'] for e in graph_events) / 1e6
                metadata['frontend_seconds'] = sum(e.get('dur', 0) for e in trace
                                                   if e.get('name') == 'Frontend') / 1e6
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            if process.returncode:
                log = output / ('k_build_' + scheme + '.log')
                log.write_text(process.stdout + process.stderr, encoding='utf-8')
                print(f'BUILD FAILED {scheme} after {elapsed:.2f}s; details: {log}', flush=True)
                print((process.stdout + process.stderr)[:3500], flush=True)
                return 1
            print(f'BUILD PASS {scheme}: {elapsed:.2f}s, {exe.stat().st_size} bytes', flush=True)
        if args.build_only:
            continue
        # Run sequentially after compilation has stopped, to avoid comparing
        # timings while this script is consuming CPU compiling another scheme.
        process = subprocess.run([str(exe), str(result_path), str(graph_path)], text=True)
        if process.returncode:
            return process.returncode
        result = json.loads(result_path.read_text(encoding='utf-8'))
        graph = json.loads(graph_path.read_text(encoding='utf-8'))
        assert result['passed']
        assert len(graph['tree']['nodes']) == 191
        assert len(graph['dag']['nodes']) == result['nodes']
        for view in graph.values():
            ids = {n['id'] for n in view['nodes']}
            assert len(ids) == len(view['nodes'])
            assert all(a in ids and b in ids for a, b, _ in view['edges'])
            assert all(n['label'] != '?' for n in view['nodes'])
        print(f'EXPORT PASS {scheme}: tree={len(graph["tree"]["nodes"])} '
              f'math_dag={len(graph["math_dag"]["nodes"])} dag={len(graph["dag"]["nodes"])}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
