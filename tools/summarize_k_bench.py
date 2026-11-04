"""Summarize the measured K checks, runtime bench and compiler time traces."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'outputs'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def execution_counts(graph):
    children = {n['id']: [] for n in graph['nodes']}
    for parent, child, _ in graph['edges']:
        children[parent].append(child)
    sizes, leaves = {}, {}
    for node in graph['nodes']:
        i = node['id']
        sizes[i] = 1 + sum(sizes[c] for c in children[i])
        leaves[i] = 1 if node['label'] == 'g' else sum(leaves[c] for c in children[i])
    return sizes[graph['root']], leaves[graph['root']]


def main():
    rows = []
    for scheme in ['central2', 'central8']:
        result_path = OUT/f'k_bench_{scheme}.json'
        graph_path = OUT/f'k_structure_{scheme}.json'
        build_path = OUT/f'k_build_{scheme}.json'
        if not all(p.exists() for p in [result_path, graph_path, build_path]):
            continue
        result, graphs, build = load(result_path), load(graph_path), load(build_path)
        tree_nodes, tree_metric_calls = execution_counts(graphs['dag'])
        assert tree_metric_calls == result['checks'][0]['direct_metric_calls']
        unique_metric_nodes = sum(n['label'] == 'g' for n in graphs['dag']['nodes'])
        assert unique_metric_nodes == result['checks'][0]['dag_metric_calls']
        rows.append((result, graphs, build, tree_nodes))
    lines = ['# K: direct expression vs compile(K).eval()', '',
             'Measured on 2026-11-04. Both engines use the same mathematical expression, '
             'difference scheme, coordinate, step and compiler flags. The metric has M=1.', '',
             '## Runtime benchmark', '',
             'Windows / Clang 22.1.6, C++17, -O2 -DNDEBUG. One CPU thread; no GPU. '
             'Three warm-up evaluations, then seven timed rounds; report the median per evaluation. '
             'Each round targets about 60 ms independently for each engine. Evaluation wrappers '
             'are noinline and results are consumed by a volatile sink. Every graph evaluation '
             'starts with an empty cache. Metric call counting remains enabled in both engines.', '',
             '| Scheme | h at r=6, theta=1.1 | Direct (us) | DAG (us) | Speedup | Metric calls direct / DAG |',
             '|---|---:|---:|---:|---:|---:|']
    for result, graphs, build, tree_nodes in rows:
        b, check = result['benchmark'], result['checks'][0]
        lines.append(f'| {result["scheme"]} | {b["h"]} | {b["direct_us"]:.3f} | {b["dag_us"]:.3f} | '
                     f'{b["speedup"]:.2f}x | {check["direct_metric_calls"]} / {check["dag_metric_calls"]} |')
    lines += ['', 'Timings are specific to this machine and run; the min/max of the seven rounds '
              'and iteration counts are retained in each JSON result. The Schwarzschild metric '
              'leaf is inexpensive, so call-count reduction is not a runtime-speedup prediction.', '',
              '## Accuracy', '',
              'Each scheme is checked at (r,theta)=(4,0.7),(6,1.1),(10,pi/2), with two steps. '
              'Central2 uses h=0.002,0.001; Central8 uses h=0.02,0.01. '
              'The independent scalar reference is K=48/r^6. Every case requires finite values, '
              'direct/DAG relative difference below 5e-8, reference error below 2e-4 for Central2 '
              'or 5e-7 for Central8, and fewer graph metric calls.', '',
              '| Scheme | Max relative direct/DAG difference | Max relative reference error | Passed |',
              '|---|---:|---:|---|']
    for result, *_ in rows:
        lines.append(f'| {result["scheme"]} | {max(c["relative_difference"] for c in result["checks"]):.3e} | '
                     f'{max(c["relative_reference_error"] for c in result["checks"]):.3e} | {result["passed"]} |')
    lines += ['', '## Compile time', '',
              '`compile(K)` is the outer InstantiateFunction event in the optional Clang time trace '
              '(inclusive duration, no summation of nested events). It includes DAG construction '
              'and instantiation of graph metadata/cache types. Full build time also includes '
              'parsing, both evaluation engines, graph export, optimization, code generation and linking. '
              'Regular build time is measured without tracing. Trace collection has significant overhead; '
              'its inclusive event time cannot be subtracted from an uninstrumented build time.', '',
              '| Scheme | Regular full build (s) | Traced compile(K) event (s) | Traced full build (s) |', '|---|---:|---:|---:|']
    for result, graphs, build, _ in rows:
        trace_path = OUT/f'k_build_{result["scheme"].lower()}_trace.json'
        profile = load(trace_path) if trace_path.exists() else {}
        instantiation = profile.get('compile_k_instantiation_seconds')
        time_text = f'{instantiation:.3f}' if instantiation is not None else 'not recorded'
        full_trace = f'{profile["compile_seconds"]:.3f}' if profile else 'not recorded'
        lines.append(f'| {result["scheme"]} | {build["compile_seconds"]:.3f} | {time_text} | {full_trace} |')
    old = OUT/'k_build_central2_before_hash_fix.json'
    if old.exists():
        before = load(old)
        lines += ['', f'Before the compile-time storage fix, Central2 measured '
                  f'{before["compile_k_instantiation_seconds"]:.3f} s for compile(K) and '
                  f'{before["compile_seconds"]:.3f} s for the full build. That older measurement '
                  'overlapped an initial Central8 build, so it is not a controlled compiler-performance '
                  'comparison. The initial Central8 build was stopped after more than 20 minutes '
                  'to remove discarded bucket candidates and runtime-tuple metaprogramming. '
                  'The current implementation also computes use counts and cache indices in one constexpr pass.']
    lines += ['', '## Exact graph shapes', '',
              'The original mathematical tree has 191 occurrences and 21 distinct mathematical node types. '
              'Partial derivatives are still unexpanded in those two views. The execution tree below is '
              'the actual lowered DAG unfolded once per incoming path, so tree and DAG counts use '
              'the same primitive operations.', '',
              '| Scheme | Expanded execution tree | Execution DAG | Cached nodes | Unique metric samples |',
              '|---|---:|---:|---:|---:|']
    for result, graphs, build, tree_nodes in rows:
        lines.append(f'| {result["scheme"]} | {tree_nodes} | {result["nodes"]} | '
                     f'{result["cached_nodes"]} | {result["checks"][0]["dag_metric_calls"]} |')
    lines += ['', 'All diagrams are exported from the actual C++ mathematical and graph types. '
              'Circle labels are operations or g; edges point from a result to its inputs. '
              'Shaded nodes are shared. The execution-tree view expands lazily to keep it readable.', '',
              '## Reproduce', '', '```powershell',
              'python tools/bench_k.py --schemes central2 central8',
              'python tools/bench_k.py --schemes central2 --trace --build-only  # optional compiler profile',
              'python tools/summarize_k_bench.py', '```', '',
              'Exact compiler commands, platform, compile times, numeric checks, timed-round ranges '
              'and actual nodes/edges are saved in k_build_*.json, k_bench_*.json and k_structure_*.json. '
              'The optional compiler trace granularity is one second; its thresholds and expanded template-depth '
              'limits are recorded in the build command. High-order K is a substantial compile-time workload.', '']
    (OUT/'k_benchmark_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(OUT/'k_benchmark_report.md')


if __name__ == '__main__':
    main()
