"""Separate formula, graph construction and evaluator compilation on small K graphs."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import time


def family(detail):
    # Keep huge expression/BuildState signatures out of the report.
    stem = detail.split('<', 1)[0][:160]
    if stem == 'TypeHashTable':
        for member in ('bind', 'inserted', 'insert'):
            if f'>::{member}<' in detail:
                return stem + '::' + member
    return stem


def summarize_trace(path):
    events = json.loads(path.read_text(encoding='utf-8'))['traceEvents']
    totals = {e['name'][6:]: e.get('dur', 0) / 1e6
              for e in events if e.get('name', '').startswith('Total ')}
    groups = defaultdict(lambda: [0, 0.0, 0.0])
    component_locations = defaultdict(lambda: [0, 0.0])
    threads = defaultdict(list)
    for e in events:
        if e.get('ph') == 'X' and not e.get('name', '').startswith('Total '):
            threads[e.get('pid'), e.get('tid')].append(e)
            detail = e.get('args', {}).get('detail', '')
            if e.get('name') == 'InstantiateFunction' and detail.startswith('dag_detail::component_block<'):
                start = detail.find('lambda at ')
                end = detail.find(')', start)
                location = detail[start + 10:end] if start >= 0 else 'unknown'
                component_locations[location][0] += 1
                component_locations[location][1] += e['dur'] / 1e6
    for thread in threads.values():
        stack = []
        for e in sorted(thread, key=lambda e: (e['ts'], -e['dur'])):
            start, end = e['ts'], e['ts'] + e['dur']
            while stack and start >= stack[-1][0]:
                stack.pop()
            key = (e['name'], family(e.get('args', {}).get('detail', '')))
            value = groups[key]
            value[0] += 1
            value[1] += e['dur'] / 1e6
            value[2] += e['dur'] / 1e6
            if stack and end <= stack[-1][0]:
                groups[stack[-1][1]][2] -= e['dur'] / 1e6
            stack.append((end, key))
    ranked = [dict(event=k[0], family=k[1], count=v[0],
                   inclusive_seconds=v[1], exclusive_recorded_seconds=v[2])
              for k, v in groups.items()]
    return dict(totals_seconds=totals,
                component_blocks_by_lambda_location={k: dict(count=v[0], inclusive_seconds=v[1])
                    for k, v in component_locations.items()},
                top_instantiations=sorted(
                    [r for r in ranked if r['event'] in ('InstantiateClass', 'InstantiateFunction')],
                    key=lambda r: r['exclusive_recorded_seconds'], reverse=True)[:25],
                note='Exclusive time subtracts recorded nested scopes only; below-threshold work remains attributed to its parent. Inclusive categories overlap and must not be added.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schemes', nargs='+', type=int, choices=(2, 4, 6, 8), default=[2, 4])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--resume', action='store_true', help='reuse finished timings for unchanged sources/compiler/flags')
    parser.add_argument('--buckets', nargs='*', type=int, default=[], help='also compare graph builds with these bucket counts')
    parser.add_argument('--output', type=Path, default=Path('outputs/compile_profile'))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if any(b < 1 for b in args.buckets):
        parser.error('--buckets must be positive')
    root = Path(__file__).resolve().parent.parent
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which('clang++')
    if not compiler:
        parser.error('clang++ is required')
    flags = ['-std=c++17', '-O2', '-DNDEBUG', '-fconstexpr-steps=1000000000',
             '-ftemplate-depth=16384', '-fbracket-depth=16384', '-ferror-limit=1',
             '-ftemplate-backtrace-limit=1', '-I', str(root)]
    report = dict(compiler=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
                  flags=flags, source_hashes={p: hashlib.sha256((root/p).read_bytes()).hexdigest()
                  for p in ('expression_graph.hpp', 'expression_graph_operations.hpp',
                            'tensor_expression.hpp', 'examples/kretschmann_formula.hpp')}, schemes={})
    if args.resume and (output/'report.json').exists():
        previous = json.loads((output/'report.json').read_text(encoding='utf-8'))
        for key in ('compiler', 'flags', 'source_hashes'):
            if previous[key] != report[key]:
                parser.error(f'Cannot resume: {key} changed')
        report = previous

    def save():
        (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    common = '#include "examples/kretschmann_formula.hpp"\n#include <iostream>\n#include <cstdlib>\n'
    for scheme in args.schemes:
        folder = output / f'central{scheme}'
        folder.mkdir(exist_ok=True)
        expression = f'kretschmann_formula<DiffScheme::Central{scheme}>()'
        header = folder/'saved_graph.hpp'
        sources = {
            'formula': common + f'int main() {{ auto e = {expression}; std::cout << sizeof(e); }}\n',
            'graph': common + '#include "expression_graph.hpp"\n' +
                f'int main() {{ using G = decltype(compile({expression})); '
                'std::cout << G::node_count << " " << G::cache_count; }\n',
            'export': common + '#include "graph_type_export.hpp"\n' +
                f'int main(int argc, char** argv) {{ using G = decltype(compile({expression})); '
                'return argc != 2 || !export_graph_type<G>(argv[1], "SavedGraph"); }\n',
            'saved_metadata': common + f'#include "{header.as_posix()}"\n' +
                'int main() { std::cout << SavedGraph::node_count << " " << SavedGraph::cache_count; }\n',
        }
        for stage, value in [('dag', 'SavedGraph{}'), ('direct', expression)]:
            sources[stage] = common + (f'#include "{header.as_posix()}"\n' if stage == 'dag' else '') + (
                'int main(int argc, char** argv) { coord<mu> x{}; '
                'x(1) = argc > 1 ? std::strtod(argv[1], nullptr) : 10.; '
                'x(2) = 1.1; double h = argc > 2 ? std::strtod(argv[2], nullptr) : .01; '
                f'auto e = {value}; std::cout.precision(17); std::cout << e.eval(x,h)(); }}\n')
        result = report['schemes'].setdefault(str(scheme), {})
        for stage, source in sources.items():
            src, exe = folder/f'{stage}.cpp', folder/f'{stage}.exe'
            src.write_text(source, encoding='utf-8')
            command = [compiler, *flags, str(src), '-o', str(exe)]
            record = result.setdefault(stage, dict(command=command, seconds=[]))
            # Export is setup for later probes, not a repeated timing target.
            for repeat in range(len(record['seconds']), 1 if stage == 'export' else args.repeats):
                start = time.perf_counter()
                built = subprocess.run(command, capture_output=True)
                elapsed = time.perf_counter()-start
                (folder/f'{stage}.log').write_bytes(built.stdout+built.stderr)
                if built.returncode:
                    save()
                    raise SystemExit(f'Failed {scheme}/{stage}; see {folder}/{stage}.log')
                record['seconds'].append(elapsed)
                print(f'Central{scheme} {stage} {repeat+1}: {elapsed:.3f}s', flush=True)
            record['median_seconds'] = statistics.median(record['seconds'])
            record['executable_bytes'] = exe.stat().st_size
            if stage == 'export':
                subprocess.run([str(exe), str(header)], check=True)
            elif stage in ('dag', 'direct'):
                checks = []
                for radius in (6., 10., 15.):
                    for step in ((.002, .001) if scheme == 2 else (.01, .005)):
                        value = float(subprocess.check_output([str(exe), str(radius), str(step)]))
                        expected = 48/radius**6
                        error = abs(value/expected-1)
                        checks.append(dict(radius=radius, h=step, value=value, relative_error=error))
                        if not error < (2e-4 if scheme == 2 else 5e-7):
                            raise RuntimeError(f'Numerical check failed: {scheme}/{stage}: {checks[-1]}')
                record['checks'] = checks
                if stage == 'direct':
                    differences = [abs(a['value']/b['value']-1)
                                   for a, b in zip(checks, result['dag']['checks'])]
                    record['max_direct_dag_relative_difference'] = max(differences)
                    if not max(differences) < 5e-8:
                        raise RuntimeError('Direct/DAG numerical comparison failed')
            else:
                record['metadata'] = subprocess.check_output([str(exe)], text=True)
            save()
            if stage != 'export':
                granularity = 1000 if scheme == 2 else 100000
                trace = folder/(f'{stage}.json' if scheme == 2 else f'{stage}_100ms.json')
                if not (args.resume and trace.exists()):
                    traced = subprocess.run([*command, f'-ftime-trace={trace}',
                                             f'-ftime-trace-granularity={granularity}'], capture_output=True)
                    if traced.returncode:
                        (folder/f'{stage}_trace.log').write_bytes(traced.stdout+traced.stderr)
                        raise RuntimeError(f'Trace failed: {scheme}/{stage}')
                record['trace'] = summarize_trace(trace)
                record['trace']['granularity_us'] = granularity
                save()
        for buckets in args.buckets:
            stage = f'buckets_{buckets}'
            src, exe = folder/f'{stage}.cpp', folder/f'{stage}.exe'
            source = sources['graph'].replace('compile(', f'compile<{buckets}>(')
            source = source.replace('int main()', f'#include "{header.as_posix()}"\nint main()')
            source = source.replace('std::cout << G::node_count',
                'static_assert(std::is_same_v<G, SavedGraph>); std::cout << G::node_count')
            src.write_text(source, encoding='utf-8')
            command = [compiler, *flags, str(src), '-o', str(exe)]
            record = result.setdefault(stage, dict(command=command, seconds=[], graph_type_matches=True))
            for repeat in range(len(record['seconds']), args.repeats):
                start = time.perf_counter()
                built = subprocess.run(command, capture_output=True)
                elapsed = time.perf_counter()-start
                (folder/f'{stage}.log').write_bytes(built.stdout+built.stderr)
                if built.returncode:
                    save()
                    raise RuntimeError(f'Bucket probe failed: {scheme}/{buckets}')
                record['seconds'].append(elapsed)
                print(f'Central{scheme} {stage} {repeat+1}: {elapsed:.3f}s', flush=True)
            record['median_seconds'] = statistics.median(record['seconds'])
            save()
    lines = ['# Small K graph compilation profile', '',
             'Normal builds use identical -O2 flags and include a small driver and linking. '
             'Each reported time is the median of the recorded untraced builds. '
             'Export is setup and timed once. Builds are sequential.', '',
             '| Scheme | Nodes / shared | Formula | Graph | Saved metadata | DAG evaluator | Direct evaluator |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for scheme, stages in report['schemes'].items():
        values = [f"{stages[k]['median_seconds']:.3f}s" for k in
                  ('formula', 'graph', 'saved_metadata', 'dag', 'direct')]
        lines.append(f"| Central{scheme} | {stages['graph']['metadata']} | " + ' | '.join(values) + ' |')
    lines += ['', 'Graph includes ExprGraph metadata/cache type formation; saved metadata isolates loading '
              'and instantiating the exported graph without evaluator bodies. These independent builds '
              'are not additive timing components.', '',
              'Trace timings are separate, instrumented measurements. Inclusive scopes overlap. '
              'Exclusive recorded time subtracts only recorded children; work below the trace threshold '
              'remains attributed to its parent. Central2 uses 1ms and larger schemes use 100ms, '
              'so fine-grained attribution across schemes is not comparable.', '']
    for scheme, stages in report['schemes'].items():
        for stage in ('graph', 'dag', 'direct'):
            trace = stages[stage]['trace']
            lines += [f'## Central{scheme}: {stage}', '',
                      '| Recorded template family | Exclusive recorded seconds |', '|---|---:|']
            for entry in trace['top_instantiations'][:8]:
                lines.append(f"| {entry['family']} | {entry['exclusive_recorded_seconds']:.3f} |")
            lines.append('')
        bucket_records = [(k, v) for k, v in stages.items() if k.startswith('buckets_')]
        if bucket_records:
            lines += [f'## Central{scheme}: bucket experiment', '',
                      'Every candidate also includes the saved graph header and statically checks exact '
                      'graph type equality. Compare candidates within this experiment.', '',
                      '| Buckets | Median build seconds |', '|---|---:|']
            for key, value in bucket_records:
                lines.append(f"| {key[8:]} | {value['median_seconds']:.3f} |")
            lines.append('')
    lines += ['All evaluator probes passed six analytic checks using the existing benchmark tolerances '
              '(Central2 h=0.002/0.001; other schemes h=0.01/0.005), and direct/DAG agreement within 5e-8. Runtime speed and peak memory '
              'are not measured by this script.', '']
    (output/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Report: {output / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
