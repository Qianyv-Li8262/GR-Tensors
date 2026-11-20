"""Export the Bach DAG once, then compile/run cached-graph checks and a bench."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    graph_options = parser.add_mutually_exclusive_group()
    graph_options.add_argument('--rebuild-graph', action='store_true')
    graph_options.add_argument('--reuse-saved-graph', action='store_true',
        help='reuse an exported DAG after confirmed implementation changes preserving node types; never use after formula, lowering or normalization changes')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    output, build = root/'outputs', root/'tests'/'.build'
    output.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    status_path = output/'bach_graph_test_status.json'
    log_path = output/'bach_graph_test.log'
    header_path = output/'bach_compiled_graph_central2.hpp'
    manifest_path = output/'bach_compiled_graph_central2_manifest.json'
    result_path = output/'bach_graph_test_results.json'
    export_exe = build/'bach_graph_export.exe'
    test_exe = build/'bach_graph_test.exe'
    compiler = shutil.which('clang++')
    if not compiler:
        raise SystemExit('clang++ is required')
    graph_inputs = ['examples/bach_formula.hpp', 'examples/bach_metric_reference.hpp',
                    'tensor_expression.hpp', 'expression_graph.hpp', 'expression_graph_operations.hpp',
                    'num_deriv.hpp', 'contraction_plan.hpp', 'contraction_executor.hpp',
                    'contraction_analysis.hpp', 'contraction_and_arithmetic.hpp', 'type_lists.hpp',
                    'Index_and_variance.hpp', 'permutation.hpp', 'symm_asymm.hpp', 'tensor_class.hpp',
                    'graph_type_export.hpp']
    fingerprint = hashlib.sha256()
    for name in graph_inputs:
        fingerprint.update(name.encode())
        fingerprint.update((root/name).read_bytes())
    digest = fingerprint.hexdigest()
    cached = False
    manifest = None
    if header_path.exists() and manifest_path.exists() and not args.rebuild_graph:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        cached = manifest.get('fingerprint') == digest or args.reuse_saved_graph
    if args.reuse_saved_graph and manifest is None:
        parser.error('--reuse-saved-graph requires an existing exported header and manifest')
    flags = ['-std=c++17', '-DNDEBUG', '-fconstexpr-steps=1000000000', '-ftemplate-depth=16384',
             '-fbracket-depth=16384', '-ferror-limit=1', '-ftemplate-backtrace-limit=1', '-I', str(root)]
    now = lambda: datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    # Keep the previous failure evidence before replacing the live status/log.
    previous = [path for path in (status_path, log_path, result_path) if path.exists()]
    if previous:
        archive = output/'bach_graph_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        archive.mkdir(parents=True)
        for path in previous:
            shutil.copy2(path, archive/path.name)
    started = time.perf_counter()
    state = {'state': 'running', 'stage': 'starting', 'started_at': now(), 'runner_pid': os.getpid(),
             'uses_saved_graph': cached, 'graph_reuse_override': args.reuse_saved_graph,
             'log': str(log_path), 'result': str(result_path),
             'saved_graph': str(header_path), 'steps': []}

    def save():
        state['elapsed_seconds'] = round(time.perf_counter()-started, 2)
        state['updated_at'] = now()
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
        temporary.replace(status_path)

    save()
    process = None
    try:
        with log_path.open('w', encoding='utf-8') as log:
            def step(name, command):
                nonlocal process
                state.update(stage=name, command=command)
                log.write('\n=== '+name+' ===\n'+' '.join(command)+'\n')
                log.flush()
                print(name, flush=True)
                step_start = time.perf_counter()
                process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                state['process_pid'] = process.pid
                save()
                while process.poll() is None:
                    time.sleep(2)
                    save()
                state['steps'].append({'stage': name, 'seconds': round(time.perf_counter()-step_start,3),
                                       'exit_code': process.returncode})
                save()
                if process.returncode:
                    raise RuntimeError(f'{name} exited with {process.returncode}; see {log_path}')

            if not cached:
                # Export first, so the costly type graph survives even if a later
                # numerical-code build fails. Never overwrite the original build.
                step('compiling_graph_export', [compiler, *flags, '-O0',
                     str(root/'examples'/'bach_graph_export.cpp'), '-o', str(export_exe)])
                step('exporting_graph_type', [str(export_exe), str(header_path)])
                if not header_path.exists():
                    raise RuntimeError('Graph exporter did not create the saved header')
                manifest_path.write_text(json.dumps({'fingerprint': digest, 'inputs': graph_inputs,
                    'saved_at': now(), 'header_bytes': header_path.stat().st_size}, indent=2), encoding='utf-8')
            linker = ['-Xlinker', '/STACK:134217728'] if os.name == 'nt' else []
            step('compiling_numerical_tests', [compiler, *flags, '-O2',
                 str(root/'examples'/'bach_graph_cached_test.cpp'), *linker, '-o', str(test_exe)])
            step('running_checks_and_benchmark', [str(test_exe), str(result_path)])
            result = json.loads(result_path.read_text(encoding='utf-8'))
            if not result['passed'] or len(result['checks']) != 16:
                raise RuntimeError('Bach numerical checks failed')
            if args.reuse_saved_graph and manifest['fingerprint'] != digest:
                # Only bless the explicitly requested compatible-graph reuse after
                # the numerical comparison and benchmark have actually passed.
                manifest.update(previous_fingerprint=manifest['fingerprint'], fingerprint=digest,
                                compatible_graph_revalidated_at=now())
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            state.update(state='succeeded', stage='finished', finished_at=now(), exit_code=0,
                         nodes=result['nodes'], cached_nodes=result['cached_nodes'], benchmark=result['benchmark'])
            save()
            print(json.dumps(state, ensure_ascii=False), flush=True)
            return 0
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
        state.update(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=repr(error), finished_at=now())
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
