"""Build the Bach DAG and persist progress so a separate shell can query it."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    root = Path(__file__).resolve().parent.parent
    output = root / 'outputs'
    build = root / 'tests' / '.build'
    output.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    status_path = output / 'bach_graph_build_status.json'
    log_path = output / 'bach_graph_build.log'
    executable = build / 'bach_graph_central2.exe'
    compiler = shutil.which('clang++')
    if not compiler:
        raise SystemExit('clang++ is required')
    command = [compiler, '-std=c++17', '-O2', '-DNDEBUG',
               '-fconstexpr-steps=1000000000', '-ftemplate-depth=16384', '-fbracket-depth=16384',
               '-ferror-limit=1', '-ftemplate-backtrace-limit=1', '-I', str(root),
               str(root / 'examples' / 'bach_graph.cpp'), '-o', str(executable)]
    local_time = lambda: datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    started = time.perf_counter()
    status = {'state': 'starting', 'scheme': 'Central2', 'derivative_depth': 4,
              'runner_pid': os.getpid(), 'started_at': local_time(), 'command': command,
              'log': str(log_path), 'executable': str(executable), 'elapsed_seconds': 0}

    def save():
        status['updated_at'] = local_time()
        status['elapsed_seconds'] = round(time.perf_counter() - started, 2)
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding='utf-8')
        temporary.replace(status_path)

    save()
    process = None
    try:
        with log_path.open('w', encoding='utf-8') as log:
            log.write(' '.join(command) + '\n\n')
            log.flush()
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            status.update(state='running', compiler_pid=process.pid)
            save()
            print(f'Bach DAG compiler started: PID {process.pid}; status: {status_path}', flush=True)
            while process.poll() is None:
                time.sleep(2)
                save()
            status.update(state='succeeded' if process.returncode == 0 else 'failed',
                          exit_code=process.returncode, finished_at=local_time())
            if process.returncode == 0:
                status['executable_bytes'] = executable.stat().st_size
            save()
            print(json.dumps(status, ensure_ascii=False), flush=True)
            return process.returncode
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
        status.update(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error=repr(error), finished_at=local_time())
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
