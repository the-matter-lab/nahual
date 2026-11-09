# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one evaluation request on this machine, one worker per GPU.

`python -m evals run --protocol paper` writes the request and starts this worker:
    python -m evals.jobs.worker --request request.json [--gpus 1] [--seconds N]
Completed tasks are skipped when the same request is run again.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from nahual.prepare import atomic_json
from evals.jobs.plan import benchmark_tasks, units


def run_process(command, env, log, stop, deadline):
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    while process.poll() is None:
        if stop.wait(2) or time.monotonic() >= deadline:
            stop.set()
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            return None
    return process.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--seconds', type=int, help='Stop starting new work after this many seconds')
    parser.add_argument('--gpus', type=int, default=None,
                        help='GPUs to use (default: those in CUDA_VISIBLE_DEVICES, else every GPU)')
    args = parser.parse_args(argv)
    visible = [d for d in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if d.strip()]
    if not visible:
        import torch
        visible = [str(i) for i in range(torch.cuda.device_count())]
    devices = visible
    if args.gpus is not None:
        devices = (visible or [str(i) for i in range(args.gpus)])[:args.gpus]
    if not devices:
        parser.error('no GPUs selected')
    request = Path(args.request)
    spec = json.loads(request.read_text())
    state = request.with_suffix('.results')
    state.mkdir(exist_ok=True)
    stop = threading.Event()
    deadline = time.monotonic() + (args.seconds or float('inf'))
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    tasks = args.tasks or benchmark_tasks(spec)
    # Build every plan before starting work. Sharded units go first because they take longest.
    plans = [unit for task in tasks for unit in units(task, spec)]
    plans.sort(key=lambda unit: '.shard' not in unit[0])
    stages = {name: unit_stages for name, unit_stages, _ in plans}
    pending = [(name, needs) for name, _, needs in plans
               if not (state / (name + '.done')).exists()]
    lock = threading.Lock()
    errors = []

    def done(name):
        return (state / (name + '.done')).exists()

    def claim():
        """Next runnable unit, 'wait' while one awaits running units, or None."""
        with lock:
            failed = {e['task'] for e in errors}
            for item in list(pending):
                name, needs = item
                if failed & set(needs):
                    pending.remove(item)
                elif all(done(n) for n in needs):
                    pending.remove(item)
                    return name
            return 'wait' if pending else None

    def worker(gpu):
        while not stop.is_set() and time.monotonic() < deadline:
            task = claim()
            if task is None:
                return
            if task == 'wait':
                stop.wait(10)
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1',
                       PYTORCH_ALLOC_CONF='expandable_segments:True')
            with open(state / (task + '.log'), 'a') as log:
                for phase, command in stages[task]:
                    code = run_process(command, env, log, stop, deadline)
                    if code is None:
                        return
                    if code:
                        with lock:
                            errors.append({'task': task, 'phase': phase, 'exit_code': code})
                        break
                else:
                    (state / (task + '.done')).write_text('complete\n')

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(worker, devices))
    completed = [task for task in tasks if (state / (task + '.done')).exists()]
    status = 'failed' if errors else ('complete' if len(completed) == len(tasks) else 'partial')
    atomic_json(request.with_suffix('.status'), dict(status=status, completed=completed,
                errors=errors))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
