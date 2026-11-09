# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified CLI: list, describe, run, and score benchmark results.

    python -m evals run --task TASK --checkpoint DIR          # one task, one GPU
    python -m evals run --protocol paper --checkpoint DIR \
        --run-name NAME [--gpus N]                              # every paper task
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from evals.catalog import discover_tasks, get_spec


def run_protocol(argv):
    """Run every paper task on its full test set with the paper's sample budgets,
    one worker per GPU. Rerunning the same command resumes and skips completed tasks."""
    from evals.jobs.plan import benchmark_tasks
    p = argparse.ArgumentParser(prog='python -m evals run --protocol paper', description=run_protocol.__doc__)
    p.add_argument('--protocol', choices=['paper'], required=True)
    p.add_argument('--checkpoint', required=True, help='Exported checkpoint directory or step_N.pt')
    p.add_argument('--run-name', '--run_name', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpus', type=int, help='GPUs to use (default: all visible GPUs)')
    p.add_argument('--tasks', nargs='+', help='Subset of the paper tasks (default: all)')
    p.add_argument('--num-examples', '--num_examples', type=int,
                   help='Smoke test only: the first N examples of each task instead of the full test set')
    p.add_argument('--request-dir', default='evals/results/requests',
                   help='Where the request and per-task logs (<run-name>.results/) are kept')
    a = p.parse_args(argv)
    request = dict(checkpoint=str(Path(a.checkpoint).resolve()), run_name=a.run_name, seed=a.seed,
                   protocol='paper', final=True, num_examples=None)
    tasks = a.tasks or benchmark_tasks(request)
    unknown = sorted(set(tasks) - set(benchmark_tasks(request)))
    if unknown:
        p.error(f'not paper tasks: {unknown}')
    if a.num_examples:
        request['task_indices'] = {task: list(range(a.num_examples)) for task in tasks}
    path = Path(a.request_dir) / f'{a.run_name}.json'
    if path.exists() and json.loads(path.read_text()) != request:
        p.error(f'{path} holds a different request; choose a new --run-name')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, indent=2) + '\n')
    from evals.jobs.worker import main as worker
    worker(['--request', str(path), '--seconds', str(10 ** 9), '--tasks', *tasks]
           + (['--gpus', str(a.gpus)] if a.gpus else []))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list', 'describe', 'run', 'score'])
    # run/score flags, including --help, are parsed by the backend's own parser.
    if not argv or argv[0] in {'-h', '--help'}:
        parser.print_help()
        return
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument('--task-dir', action='append', default=[])
    extra_args, argv = extra.parse_known_args(argv)
    catalog = discover_tasks(extra_args.task_dir)
    command = parser.parse_args(argv[:1]).command
    if command == 'list':
        p = argparse.ArgumentParser(description='List tasks without loading models or scientific dependencies')
        p.add_argument('--json', action='store_true')
        a = p.parse_args(argv[1:])
        specs = catalog
        if a.json:
            print(json.dumps([asdict(s) for s in specs.values()], indent=2))
        else:
            for name, spec in specs.items():
                print(f'{name:30} {",".join(spec.backends):24} {spec.description}')
        return
    if command == 'describe':
        p = argparse.ArgumentParser(); p.add_argument('task')
        print(json.dumps(asdict(get_spec(p.parse_args(argv[1:]).task)), indent=2))
        return
    if command == 'run' and any(arg == '--protocol' or arg.startswith('--protocol=') for arg in argv):
        return run_protocol(argv[1:])
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument('--backend', choices=['checkpoint'], default='checkpoint')
    options, rest = p.parse_known_args(argv[1:])
    if command == 'score':
        rest += ['--phase', 'metrics']
    from evals.checkpoint_cli import get_parser, run_eval_pipeline
    run_eval_pipeline(get_parser().parse_args(rest))


if __name__ == '__main__':
    main()
