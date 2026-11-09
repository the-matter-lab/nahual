# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn task metadata into subprocess stages for the evaluation worker."""
import json
import sys
from evals.catalog import discover_tasks, get_spec, resolve


def benchmark_tasks(request=None):
    """The benchmark's tasks. Tasks with checkpoint.routine=false run only on final requests."""
    final = request is None or request.get('final', True)
    return [name for name, spec in discover_tasks().items()
            if spec.paper and (final or spec.checkpoint.get('routine', True))]


def _example_limit(spec, request):
    if spec.name in request.get('task_indices', {}):
        return len(request['task_indices'][spec.name])
    if request.get('protocol') == 'paper':  # Full test sets, no caps.
        return spec.config.num_examples
    limit = request['num_examples']
    if limit is None:
        limit = spec.checkpoint.get('final_examples', spec.config.num_examples)
    cap = spec.checkpoint.get('max_examples')
    if cap is not None:
        limit = cap if limit is None else min(cap, limit)
    return limit


def _protocol(spec, request):
    """Task setting overrides (e.g. dataset, samples_per_prompt) for paper-protocol requests."""
    settings = dict(spec.checkpoint.get('paper_protocol', {})) if request.get('protocol') == 'paper' else {}
    settings.update(request.get('task_overrides', {}).get(spec.name, {}))
    return settings


def commands(name, request):
    spec = get_spec(name)
    spec.require_backend('checkpoint')
    limit = _example_limit(spec, request)
    cmd = [sys.executable, '-m', 'evals', 'run', '--backend', 'checkpoint',
           '--task', name, '--checkpoint', request['checkpoint'],
           '--run_name', request['run_name'], '--seed', str(request['seed']), '--num_steps', '50']
    if limit is not None:
        cmd += ['--num_examples', str(limit)]
    for key, value in _protocol(spec, request).items():
        if key == 'task_options':
            value = json.dumps(value, sort_keys=True)
            key = 'task-options'
        elif isinstance(value, list):
            value = ','.join(map(str, value))
        cmd = _with(cmd, '--' + key, str(value))
    if name in request.get('task_indices', {}):
        cmd += ['--indices', ','.join(map(str, request['task_indices'][name]))]
    stages = [('evaluate', cmd)]
    refinement = spec.checkpoint.get('refinement')
    if refinement:
        stages.extend(resolve(refinement)(spec, request, limit))
    return stages


def balanced_shards(costs, n):
    """Heaviest-first greedy assignment of example indices to n shards."""
    shards, loads = [[] for _ in range(n)], [0] * n
    for i in sorted(range(len(costs)), key=lambda i: -costs[i]):
        j = loads.index(min(loads))
        shards[j].append(i)
        loads[j] += costs[i]
    return [sorted(s) for s in shards if s]


def units(name, request):
    """Schedulable units for one task, as (unit, stages, required units).

    With checkpoint.shard_examples=S, generation is split into ceil(examples/S)
    shards over disjoint indices, each with its own resumable run name, followed by
    one merge-and-score unit. Shards are cost-balanced if the task defines
    compute_costs, otherwise interleaved.
    """
    spec = get_spec(name)
    stages = commands(name, request)
    size = request.get('task_shard_examples', {}).get(name, request.get('shard_examples') or spec.checkpoint.get('shard_examples'))
    limit = _example_limit(spec, request)
    if not size or (limit is not None and limit <= size):
        return [(name, stages, ())]
    from evals.datasets import load_dataset_split
    dataset = load_dataset_split(_protocol(spec, request).get('dataset', spec.config.dataset))
    selected = request.get('task_indices', {}).get(name)
    if selected is None:
        selected = list(range(len(dataset) if limit is None else min(limit, len(dataset))))
    count = len(selected)
    n = -(-count // size)
    if n <= 1:
        return [(name, stages, ())]
    task = spec.create()
    if hasattr(task, 'compute_costs'):
        # compute_costs sees the selected rows. Shards keep the original dataset indices.
        subset = dataset.select(selected) if hasattr(dataset, 'select') else [dataset[i] for i in selected]
        shards = [[selected[i] for i in shard] for shard in balanced_shards(task.compute_costs(subset, count), n)]
    else:
        shards = [selected[i::n] for i in range(min(n, count))]
    base, out = stages[0][1], []
    names = [f"{request['run_name']}_shard{i}of{len(shards)}" for i in range(len(shards))]
    for i, indices in enumerate(shards):
        shard_cmd = _with(_with(base, '--run_name', names[i]), '--indices', ','.join(map(str, indices)))
        out.append((f'{name}.shard{i}', [('generate', shard_cmd + ['--phase', 'generate'])], ()))
    merge = _with(base, '--run_name', request['run_name']) + [
        '--phase', 'metrics', '--merge_shards', ','.join(names)]
    out.append((name, [('merge', merge)] + stages[1:], tuple(u for u, _, _ in out)))
    return out


def _with(cmd, flag, value):
    cmd = list(cmd)
    if flag in cmd:
        cmd[cmd.index(flag) + 1] = value
    else:
        cmd += [flag, value]
    return cmd
