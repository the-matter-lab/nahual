# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task discovery from task.json manifests. Task code is imported only when a task is created."""
from copy import deepcopy
from dataclasses import dataclass, field
from importlib import import_module
import json
from pathlib import Path
from typing import Any
from evals.api import Task, TaskConfig


@dataclass(frozen=True)
class TaskSpec:
    name: str
    factory: str
    config: TaskConfig
    description: str = ''
    paper: bool = False
    backends: tuple[str, ...] = ('checkpoint',)
    response: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] = field(default_factory=dict)

    def create(self, **overrides) -> Task:
        config = deepcopy(self.config)
        for key, value in overrides.items():
            if key not in TaskConfig.__dataclass_fields__:
                raise ValueError(f'Unknown task setting: {key}')
            setattr(config, key, value)
        task = resolve(self.factory)(config)
        if not isinstance(task, Task):
            raise TypeError(f'{self.factory} must create an evals.api.Task')
        return task

    def require_backend(self, name):
        if name not in self.backends:
            raise ValueError(f'{self.name} supports {", ".join(self.backends)}, not {name}')


def resolve(reference):
    module, sep, name = reference.partition(':')
    if not sep or not module or not name:
        raise ValueError(f'Expected module:callable, got {reference!r}')
    return getattr(import_module(module), name)


def discover(root=None) -> dict[str, TaskSpec]:
    """Read manifests without importing task code or GPU/chemistry dependencies."""
    root = Path(root) if root else Path(__file__).parent / 'tasks'
    catalog = {}
    for manifest in sorted(root.glob('*/task.json')):
        document = json.loads(manifest.read_text())
        if document.get('schema_version') != 1:
            raise ValueError(f'Unsupported task schema: {manifest}')
        for raw in document['tasks']:
            entry = deepcopy(raw)
            metadata = {k: entry.pop(k) for k in ('factory','description','paper','backends','response','checkpoint') if k in entry}
            name = entry['name']
            if name in catalog:
                raise ValueError(f'Duplicate task {name!r}: {manifest}')
            if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
                raise ValueError(f'Unsafe task name: {name!r}')
            entry.setdefault('output_dir', f'evals/results/{name}')
            config = TaskConfig(**entry)
            for key in ('max_new_tokens','samples_per_prompt','prompt_batch_size','num_steps'):
                if getattr(config,key) <= 0:
                    raise ValueError(f'{manifest}: {key} must be positive')
            if 'backends' in metadata:
                metadata['backends'] = tuple(metadata['backends'])
            catalog[name] = TaskSpec(name=name, config=config, **metadata)
    return catalog


TASK_REGISTRY: dict[str, TaskSpec] = {}


def discover_tasks(extra_roots=()):
    specs = discover()
    for root in extra_roots:
        extra = discover(root)
        duplicates = set(specs) & set(extra)
        if duplicates:
            raise ValueError(f"Duplicate contributed tasks: {sorted(duplicates)}")
        specs.update(extra)
    TASK_REGISTRY.clear()
    TASK_REGISTRY.update(specs)
    return TASK_REGISTRY


def get_spec(name):
    if not TASK_REGISTRY:
        discover_tasks()
    try:
        return TASK_REGISTRY[name]
    except KeyError:
        raise ValueError(f'Unknown task {name!r}. Available: {", ".join(sorted(TASK_REGISTRY))}') from None


def get_task(name, **overrides):
    return get_spec(name).create(**overrides)
