# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public benchmark interface.

A task defines the question and its scoring. A backend runs the model.
Predictions are saved as lists of Prediction dicts.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, NotRequired


class Sample(TypedDict):
    prediction: list[Any]
    metric: NotRequired[dict[str, Any]]


class Prediction(TypedDict):
    prompt_index: int
    prompt: list[Any]
    ground_truth: Any
    samples: list[Sample]


@dataclass
class TaskConfig:
    name: str
    dataset: str
    output_dir: str
    max_new_tokens: int = 240
    samples_per_prompt: int = 10
    prompt_batch_size: int = 16
    num_examples: int | None = None
    num_steps: int = 50
    k_values: list[int] = field(default_factory=lambda: [1, 3, 10])
    default_temperature: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Generation:
    predictions: list[Prediction]
    save_path: str | None
    duration: float
    peak_gpu_memory_gb: float | None = None


class Backend(Protocol):
    name: str

    def generate(self, task: 'Task', dataset: Any, args: Any) -> Generation: ...


class Task(ABC):
    """Base class for a benchmark. Subclasses implement metric and aggregate.

    Optional hooks: setup runs once before scoring. post_process adds fields such as
    energies to samples. load_dataset can load data outside the data release.
    custom_generate(model, dataset, config, args), if set, replaces the default
    checkpoint generation loop.
    """
    custom_generate = None

    def __init__(self, config: TaskConfig):
        self.config = config

    def load_dataset(self):
        from evals.datasets import load_dataset_split
        return load_dataset_split(self.config.dataset)

    def setup(self, args=None) -> None:
        pass

    def post_process(self, results: list[Prediction], dataset: Any, save_path: str) -> None:
        pass

    @abstractmethod
    def metric(self, ground_truth, prediction, prompt=None) -> dict[str, Any]: ...

    @abstractmethod
    def aggregate(self, results: list[Prediction], duration: float) -> dict[str, Any]: ...

    def artifacts(self) -> dict[str, Any]:
        """Optional extra .pt artifacts, keyed by filename (without directories)."""
        return {}


class RMSDTask(Task):
    """Reusable benchmark for one molecular geometry; configured entirely in JSON."""
    def metric(self, ground_truth, prediction, prompt=None):
        from evals.metrics.geometry import rmsd_metric
        return rmsd_metric(ground_truth, prediction)

    def aggregate(self, results, duration):
        from evals.metrics.geometry import topk_rmsd_aggregation
        return topk_rmsd_aggregation(results, self.config.k_values, duration)
