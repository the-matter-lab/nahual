# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contributor-facing contracts and boundaries of the evaluation framework."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from evals.api import Generation, TaskConfig
from evals.catalog import discover
from evals.engine import run, score
from evals.jobs.plan import commands, benchmark_tasks
from examples.evaluation.task import ExactText

ROOT = Path(__file__).resolve().parents[1]


class CatalogTests(unittest.TestCase):
    def test_discovery_does_not_import_runtime_dependencies(self):
        script = ('import sys; from evals.catalog import discover; '
                  'assert len(discover()) == 27; '
                  'assert not ({"torch", "numpy", "rdkit", "transformers"} & set(sys.modules))')
        subprocess.run([sys.executable, '-c', script], cwd=ROOT, check=True)

    def test_new_folder_is_discovered_without_registry_changes(self):
        with TemporaryDirectory() as root:
            folder = Path(root) / 'contributed'; folder.mkdir()
            manifest = json.loads((ROOT / 'examples/evaluation/task.json').read_text())
            (folder / 'task.json').write_text(json.dumps(manifest))
            spec = discover(root)['exact_text_example']
            task = spec.create()
            self.assertEqual(task.metric(['yes'], ['yes'])['score'], 1)
            second = Path(root) / 'duplicate'; second.mkdir()
            (second / 'task.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Duplicate task'):
                discover(root)

    def test_every_task_constructs_and_configs_are_isolated(self):
        for name, spec in discover().items():
            with self.subTest(task=name):
                first, second = spec.create(), spec.create()
                first.config.k_values.append(99)
                first.config.extra['changed'] = True
                self.assertNotIn(99, second.config.k_values)
                self.assertNotIn('changed', second.config.extra)
                self.assertEqual(first.config.name, name)

    def test_tasks_do_not_import_other_benchmark_families(self):
        for file in (ROOT / 'evals/tasks').glob('*/*.py'):
            for node in ast.walk(ast.parse(file.read_text())):
                if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('evals.tasks.'):
                    self.assertEqual(node.module.split('.')[2], file.parent.name, str(file))

    def test_checkpoint_plans_preserve_paper_limits_and_refinements(self):
        self.assertEqual(len(benchmark_tasks()), 26)
        request = dict(checkpoint='checkpoint.pt', run_name='test', seed=0, num_examples=None, final=True)
        cond = commands('qm9_edm_cond_mu', request)
        self.assertEqual(len(cond), 2)
        self.assertIn('1000', cond[0][1])
        self.assertEqual(cond[1][0], 'self_oracle')
        self.assertIn('50', commands('geomconf_diverse', request)[0][1])
        self.assertEqual(len(commands('qcg_1to10', request)), 2)
        self.assertEqual(len(commands('qcg_1to10', dict(request, final=False, num_examples=128))), 1)
        self.assertEqual(len(commands('t1x', request)), 1)


class EngineTests(unittest.TestCase):
    def test_backends_share_one_scoring_and_artifact_path(self):
        class PreparedTask(ExactText):
            def load_dataset(self):
                return ['local fixture']
            def post_process(self, results, dataset, save_path):
                self.prepared = dataset
            def artifacts(self):
                return {'extra.pt': {'checked': True}}
        predictions = [dict(prompt_index=0, prompt=['Name H2O'], ground_truth=['water'],
                            samples=[{'prediction':['water']},{'prediction':['methane']}])]
        with TemporaryDirectory() as root:
            for backend_name in ('checkpoint',):
                class FixtureBackend:
                    name = backend_name
                    identity = 'fixture'
                    def generate(self, task, dataset, args):
                        self.dataset = dataset
                        return Generation(deepcopy(predictions), root, 2)
                task = PreparedTask(TaskConfig('fixture','unused',root))
                generated, metrics = run(task, FixtureBackend(), SimpleNamespace(phase=None))
                self.assertEqual(metrics['accuracy'], 0.5)
                self.assertEqual(task.prepared, ['local fixture'])
                self.assertTrue((Path(root) / 'extra.pt').exists())
                metadata = json.loads((Path(root) / 'metrics.json').read_text())
                self.assertEqual(metadata['checkpoint'], 'fixture')

    def test_saved_results_rescore_without_loading_a_model(self):
        from evals.artifacts import save_predictions, load_predictions
        task = ExactText(TaskConfig('fixture','unused','unused'))
        predictions = [dict(prompt_index=0,prompt=[],ground_truth=['a'],samples=[dict(prediction=['a'])])]
        with TemporaryDirectory() as root:
            save_predictions(root, predictions)
            with patch('evals.backends.checkpoint.load_checkpoint_model', side_effect=AssertionError('No model needed')):
                metrics = score(task, load_predictions(root), root, 0, SimpleNamespace(checkpoint=None))
            self.assertEqual(metrics['accuracy'], 1)


class SurfaceAdsorptionMetricTests(unittest.TestCase):
    def test_adsorbate_rmsd_covers_only_the_adsorbate(self):
        import numpy as np
        from evals.metrics.periodic import gmae_metric
        from nahual.tokens import XYZ
        cell = [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 20]]  # origin and lattice vectors
        slab = [[i, j, 0] for i in (1, 4, 7) for j in (1, 4, 7)]
        lattice = [119, 120, 121, 122]
        surface = XYZ(atoms=np.array(lattice + [29] * 9), coords=np.array(cell + slab, dtype=float))
        adsorbate = XYZ(atoms=np.array([8, 1]), coords=np.array([[0, 0, 0], [0.96, 0, 0]]))
        truth = XYZ(atoms=np.array(lattice + [29] * 9 + [8, 1]),
                    coords=np.array(cell + slab + [[4, 4, 2], [4.96, 4, 2]], dtype=float))
        shifted = truth.coords.copy()
        shifted[-2:, 2] += 1.0  # the adsorbate sits 1 Å too high; the slab is exact
        prediction = XYZ(atoms=truth.atoms.copy(), coords=shifted)
        # The prompt gives the adsorbate first, then the surface.
        info = gmae_metric([truth], [prediction], prompt=['adsorbate', adsorbate, 'surface', surface])['info']
        self.assertAlmostEqual(info['ads_rmsd'], 1.0, places=6)
        self.assertAlmostEqual(info['full_rmsd'], (2 / 11) ** 0.5, places=6)
