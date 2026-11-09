# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from evals.tasks.conformers import geomconf_diverse as geom


class GeometryBudgetTests(unittest.TestCase):
    def test_checkpoint_restores_random_streams(self):
        with TemporaryDirectory() as directory:
            torch.manual_seed(17)
            np.random.seed(17)
            geom._save_checkpoint(directory, [], [], [0], 'config')
            expected_torch = torch.rand(8)
            expected_numpy = np.random.random(8)
            expected_cuda = torch.rand(8, device='cuda') if torch.cuda.is_available() else None
            torch.manual_seed(99)
            np.random.seed(99)
            self.assertEqual(geom._load_checkpoint(directory, 'config'), ([], []))
            torch.testing.assert_close(torch.rand(8), expected_torch, rtol=0, atol=0)
            np.testing.assert_array_equal(np.random.random(8), expected_numpy)
            if expected_cuda is not None:
                torch.testing.assert_close(torch.rand(8, device='cuda'), expected_cuda, rtol=0, atol=0)

    def test_legacy_resume_keeps_completed_molecules_but_restarts_partial_ones(self):
        completed = [{'prompt_index': 1}]
        partial = [{'prompt_index': 2}]
        with TemporaryDirectory() as directory:
            torch.save(dict(completed=completed, in_progress=partial, config_hash='config'),
                       Path(directory) / 'checkpoint.pt')
            self.assertEqual(geom._load_checkpoint(directory, 'config'), (completed, []))

    def test_batch_respects_remaining_conformer_attempts(self):
        # Ten conformers per sequence, mostly invalid, with 64 attempts left.
        state = dict(samples=[{}] * 20, n_valid=20, n_total_conformers=200,
                     target=132, max_attempts=264)
        self.assertEqual(geom._rows_needed(state, 160), 7)

    def test_near_target_does_not_add_speculative_headroom(self):
        state = dict(samples=[{}] * 20, n_valid=120, n_total_conformers=200,
                     target=132, max_attempts=264)
        self.assertEqual(geom._rows_needed(state, 160), 2)

    def test_invalid_sequences_exhaust_generation_budget(self):
        config=SimpleNamespace(prompt_batch_size=2,max_new_tokens=8,num_steps=1,
                               num_examples=1,name='geomconf_diverse')
        row={'instruction':[{'type':'text','text':'C'}],
             'response':[{'type':'text','text':'1'},
                         {'type':'xyz','atoms':[6],'coords':[[0,0,0]]}]}
        handler=SimpleNamespace(register=lambda callback:None,triggered=False,
                                cancel_timer=lambda:None)
        calls = []
        def invalid(model, prompts, *args):
            calls.append(len(prompts))
            if sum(calls) > 4:
                raise AssertionError('Unbounded generation')
            return [[] for _ in prompts]
        with patch.object(geom,'_generate_rows',side_effect=invalid) as generate:
            with patch.object(geom,'_SIGTERMHandler',return_value=handler),patch.object(geom,'tokenize_seq',return_value=(np.array([1]),np.zeros((1,3)))):
                results=geom.GeomconfDiverseTask(config).custom_generate(None,[row],config,SimpleNamespace(device='cpu'))
        self.assertEqual(sum(calls),4)
        self.assertEqual(len(results[0]['samples']),4)
