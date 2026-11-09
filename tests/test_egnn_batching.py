# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched EGNN scoring matches per-molecule scoring and keeps sample order."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from evals.tasks.conditional_molecules import predictor as p


class EGNNBatchTests(unittest.TestCase):
    def test_tensor_layout_matches_legacy_scalar_construction(self):
        atoms = [np.array([6, 1, 8]), np.array([7, 9])]
        coords = [np.arange(9).reshape(3, 3), np.arange(6).reshape(2, 3)]
        batch = p.xyz_to_egnn_tensors(atoms, coords, 'cpu')
        for i in range(2):
            scalar = p.xyz_to_egnn_tensors(atoms[i:i+1], coords[i:i+1], 'cpu')
            for key in ['nodes', 'atom_positions', 'atom_mask']:
                torch.testing.assert_close(batch[key][i*29:(i+1)*29], scalar[key])
            torch.testing.assert_close(batch['edge_mask'][i*29*29:(i+1)*29*29], scalar['edge_mask'])
            for edge, scalar_edge in zip(batch['edges'], scalar['edges']):
                torch.testing.assert_close(edge[i*29*29:(i+1)*29*29] - i*29, scalar_edge)
        self.assertEqual(batch['edge_mask'].sum().item(), 8)

    def test_invalid_slots_and_order_survive_batching(self):
        xyz = lambda n: SimpleNamespace(atoms=np.ones(n, dtype=int), coords=np.zeros((n, 3)))
        inputs = [xyz(2), None, xyz(30), xyz(3), xyz(1)]
        def score(model, atoms, coords, mean, mad, prop, device):
            return np.array([len(a) for a in atoms], dtype=float)
        with patch.object(p, 'predict_property_prompt_units', side_effect=score) as infer:
            values = p.score_molecules(None, inputs, 0, 1, 'alpha', 'cpu', batch_size=2)
        self.assertEqual(values, [2., None, None, 3., 1.])
        self.assertEqual(infer.call_count, 2)

    def test_model_errors_are_not_silently_scored_as_invalid_molecules(self):
        xyz = SimpleNamespace(atoms=np.array([6]), coords=np.zeros((1, 3)))
        with patch.object(p, 'predict_property_prompt_units', side_effect=RuntimeError('device failure')):
            with self.assertRaisesRegex(RuntimeError, 'device failure'):
                p.score_molecules(None, [xyz], 0, 1, 'alpha', 'cpu')


if __name__ == '__main__':
    unittest.main()
