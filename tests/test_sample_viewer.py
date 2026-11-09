# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The browser viewer turns data rows and saved samples into chat messages, with unit cells."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from nahual.tokens import XYZ

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('viewer_server', ROOT / 'scripts/viewer/server.py')
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)

WATER = XYZ(atoms=np.array([8, 1, 1]), coords=np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]], float))
SALT = XYZ(atoms=np.array([119, 120, 121, 122, 11, 17]),
           coords=np.array([[0, 0, 0], [5.6, 0, 0], [0, 5.6, 0], [0, 0, 5.6], [0, 0, 0], [2.8, 2.8, 2.8]], float))


class ViewerTests(unittest.TestCase):
    def test_segments_of_strings_structures_and_data_rows(self):
        row = [{'type': 'text', 'text': 'Generate', 'atoms': None, 'coords': None},
               {'type': 'xyz', 'text': None, 'atoms': [8, 1, 1], 'coords': WATER.coords.ravel().tolist()}]
        text, water = viewer.segments(row)
        self.assertEqual(text, {'type': 'text', 'text': 'Generate'})
        self.assertEqual((water['xyz'].splitlines()[0], water['formula']), ('3', 'H2O'))
        salt = viewer.segment(SALT)
        self.assertEqual(salt['xyz'].splitlines()[2:], ['Na 0.0000 0.0000 0.0000', 'Cl 2.8000 2.8000 2.8000'])
        self.assertEqual(salt['cell']['vectors'], [[5.6, 0, 0], [0, 5.6, 0], [0, 0, 5.6]])
        self.assertIsNone(salt['cell']['vacuum'])
        self.assertEqual(salt['atoms'][:4], [119, 120, 121, 122])  # kept, so the prompt can be sent back
        # A 4 x 4 x 2 slab in the bottom fifth of a tall cell, wrapped across the cell boundary along c.
        grid = np.array([[i, j, k] for i in range(4) for j in range(4) for k in range(2)], float)
        slab = (grid * [2.5, 2.5, 2.0] + [0, 0, 29.0]) % [10, 10, 30]
        self.assertEqual(viewer.vacuum_axis(slab, np.diag([10.0, 10.0, 30.0])), 2)
        self.assertIsNone(viewer.vacuum_axis(slab[:8], np.diag([10.0, 10.0, 30.0])))  # too few atoms to tell
        self.assertEqual(viewer.segments(['', '  ', '<|im_end|><|im_end|>']), [])
        self.assertEqual(viewer.segments(['1.53 D<|im_end|>']), [{'type': 'text', 'text': '1.53 D'}])
        # A response ends at the first stop marker: later structures are discarded.
        self.assertEqual(len(viewer.segments([SALT, '<|im_end|>.', WATER, 'more'])), 1)
        self.assertEqual(viewer.formula([6, 1, 1, 1, 17]), 'CH3Cl')
        self.assertEqual(viewer.title(['Relax  this\nmolecule:', WATER]), 'Relax this molecule: H2O')
        self.assertEqual(viewer.title([WATER]), 'H2O')

    def test_sample_files_are_listed_by_run_and_task_in_both_layouts(self):
        with tempfile.TemporaryDirectory() as d:
            hub, results = Path(d) / 'samples', Path(d) / 'results'
            for path in (hub / 'nahual/seed0/t1x', hub / 'nahual/seed0/qcg_1to10_xtb_relaxed_crude',
                         hub / 'diffmlp/seed1/csp', results / 't1x/my_run', results / 'requests/x'):
                path.mkdir(parents=True)
                (path / 'predictions.pt').touch()
            runs = viewer.sample_runs([hub.resolve(), results.resolve()])
            self.assertEqual({r: sorted(t) for r, t in runs.items()},
                             {'nahual/seed0': ['qcg_1to10_xtb_relaxed_crude', 't1x'], 'diffmlp/seed1': ['csp'],
                              'my_run': ['t1x']})

    def test_samples_are_read_without_torch_and_only_under_sample_roots(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'csp' / 'predictions.pt'
            path.parent.mkdir()
            entries = [{'prompt_index': i, 'prompt': [f'Generate crystal {i}'], 'ground_truth': [SALT],
                        'samples': [{'prediction': [SALT], 'metric': {'score': 0.5}},
                                    {'prediction': ['no structure'], 'metric': {'score': float('inf')}}]}
                       for i in range(3)]
            entries[2]['ground_truth'] = {'responses': [[SALT], [WATER]]}  # several references (ODAC)
            torch.save(entries, path)
            self.assertIsInstance(viewer.load_predictions(path)[0]['ground_truth'][0], viewer.XYZ)
            store = viewer.Store([d])
            source = 'samples:' + str(path.resolve())
            self.assertEqual([s['id'] for s in store.sources()['sources'] if s['id'].startswith('samples:')], [source])
            self.assertEqual(store.items(source, 1, 10), {'count': 3, 'offset': 1,
                                                          'titles': ['Generate crystal 1', 'Generate crystal 2']})
            page = store.example(source, 5)
            self.assertEqual((page['count'], page['index'], len(page['references'])), (3, 2, 2))
            self.assertEqual([s['note'] for s in page['samples']], ['does not match the reference structure'] * 2)
            self.assertEqual(viewer.parse_message(page['message'])[0], {'type': 'text', 'text': 'Generate crystal 2'})
            with self.assertRaises(PermissionError):
                viewer.Store([Path(d) / 'other']).example(source, 0)

    def test_data_rows_come_from_parquet_row_groups(self):
        import json
        import pyarrow as pa
        import pyarrow.parquet as pq
        segment = lambda text, atoms=None, coords=None: {'type': 'xyz' if atoms else 'text', 'text': text,
                                                         'atoms': atoms, 'coords': coords}
        rows = [{'instruction': [segment(f'Relax molecule {i}'), segment(None, [8, 1, 1], WATER.coords.ravel().tolist())],
                 'response': [segment(None, [8, 1, 1], WATER.coords.ravel().tolist())]} for i in range(5)]
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'splits/demo').mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(rows), Path(d) / 'splits/demo/test-0.parquet', row_group_size=2)
            (Path(d) / 'splits/grouped').mkdir()
            pq.write_table(pa.Table.from_pylist([{k: json.dumps(v) for k, v in rows[0].items()}]),
                           Path(d) / 'splits/grouped/test-0.parquet')
            (Path(d) / 'manifest.json').write_text(json.dumps({'splits': {
                'demo': {'roles': ['test'], 'rows': 5, 'files': {'splits/demo/test-0.parquet': ''}},
                'grouped': {'roles': ['train'], 'rows': 1, 'files': {'splits/grouped/test-0.parquet': ''}}}}))
            saved, viewer.raw_dir = viewer.raw_dir, Path(d)
            try:
                store = viewer.Store([])
                self.assertEqual(store.sources()['sources'], [
                    {'id': 'data:grouped', 'group': 'Data · train', 'label': 'grouped', 'count': 1},
                    {'id': 'data:demo', 'group': 'Data · test', 'label': 'demo', 'count': 5}])
                self.assertEqual(store.sources()['source'], 'data:grouped')  # NCI Atlas train when present, else the first
                self.assertEqual(store.items('data:demo', 1, 3)['titles'], [f'Relax molecule {i} H2O' for i in (1, 2, 3)])
                for source, index in (('data:demo', 3), ('data:grouped', 0)):  # segments as lists or JSON strings
                    page = store.example(source, index)
                    self.assertEqual((page['prompt'][0]['text'], page['references'][0][0]['formula']),
                                     (f'Relax molecule {index}', 'H2O'))
            finally:
                viewer.raw_dir = saved

    def test_each_sample_metric_is_named(self):
        describe = viewer.describe
        self.assertEqual(describe('t1x', {'metric': {'score': 0.1534, 'valid': True, 'info': {}}}, []), 'RMSD to reference 0.153 Å')
        self.assertEqual(describe('csp', {'metric': {'score': 0.05, 'valid': True, 'info': {'match': True, 'rms': 0.05}}}, []),
                         'matches the reference structure, normalized RMS distance 0.050')
        self.assertEqual(describe('gmae_fg', {'metric': {'score': 0.8, 'valid': True, 'info': {'ads_rmsd': 0.8, 'full_rmsd': 0.9}}}, []),
                         'adsorbate RMSD 0.800 Å · full RMSD 0.900 Å')
        self.assertEqual(describe('qm9_edm_cond_gap', {'egnn_pred': 8359.123, 'self_pred': 8261.4, 'metric': {}}, []),
                         'EGNN oracle property 8359.1 meV · self-predicted 8261.4 meV')
        self.assertEqual(describe('coremof_pore_volume', {'prediction': ['0.76 cm3/g'], 'metric': {'score': 0.0, 'valid': True}},
                                  ['0.82 cm3/g']), 'absolute error 0.06 cm3/g')
        self.assertEqual(describe('nci', {'metric': {'score': float('inf'), 'valid': False, 'info': {'msg': 'Atom mismatch'}}}, []),
                         'not scored: Atom mismatch')

    def test_messages_round_trip_structures_and_cells(self):
        prompt = viewer.segments(['Relax this crystal', SALT, 'and this molecule', WATER])
        parsed = viewer.parse_message(viewer.message(prompt))
        self.assertEqual([p['type'] for p in parsed], ['text', 'xyz', 'text', 'xyz'])
        self.assertEqual(parsed[1]['atoms'], [119, 120, 121, 122, 11, 17])
        np.testing.assert_allclose(parsed[1]['coords'], SALT.coords, atol=1e-5)
        self.assertEqual(parsed[2], {'type': 'text', 'text': 'and this molecule'})
        typed = viewer.parse_message('Optimize\n```xyz\n2\nH2\nH 0 0 0\nh 0 0 0.74\n```')
        self.assertEqual(typed[1]['atoms'], [1, 1])
        with self.assertRaises(ValueError):
            viewer.parse_message('```xyz\n3\n\nH 0 0 0\n```')

    def test_checkpoints_beside_the_samples_default_to_nahual_seed0(self):
        with tempfile.TemporaryDirectory() as d:
            for model in ('diffmlp/seed0', 'nahual/seed0'):
                (Path(d) / 'checkpoints' / model).mkdir(parents=True)
                (Path(d) / 'checkpoints' / model / 'step_115000.pt').touch()
            (Path(d) / 'samples').mkdir()
            store = viewer.Store([Path(d) / 'samples'], device='cpu')
            labels = {c['label']: c['path'] for c in store.sources()['checkpoints']}
            self.assertIn('diffmlp/seed0', labels)
            self.assertEqual(store.default, labels['nahual/seed0'])

    def test_server_starts_with_the_standard_library_only(self):
        import subprocess
        import sys
        code = ('import runpy, sys; runpy.run_path(sys.argv[1], run_name="viewer"); '
                'print(sorted(m for m in ("numpy", "torch", "transformers", "pyarrow") if m in sys.modules))')
        out = subprocess.run([sys.executable, '-c', code, str(ROOT / 'scripts/viewer/server.py')],
                             capture_output=True, text=True, check=True).stdout
        self.assertEqual(out.strip(), '[]')


if __name__ == '__main__':
    unittest.main()
