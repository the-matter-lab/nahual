# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from nahual.prepare import FILES,atomic_json,identity,sha256,verify_audit

class AuditTests(unittest.TestCase):
    def test_changed_data_or_mixture_is_rejected(self):
        spec={'train':{'train':1},'validation':['val']}
        with TemporaryDirectory() as d:
            root=Path(d);dataset=root/'train';dataset.mkdir()
            for f in FILES:(dataset/f).write_bytes(b'original')
            atomic_json(dataset/'manifest.json',{'fixture':True})
            report={'status':'passed','mixture':spec,'processed':{'train':{'manifest_sha256':sha256(dataset/'manifest.json'),'files':{f:identity(dataset/f) for f in FILES}}}}
            atomic_json(root/'audit.json',report);verify_audit(spec,root)
            with self.assertRaises(ValueError):verify_audit({'train':{'train':2}},root)
            (dataset/'tokens.npy').write_bytes(b'changed data')
            with self.assertRaises(ValueError):verify_audit(spec,root)
