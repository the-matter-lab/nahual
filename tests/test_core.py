# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scientific invariants of the two architectures and split filtering."""
import unittest
from dataclasses import replace
import numpy as np
import torch
from nahual.config import Config
from nahual.model import NextTokenDiffusion, DiffusionMLP, create_cu_seqlens
from nahual.data import ChunkBuilder
from nahual.tokens import GEN

class Invariants(unittest.TestCase):
    def config(self):
        return replace(Config(),hidden_size=32,head_dim=8,num_attention_heads=4,num_key_value_heads=2,intermediate_size=64,num_hidden_layers=1,num_coord_transformer_layers=1,diff_w=32,diff_d=2,coord_fourier=8,compile=False)

    def test_packed_documents_do_not_supervise_boundaries(self):
        seq=(np.array([10,20,30]),np.zeros((3,3),dtype=np.float32),0)
        ds=ChunkBuilder([seq,seq,seq],chunk_size=7,shuffle=False,max_len=8)
        b=ds[0]
        self.assertEqual(b[-1].tolist(),[True,True,False,True,True,False,True])
        self.assertEqual(create_cu_seqlens(torch.tensor(b[-2])).tolist(),[0,3,6,7])

    def test_denoisers_are_causal_and_mlp_is_token_independent(self):
        torch.manual_seed(23)
        cfg=self.config(); atoms=torch.full((1,4),GEN+2);xyz=torch.randn(1,4,3);h=torch.randn(1,4,32);sigma=torch.ones(1,4,1)
        for cls in (NextTokenDiffusion,DiffusionMLP):
            model=cls(cfg)
            # Zero-initialized gates and outputs would make this test vacuous.
            with torch.no_grad():
                for n,p in model.named_parameters():
                    if 'adaLN' in n or 'final_layer.linear' in n or 'diff_head.linear' in n:p.normal_(0,0.03)
            pos=model.rotary_emb(torch.arange(4)[None],device='cpu',dtype=torch.float32)
            a=model.denoise(xyz,sigma,atoms,h,pos)
            changed=xyz.clone();changed[:,-1]+=8
            b=model.denoise(changed,sigma,atoms,h,pos)
            torch.testing.assert_close(a[:,:-1],b[:,:-1],atol=1e-6,rtol=1e-6)
            if cls is DiffusionMLP:
                changed=xyz.clone();changed[:,0]+=8
                b=model.denoise(changed,sigma,atoms,h,pos)
                torch.testing.assert_close(a[:,1:],b[:,1:],atol=1e-6,rtol=1e-6)
            a.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_empty_response_loss_is_finite(self):
        cfg=replace(self.config(),response_loss_only=True)
        atoms=torch.tensor([12,GEN+2,14,GEN+3]);xyz=torch.randn(4,3)
        args=(torch.arange(4),atoms,xyz,atoms,xyz,torch.zeros(4,dtype=torch.int32),torch.zeros(4,dtype=torch.bool))
        for cls in (NextTokenDiffusion,DiffusionMLP):
            model=cls(cfg)
            losses=model._forward_impl(*args)
            for loss in losses:
                self.assertTrue(torch.isfinite(loss));self.assertEqual(loss.item(),0)

    def test_trunk_cache_matches_full_causal_forward(self):
        cfg=self.config();m=NextTokenDiffusion(cfg).eval();atoms=torch.tensor([[12,GEN+2,14,GEN+3]]);xyz=torch.randn(1,4,3)
        pos=m.rotary_emb(torch.arange(4)[None],device='cpu',dtype=torch.float32)
        full,_=m.encode1(pos,atoms,xyz)
        cache=None;results=[]
        for i in range(4):
            h,cache=m.encode1(tuple(x[:,i:i+1] for x in pos),atoms[:,i:i+1],xyz[:,i:i+1],past_key_values=cache);results.append(h)
        torch.testing.assert_close(full,torch.cat(results,1),atol=1e-5,rtol=1e-5)


class CheckpointTests(unittest.TestCase):
    def test_meta_loading_restores_nonpersistent_rope(self):
        import tempfile
        from pathlib import Path
        from dataclasses import asdict
        from evals.backends.checkpoint import load_checkpoint_model
        cfg=replace(Invariants().config(),model_impl='next_token')
        m=NextTokenDiffusion(cfg)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.pt'
            torch.save({'config':asdict(cfg),'model_state_dict':m.state_dict()},p)
            restored=load_checkpoint_model(str(p),device='cpu')
        self.assertFalse(any(b.is_meta for b in restored.buffers()))
        torch.testing.assert_close(m.rotary_emb.inv_freq,restored.rotary_emb.inv_freq)


class ReleaseDefaultsTests(unittest.TestCase):
    def test_logging_is_off_and_accumulation_is_one_by_default(self):
        cfg=Config()
        self.assertFalse(cfg.wandb);self.assertEqual(cfg.entity,'');self.assertEqual(cfg.grad_accum_steps,1)

    def test_paper_configs_load_without_logging_or_cluster_hooks(self):
        import json
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        for name in ('next_token_seed0','next_token_seed1','diffmlp_seed0','diffmlp_seed1'):
            values=json.loads((root/f'configs/{name}.json').read_text())
            self.assertFalse(values.get('wandb',False),name);self.assertNotIn('entity',values)
            self.assertFalse(values.get('async_eval',False),name);self.assertFalse(values.get('eval_template'),name)
            cfg=Config.from_dict(values)
            self.assertEqual((cfg.max_steps,cfg.chunk_size,cfg.adam_lr,cfg.ema_decay),(115000,24480,4e-4,0.999))

    def test_varlen_falls_back_to_flex_once_without_torch_support(self):
        import warnings
        from unittest.mock import patch
        import nahual.model as model
        with patch.object(model,'VARLEN_ATTENTION_AVAILABLE',False),patch.object(model,'_warned_varlen_fallback',False):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                self.assertEqual(model.resolve_attention_backend('varlen'),'flex')
                self.assertEqual(model.resolve_attention_backend('varlen'),'flex')
                net=NextTokenDiffusion(replace(Invariants().config(),attention_backend='varlen'))
            self.assertEqual(net.attention_backend,'flex')
            self.assertEqual(sum('varlen' in str(w.message) for w in caught),1)
        with patch.object(model,'VARLEN_ATTENTION_AVAILABLE',True):
            self.assertEqual(model.resolve_attention_backend('varlen'),'varlen')
        with self.assertRaisesRegex(ValueError,'attention_backend'):
            model.resolve_attention_backend('sdpa')


class CheckpointLoadTests(unittest.TestCase):
    def test_training_checkpoint_loads_ema_by_default_and_raw_weights_on_request(self):
        import tempfile
        from pathlib import Path
        from dataclasses import asdict
        from evals.backends.checkpoint import load_checkpoint_model
        cfg=Invariants().config()
        torch.manual_seed(1);model=NextTokenDiffusion(cfg);ema=NextTokenDiffusion(cfg)
        with torch.no_grad():
            for p in ema.parameters():p.normal_(0,0.05)
        with tempfile.TemporaryDirectory() as d:
            ckpt=Path(d)/'step_7.pt'
            torch.save({'config':asdict(cfg),'model_state_dict':model.state_dict(),'ema_state_dict':ema.state_dict(),
                        'optimizer_state_dict':{},'scheduler_state_dict':{},'step':7,'total_tokens':99},ckpt)
            for use_ema,reference in ((True,ema),(False,model)):
                restored=load_checkpoint_model(str(ckpt),device='cpu',ema=use_ema)
                for name,tensor in reference.state_dict().items():
                    self.assertTrue(torch.equal(tensor,restored.state_dict()[name]),name)


if __name__=='__main__':unittest.main()
