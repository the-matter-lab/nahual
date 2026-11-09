# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a sequence of text/XYZ segments from a JSON prompt."""
import argparse
import json
import re
import numpy as np
import torch
from evals.backends.checkpoint import load_checkpoint_model
from nahual.tokens import STOP, XYZ, make_seq, parse_seq, apply_chat_template

SPECIAL = re.compile(r'<\|[a-z_]+\|>')  # chat markers such as <|im_end|>


def generate(model, prompt, samples=1, steps=50, temperature=1.0, max_new_tokens=512, device='cuda'):
    """Sample responses to a prompt (a list of text/xyz segment dicts).

    Returns one list of segment dicts per sample.
    """
    atoms, coords = apply_chat_template(make_seq(prompt), samples)
    prefix = (torch.as_tensor(atoms, device=device), torch.as_tensor(coords, device=device))
    with torch.no_grad(), torch.autocast(device.split(':')[0], dtype=torch.bfloat16):
        result = model.generate(prefix=prefix, bsz=samples, max_len=max_new_tokens, keep_prefix=False,
                                num_steps=steps, temperature=temperature)
    out = []
    for aa, xx in zip(result[0].cpu().numpy(), result[1].cpu().numpy()):
        end = np.flatnonzero(aa == STOP)  # the response ends at the first stop token
        if len(end):
            aa, xx = aa[:end[0]], xx[:end[0]]
        segments = []
        for s in parse_seq(aa, xx):
            if isinstance(s, XYZ):
                segments.append({'type': 'xyz', 'atoms': s.atoms.tolist(), 'coords': s.coords.tolist()})
            elif SPECIAL.sub('', s).strip():
                segments.append({'type': 'text', 'text': SPECIAL.sub('', s).strip()})
        out.append(segments)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--prompt', required=True, help='JSON file: a list of text/xyz segments')
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=1)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-ema', action='store_true')
    a=p.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    model=load_checkpoint_model(a.checkpoint,a.device,not a.no_ema)
    out=generate(model,json.load(open(a.prompt)),a.samples,a.steps,a.temperature,a.max_new_tokens,a.device)
    with open(a.output,'w') as f:json.dump(out,f,indent=2)

if __name__=='__main__':main()
