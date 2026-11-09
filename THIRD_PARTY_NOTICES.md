# Third-party notices

Nahual is licensed under the Apache License 2.0 (`LICENSE`). Some files adapt code
from the projects below, under their own licenses.

| Code in this repository | Source | License |
|---|---|---|
| `nahual/model.py`: Qwen3 layers (`Qwen3RMSNorm`, `Qwen3MLP`, rotary embedding, `Qwen3Attention`) | [transformers](https://github.com/huggingface/transformers), `models/qwen3/modeling_qwen3.py` | Apache-2.0, Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team |
| `nahual/model.py`: `ResBlock`, `DiffMLPFinalLayer` | [MAR](https://github.com/LTH14/mar), `models/diffloss.py` | MIT, Copyright (c) 2024 Tianhong Li |
| `nahual/model.py`: `MPFourier` | [EDM2](https://github.com/NVlabs/edm2), `training/networks_edm2.py` | Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES |
| `nahual/inorganic.py` | [DiffCSP](https://github.com/jiaor17/DiffCSP), `scripts/eval_utils.py` | MIT, Copyright (c) 2023 Rui Jiao |
| `nahual/metrics.py`: `check_validity_smiles` | [symphony](https://github.com/atomicarchitects/symphony), `analyses/metrics.py` | MIT, Copyright (c) 2024 The Atomic Architects |
| `nahual/metrics.py`: covalent radii and bonding rule of `get_bond_graph` | [3Dmol.js](https://github.com/3dmol/3Dmol.js) | BSD-3-Clause, Copyright (c) 2014, University of Pittsburgh and contributors |
| `evals/tasks/conditional_molecules/egnn/` (unchanged) and `evals/tasks/conditional_molecules/splits.py` | [EDM](https://github.com/ehoogeboom/e3_diffusion_for_molecules) | MIT, Copyright (c) 2022 Emiel Hoogeboom, Victor Garcia Satorras, Clément Vignac |

`scripts/viewer/` loads 3Dmol.js (BSD-3-Clause) from a CDN at run time; it is not
included here.

## MIT License

Applies to the MAR, DiffCSP, symphony and EDM code listed above, with the copyright
lines given in the table.

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## BSD-3-Clause License (3Dmol.js)

```
* 3Dmol.js
	Copyright (c) 2014, University of Pittsburgh and contributors
	All rights reserved.

	Redistribution and use in source and binary forms, with or without 
	modification, are permitted provided that the following conditions are met:

	1. Redistributions of source code must retain the above copyright 
	notice, this list of conditions and the following disclaimer.

	2. Redistributions in binary form must reproduce the above copyright 
	notice, this list of conditions and the following disclaimer in the 
	documentation and/or other materials provided with the distribution.

	3. Neither the name of the copyright holder nor the names of its 
	contributors may be used to endorse or promote products derived from 
	this software without specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
	"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
	LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
	PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
	HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
	SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
	TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
	PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
	LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
