#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PYTHON="${NAHUAL_PYTHON:-python}"
if [[ -z "${NAHUAL_PYTHON:-}" && -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; fi
"$PYTHON" -m nahual.prepare preprocess --workers="${NAHUAL_PREPARE_WORKERS:-8}"
"$PYTHON" -m nahual.prepare audit
