# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Asset paths. Large assets live outside the source tree and can be set with NAHUAL_* environment variables."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
qwen3_dirs = {"0.6B": os.environ.get("NAHUAL_QWEN3_0pt6B_DIR", str(ROOT / "models/Qwen3-0.6B"))}
processed_dir = Path(os.environ.get("NAHUAL_PROCESSED_DIR", ROOT / "data/processed"))
raw_dir = Path(os.environ.get("NAHUAL_RAW_DIR", ROOT / "data/raw"))
asset_dir = Path(os.environ.get("NAHUAL_ASSET_DIR", ROOT / "data"))
