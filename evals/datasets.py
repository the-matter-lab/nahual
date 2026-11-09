# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default benchmark data source. Tasks may override Task.load_dataset()."""

def load_dataset_split(name):
    from nahual.prepare import load_raw
    return load_raw(name)
