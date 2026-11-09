# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from nahual.train import train, parse_args_from_config

if __name__ == "__main__":
    train(parse_args_from_config())
