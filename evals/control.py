# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cooperative interruption without depending on any model runtime."""
import signal
import threading


class InterruptGuard:
    def __init__(self):
        self.requested = False
        self.previous = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
                self.previous[sig] = signal.getsignal(sig)
                signal.signal(sig, self._request)

    def _request(self, *_):
        self.requested = True

    def check(self):
        if self.requested:
            raise InterruptedError('Evaluation interrupted; completed batches are saved for resume')

    def close(self):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)
