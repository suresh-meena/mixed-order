"""Experiments package initializer.

Ensures `src/` is available on `sys.path` so `mixed_order` imports work when
chapter experiment modules are run directly.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

__all__ = []
