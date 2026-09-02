"""
Backward-compatibility alias module for AvicennaGuard pipeline.

LogicGuard was the initial project codename, now standardized to AvicennaGuard.
This module re-exports AvicennaGuard and LogicGuard to preserve 100% backward
compatibility with existing scripts and external imports.
"""

from avicennaguard.pipeline.avicennaguard import AvicennaGuard, LogicGuard

__all__ = [
    "AvicennaGuard",
    "LogicGuard",
]
