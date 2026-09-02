"""AvicennaGuard two-stage pipeline: Stage 1 (neural parser) + Stage 2 (BFS validator)."""

from avicennaguard.pipeline.avicennaguard import AvicennaGuard

LogicGuard = AvicennaGuard  # Backward-compatibility alias

__all__ = ["AvicennaGuard", "LogicGuard"]
