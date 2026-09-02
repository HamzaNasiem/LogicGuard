"""
Core epistemic state definitions and data structures for AvicennaGuard.

Exports the four-state Avicennian epistemic classification (YAQEEN, WAHM,
SHAKK, ZANN), syllogistic query types, and the Stage 2 validation result container.
"""

from avicennaguard.core.epistemic_states import (
    EpistemicState,
    QueryType,
    ValidatorResult,
)

__all__ = [
    "EpistemicState",
    "QueryType",
    "ValidatorResult",
]
