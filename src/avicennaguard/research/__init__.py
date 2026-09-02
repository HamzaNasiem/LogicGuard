"""
Research evaluation adapters bridging legacy experiment scripts to the package.

Exports ResearchValidator for running evaluations on legacy step formats.
"""

from avicennaguard.research.adapter import ResearchValidator

__all__ = [
    "ResearchValidator",
]
