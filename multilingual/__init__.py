"""Standalone multilingual claim preprocessing for TruthCheck.

Nothing in this package imports the existing application pipeline.  Integrators
can call :class:`MultilingualProcessor` and decide how to use its output.
"""

from .processor import MultilingualProcessor, MultilingualProcessorConfig
from .types import MultilingualClaim, TokenTag

__all__ = [
    "MultilingualClaim",
    "MultilingualProcessor",
    "MultilingualProcessorConfig",
    "TokenTag",
]
