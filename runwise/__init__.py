"""
Runwise: Token-efficient ML training run analysis for AI agents.

Parses W&B and local logs, generates condensed summaries optimized for LLM context windows.
"""

__version__ = "0.1.0"

from .config import MetricSchema, RunwiseConfig
from .core import RunAnalyzer

__all__ = ["RunAnalyzer", "RunwiseConfig", "MetricSchema"]
