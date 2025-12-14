"""
Runwise: Token-efficient ML training run analysis for AI agents.

Parses W&B and local logs, generates condensed summaries optimized for LLM context windows.
"""

__version__ = "0.1.0"

from .core import RunAnalyzer
from .config import RunwiseConfig, MetricSchema

__all__ = ["RunAnalyzer", "RunwiseConfig", "MetricSchema"]
