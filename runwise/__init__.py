"""
Runwise: Token-efficient ML training run analysis for AI agents.

Parses W&B and local logs, generates condensed summaries optimized for LLM context windows.
Includes sparkline visualizations and anomaly detection.
"""

__version__ = "0.2.2"

from .anomalies import Anomaly, AnomalyConfig, detect_anomalies, format_anomalies
from .config import MetricSchema, RunwiseConfig
from .core import RunAnalyzer
from .sparklines import sparkline, sparkline_with_stats, trend_indicator

# TensorBoard support is optional - import separately
# from .tensorboard import TensorBoardParser, TENSORBOARD_AVAILABLE

__all__ = [
    # Core
    "RunAnalyzer",
    "RunwiseConfig",
    "MetricSchema",
    # Sparklines
    "sparkline",
    "sparkline_with_stats",
    "trend_indicator",
    # Anomaly detection
    "Anomaly",
    "AnomalyConfig",
    "detect_anomalies",
    "format_anomalies",
]
