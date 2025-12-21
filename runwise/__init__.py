"""
Runwise: Token-efficient ML training run analysis for AI agents.

Parses W&B and local logs, generates condensed summaries optimized for LLM context windows.
Includes sparkline visualizations and anomaly detection.
"""

__version__ = "0.5.0"

from .anomalies import Anomaly, AnomalyConfig, detect_anomalies, format_anomalies
from .config import MetricSchema, RunwiseConfig
from .core import RunAnalyzer
from .sparklines import (
    calculate_slope,
    calculate_windowed_slopes,
    sparkline,
    sparkline_with_stats,
    trend_indicator,
)

# TensorBoard support is optional - import separately
# from .tensorboard import TensorBoardParser, TENSORBOARD_AVAILABLE

# W&B API support is optional - import separately
# from .wandb_api import WandbAPIClient, WANDB_API_AVAILABLE

__all__ = [
    # Core
    "RunAnalyzer",
    "RunwiseConfig",
    "MetricSchema",
    # Sparklines
    "sparkline",
    "sparkline_with_stats",
    "trend_indicator",
    "calculate_slope",
    "calculate_windowed_slopes",
    # Anomaly detection
    "Anomaly",
    "AnomalyConfig",
    "detect_anomalies",
    "format_anomalies",
]
