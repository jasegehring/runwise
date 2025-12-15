"""
Configuration for Runwise.

Defines metric schemas that map project-specific metrics to standardized categories.
Includes configurable anomaly detection thresholds.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .anomalies import AnomalyConfig


@dataclass
class MetricGroup:
    """A group of related metrics (e.g., 'training', 'validation')."""
    name: str
    display_name: str
    metrics: dict[str, dict]  # metric_key -> {display_name, format, higher_is_better}


@dataclass
class MetricSchema:
    """
    Schema defining how to interpret and display metrics for a project.

    This allows Runwise to work with any ML project by mapping their
    specific metric names to standardized categories.
    """
    # Core training metrics
    loss_key: str = "train/loss"
    step_key: str = "train/step"

    # Primary accuracy metrics (for run list display)
    primary_metric: str = "train/accuracy"
    primary_metric_name: str = "Accuracy"

    # Metric groups for detailed analysis
    groups: list[MetricGroup] = field(default_factory=list)

    # Per-step metrics (for iterative/recursive models)
    per_step_pattern: Optional[str] = None  # e.g., "train/ce_step_{i}"
    num_steps: int = 0

    # Validation sets
    validation_sets: dict[str, str] = field(default_factory=dict)  # prefix -> display_name

    # Custom analysis functions
    custom_analyzers: list[Callable] = field(default_factory=list)

    @classmethod
    def default(cls) -> "MetricSchema":
        """Default schema for generic ML training."""
        return cls(
            loss_key="train/loss",
            step_key="_step",
            primary_metric="train/accuracy",
            primary_metric_name="Accuracy",
            groups=[
                MetricGroup(
                    name="training",
                    display_name="TRAINING",
                    metrics={
                        "train/loss": {"display": "Loss", "format": ".4f", "higher_is_better": False},
                        "train/accuracy": {"display": "Accuracy", "format": ".1%", "higher_is_better": True},
                    }
                ),
            ],
            validation_sets={
                "val": "Validation",
                "test": "Test",
            }
        )

    @classmethod
    def from_file(cls, path: Path) -> "MetricSchema":
        """Load schema from JSON file."""
        with open(path) as f:
            data = json.load(f)

        groups = []
        for g in data.get("groups", []):
            groups.append(MetricGroup(
                name=g["name"],
                display_name=g["display_name"],
                metrics=g["metrics"]
            ))

        return cls(
            loss_key=data.get("loss_key", "train/loss"),
            step_key=data.get("step_key", "_step"),
            primary_metric=data.get("primary_metric", "train/accuracy"),
            primary_metric_name=data.get("primary_metric_name", "Accuracy"),
            groups=groups,
            per_step_pattern=data.get("per_step_pattern"),
            num_steps=data.get("num_steps", 0),
            validation_sets=data.get("validation_sets", {}),
        )


@dataclass
class RunwiseConfig:
    """
    Main configuration for Runwise.

    Can be loaded from:
    1. runwise.json in project root
    2. Environment variables (RUNWISE_WANDB_DIR, etc.)
    3. Explicit paths
    """
    # Paths
    wandb_dir: Path = field(default_factory=lambda: Path("wandb"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))

    # Project info
    project_name: str = "ML Project"

    # Metric schema
    schema: MetricSchema = field(default_factory=MetricSchema.default)

    # Anomaly detection settings
    anomaly_config: AnomalyConfig = field(default_factory=AnomalyConfig)

    # Output settings
    downsample_interval: int = 1000  # For training curves
    max_validation_history: int = 10  # How many val results to show

    @classmethod
    def auto_detect(cls, project_root: Optional[Path] = None) -> "RunwiseConfig":
        """
        Auto-detect configuration from project structure.

        Looks for:
        1. runwise.json in project root
        2. wandb/ directory
        3. logs/ directory
        """
        if project_root is None:
            project_root = Path.cwd()

        config = cls()

        # Check for config file
        config_file = project_root / "runwise.json"
        if config_file.exists():
            with open(config_file) as f:
                data = json.load(f)

            config.project_name = data.get("project_name", config.project_name)
            config.downsample_interval = data.get("downsample_interval", config.downsample_interval)

            if "wandb_dir" in data:
                config.wandb_dir = project_root / data["wandb_dir"]
            if "logs_dir" in data:
                config.logs_dir = project_root / data["logs_dir"]

            if "schema" in data:
                config.schema = MetricSchema.from_file(project_root / data["schema"])
            elif "schema_inline" in data:
                # Inline schema in config file
                schema_data = data["schema_inline"]
                groups = []
                for g in schema_data.get("groups", []):
                    groups.append(MetricGroup(
                        name=g["name"],
                        display_name=g["display_name"],
                        metrics=g["metrics"]
                    ))
                config.schema = MetricSchema(
                    loss_key=schema_data.get("loss_key", "train/loss"),
                    step_key=schema_data.get("step_key", "_step"),
                    primary_metric=schema_data.get("primary_metric", "train/accuracy"),
                    primary_metric_name=schema_data.get("primary_metric_name", "Accuracy"),
                    groups=groups,
                    per_step_pattern=schema_data.get("per_step_pattern"),
                    num_steps=schema_data.get("num_steps", 0),
                    validation_sets=schema_data.get("validation_sets", {}),
                )

            # Load anomaly detection settings
            if "anomaly_detection" in data:
                ad = data["anomaly_detection"]
                config.anomaly_config = AnomalyConfig(
                    spike_threshold=ad.get("spike_threshold", 3.5),
                    spike_window=ad.get("spike_window", 100),
                    overfit_ratio_threshold=ad.get("overfit_ratio_threshold", 1.5),
                    overfit_baseline_steps=tuple(ad.get("overfit_baseline_steps", [100, 500])),
                    plateau_min_steps=ad.get("plateau_min_steps", 500),
                    plateau_improvement_threshold=ad.get("plateau_improvement_threshold", 0.01),
                    gradient_vanish_threshold=ad.get("gradient_vanish_threshold", 1e-7),
                    gradient_explode_multiplier=ad.get("gradient_explode_multiplier", 10.0),
                    gradient_sustained_steps=ad.get("gradient_sustained_steps", 10),
                    throughput_drop_threshold=ad.get("throughput_drop_threshold", 0.4),
                )

        # Override from environment
        if env_wandb := os.environ.get("RUNWISE_WANDB_DIR"):
            config.wandb_dir = Path(env_wandb)
        if env_logs := os.environ.get("RUNWISE_LOGS_DIR"):
            config.logs_dir = Path(env_logs)

        # Auto-detect directories if not set
        if not config.wandb_dir.exists():
            for candidate in [project_root / "wandb", Path.home() / "wandb"]:
                if candidate.exists():
                    config.wandb_dir = candidate
                    break

        if not config.logs_dir.exists():
            for candidate in [project_root / "logs", project_root / "outputs"]:
                if candidate.exists():
                    config.logs_dir = candidate
                    break

        return config

    def save(self, path: Path):
        """Save configuration to file."""
        data = {
            "project_name": self.project_name,
            "wandb_dir": str(self.wandb_dir),
            "logs_dir": str(self.logs_dir),
            "downsample_interval": self.downsample_interval,
            "schema_inline": {
                "loss_key": self.schema.loss_key,
                "step_key": self.schema.step_key,
                "primary_metric": self.schema.primary_metric,
                "primary_metric_name": self.schema.primary_metric_name,
                "per_step_pattern": self.schema.per_step_pattern,
                "num_steps": self.schema.num_steps,
                "validation_sets": self.schema.validation_sets,
                "groups": [
                    {
                        "name": g.name,
                        "display_name": g.display_name,
                        "metrics": g.metrics,
                    }
                    for g in self.schema.groups
                ]
            },
            "anomaly_detection": {
                "spike_threshold": self.anomaly_config.spike_threshold,
                "spike_window": self.anomaly_config.spike_window,
                "overfit_ratio_threshold": self.anomaly_config.overfit_ratio_threshold,
                "overfit_baseline_steps": list(self.anomaly_config.overfit_baseline_steps),
                "plateau_min_steps": self.anomaly_config.plateau_min_steps,
                "plateau_improvement_threshold": self.anomaly_config.plateau_improvement_threshold,
                "gradient_vanish_threshold": self.anomaly_config.gradient_vanish_threshold,
                "gradient_explode_multiplier": self.anomaly_config.gradient_explode_multiplier,
                "gradient_sustained_steps": self.anomaly_config.gradient_sustained_steps,
                "throughput_drop_threshold": self.anomaly_config.throughput_drop_threshold,
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
