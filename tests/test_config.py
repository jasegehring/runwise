"""Tests for runwise.config module."""

import json
import tempfile
from pathlib import Path

from runwise.config import MetricGroup, MetricSchema, RunwiseConfig


class TestMetricGroup:
    """Tests for MetricGroup dataclass."""

    def test_create_metric_group(self):
        """Test creating a MetricGroup."""
        group = MetricGroup(
            name="training",
            display_name="TRAINING",
            metrics={
                "loss": {"display": "Loss", "format": ".4f"},
                "accuracy": {"display": "Acc", "format": ".1%"},
            }
        )
        assert group.name == "training"
        assert group.display_name == "TRAINING"
        assert "loss" in group.metrics
        assert group.metrics["loss"]["display"] == "Loss"


class TestMetricSchema:
    """Tests for MetricSchema dataclass."""

    def test_default_schema(self):
        """Test default schema creation."""
        schema = MetricSchema.default()

        assert schema.loss_key == "train/loss"
        assert schema.step_key == "_step"
        assert schema.primary_metric == "train/accuracy"
        assert len(schema.groups) == 1
        assert schema.groups[0].name == "training"

    def test_schema_from_file(self):
        """Test creating schema from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_path = Path(tmpdir) / "schema.json"
            data = {
                "loss_key": "custom/loss",
                "step_key": "step",
                "primary_metric": "val/accuracy",
                "primary_metric_name": "Val Accuracy",
                "groups": [
                    {
                        "name": "custom",
                        "display_name": "CUSTOM",
                        "metrics": {
                            "custom/loss": {"display": "Loss", "format": ".4f"},
                        }
                    }
                ],
                "validation_sets": {"val": "Validation"},
            }
            with open(schema_path, "w") as f:
                json.dump(data, f)

            schema = MetricSchema.from_file(schema_path)

            assert schema.loss_key == "custom/loss"
            assert schema.step_key == "step"
            assert schema.primary_metric == "val/accuracy"
            assert len(schema.groups) == 1
            assert schema.groups[0].name == "custom"

    def test_schema_per_step_pattern(self):
        """Test schema with per-step pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_path = Path(tmpdir) / "schema.json"
            data = {
                "loss_key": "loss",
                "step_key": "_step",
                "primary_metric": "accuracy",
                "per_step_pattern": "loss_step_{i}",
                "num_steps": 8,
            }
            with open(schema_path, "w") as f:
                json.dump(data, f)

            schema = MetricSchema.from_file(schema_path)

        assert schema.per_step_pattern == "loss_step_{i}"
        assert schema.num_steps == 8


class TestRunwiseConfig:
    """Tests for RunwiseConfig dataclass."""

    def test_default_config(self):
        """Test default config creation."""
        config = RunwiseConfig()

        assert config.project_name == "ML Project"
        assert config.wandb_dir == Path("wandb")
        assert config.logs_dir == Path("logs")
        assert config.downsample_interval == 1000

    def test_config_with_custom_paths(self):
        """Test config with custom paths."""
        config = RunwiseConfig(
            wandb_dir=Path("/custom/wandb"),
            logs_dir=Path("/custom/logs"),
            project_name="Custom Project",
        )

        assert config.wandb_dir == Path("/custom/wandb")
        assert config.logs_dir == Path("/custom/logs")
        assert config.project_name == "Custom Project"

    def test_config_from_file(self, temp_config_file):
        """Test loading config from file via auto_detect."""
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_config_file.parent)
            config = RunwiseConfig.auto_detect()

            assert config.project_name == "Test Project"
            assert config.schema.loss_key == "train/loss"
            assert config.schema.primary_metric == "val/accuracy"
        finally:
            os.chdir(original_cwd)

    def test_config_save(self):
        """Test saving config to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunwiseConfig(
                project_name="Save Test",
                wandb_dir=Path("wandb"),
                logs_dir=Path("logs"),
            )
            config_path = Path(tmpdir) / "runwise.json"
            config.save(config_path)

            assert config_path.exists()

            with open(config_path) as f:
                data = json.load(f)

            assert data["project_name"] == "Save Test"
            assert "schema_inline" in data

    def test_auto_detect_with_wandb_dir(self, temp_wandb_dir):
        """Test auto-detection with wandb directory."""
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_wandb_dir["wandb_dir"].parent)
            config = RunwiseConfig.auto_detect()
            assert config.wandb_dir.exists()
        finally:
            os.chdir(original_cwd)

    def test_auto_detect_with_config_file(self, temp_config_file):
        """Test auto-detection with config file."""
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_config_file.parent)
            config = RunwiseConfig.auto_detect()
            assert config.project_name == "Test Project"
        finally:
            os.chdir(original_cwd)

    def test_config_schema_property(self):
        """Test that schema property works."""
        config = RunwiseConfig()

        assert config.schema is not None
        assert config.schema.loss_key == "train/loss"
