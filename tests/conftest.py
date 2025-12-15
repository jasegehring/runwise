"""Pytest fixtures for runwise tests."""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_wandb_dir():
    """Create a temporary W&B directory structure with mock data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wandb_dir = Path(tmpdir) / "wandb"
        wandb_dir.mkdir()

        # Create two mock runs
        run1_dir = wandb_dir / "run-20251214_100000-abc123"
        run1_files = run1_dir / "files"
        run1_files.mkdir(parents=True)

        run2_dir = wandb_dir / "run-20251214_110000-def456"
        run2_files = run2_dir / "files"
        run2_files.mkdir(parents=True)

        # Run 1: Finished run with good metrics
        with open(run1_files / "wandb-summary.json", "w") as f:
            json.dump({
                "_step": 10000,
                "_runtime": 3600,
                "train/loss": 0.25,
                "train/accuracy": 0.92,
                "val/loss": 0.30,
                "val/accuracy": 0.88,
            }, f)

        with open(run1_files / "config.yaml", "w") as f:
            f.write("""learning_rate:
  value: 0.001
batch_size:
  value: 64
model:
  value: transformer
dropout:
  value: 0.1
""")

        with open(run1_files / "wandb-metadata.json", "w") as f:
            json.dump({
                "displayName": "baseline-run",
                "notes": "Testing baseline configuration with default hyperparameters",
                "tags": ["baseline", "v1"],
                "group": "initial-experiments",
                "exitcode": 0,
            }, f)

        # Create history file
        with open(run1_files / "wandb-history.jsonl", "w") as f:
            for step in range(0, 10001, 100):
                loss = 1.0 - (step / 10000) * 0.75
                acc = 0.5 + (step / 10000) * 0.42
                f.write(json.dumps({
                    "_step": step,
                    "train/loss": loss,
                    "train/accuracy": acc,
                }) + "\n")

        # Run 2: Different config, slightly worse metrics
        with open(run2_files / "wandb-summary.json", "w") as f:
            json.dump({
                "_step": 8000,
                "_runtime": 2800,
                "train/loss": 0.35,
                "train/accuracy": 0.85,
                "val/loss": 0.40,
                "val/accuracy": 0.80,
            }, f)

        with open(run2_files / "config.yaml", "w") as f:
            f.write("""learning_rate:
  value: 0.01
batch_size:
  value: 32
model:
  value: transformer
dropout:
  value: 0.2
""")

        with open(run2_files / "wandb-metadata.json", "w") as f:
            json.dump({
                "displayName": "high-lr-test",
                "notes": "Testing higher learning rate",
                "tags": ["experiment", "lr-test"],
                "exitcode": 0,
            }, f)

        # Create latest-run symlink
        latest_link = wandb_dir / "latest-run"
        latest_link.symlink_to(run2_dir)

        yield {
            "wandb_dir": wandb_dir,
            "run1_id": "abc123",
            "run2_id": "def456",
            "run1_dir": run1_dir,
            "run2_dir": run2_dir,
        }


@pytest.fixture
def temp_logs_dir():
    """Create a temporary logs directory with mock log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logs_dir = Path(tmpdir) / "logs"
        logs_dir.mkdir()

        # Create a JSONL log file
        with open(logs_dir / "training.jsonl", "w") as f:
            for step in range(0, 1001, 10):
                f.write(json.dumps({
                    "step": step,
                    "loss": 1.0 - (step / 1000) * 0.7,
                    "accuracy": 0.5 + (step / 1000) * 0.4,
                }) + "\n")

        yield logs_dir


@pytest.fixture
def temp_config_file():
    """Create a temporary runwise.json config file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "runwise.json"
        config_data = {
            "project_name": "Test Project",
            "wandb_dir": "wandb",
            "logs_dir": "logs",
            "schema_inline": {
                "loss_key": "train/loss",
                "step_key": "_step",
                "primary_metric": "val/accuracy",
                "primary_metric_name": "Val Accuracy",
                "groups": [
                    {
                        "name": "training",
                        "display_name": "TRAINING",
                        "metrics": {
                            "train/loss": {"display": "Loss", "format": ".4f"},
                            "train/accuracy": {"display": "Accuracy", "format": ".1%"},
                        }
                    }
                ],
                "validation_sets": {"val": "Validation"},
            }
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        yield config_path
