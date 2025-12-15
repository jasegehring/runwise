"""Tests for runwise.core module."""

import pytest
from pathlib import Path

from runwise.core import RunAnalyzer, RunInfo
from runwise.config import RunwiseConfig


class TestRunInfo:
    """Tests for RunInfo dataclass."""

    def test_default_values(self):
        """Test RunInfo has correct default values."""
        run = RunInfo(
            run_id="test123",
            directory=Path("/tmp/test"),
            date="2025-01-01",
            time="12:00:00",
        )
        assert run.metrics == {}
        assert run.config == {}
        assert run.tags == []
        assert run.state == "unknown"
        assert run.name == ""
        assert run.notes == ""
        assert run.group == ""

    def test_post_init_none_handling(self):
        """Test that None values are converted to empty containers."""
        run = RunInfo(
            run_id="test123",
            directory=Path("/tmp/test"),
            date="2025-01-01",
            time="12:00:00",
            metrics=None,
            config=None,
            tags=None,
        )
        assert run.metrics == {}
        assert run.config == {}
        assert run.tags == []


class TestRunAnalyzer:
    """Tests for RunAnalyzer class."""

    def test_list_runs(self, temp_wandb_dir):
        """Test listing runs from wandb directory."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        runs = analyzer.list_runs(limit=10)
        assert len(runs) == 2

        # Runs should be sorted by date/time descending
        assert runs[0].run_id == "def456"  # More recent
        assert runs[1].run_id == "abc123"

    def test_get_latest_run(self, temp_wandb_dir):
        """Test getting latest run via symlink."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        latest = analyzer.get_latest_run()
        assert latest is not None
        assert latest.run_id == "def456"

    def test_find_run(self, temp_wandb_dir):
        """Test finding specific run by ID."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        run = analyzer.find_run("abc123")
        assert run is not None
        assert run.run_id == "abc123"

        # Test partial match
        run = analyzer.find_run("abc")
        assert run is not None
        assert run.run_id == "abc123"

        # Test not found
        run = analyzer.find_run("nonexistent")
        assert run is None

    def test_parse_run_metadata(self, temp_wandb_dir):
        """Test that run metadata is parsed correctly."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        run = analyzer.find_run("abc123")
        assert run.name == "baseline-run"
        assert run.notes == "Testing baseline configuration with default hyperparameters"
        assert run.tags == ["baseline", "v1"]
        assert run.group == "initial-experiments"
        assert run.state == "finished"

    def test_parse_run_metrics(self, temp_wandb_dir):
        """Test that run metrics are parsed correctly."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        run = analyzer.find_run("abc123")
        assert run.final_step == 10000
        assert run.metrics["train/loss"] == 0.25
        assert run.metrics["train/accuracy"] == 0.92

    def test_parse_run_config(self, temp_wandb_dir):
        """Test that run config is parsed correctly."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        run = analyzer.find_run("abc123")
        assert run.config["learning_rate"] == 0.001
        assert run.config["batch_size"] == 64
        assert run.config["model"] == "transformer"

    def test_summarize_run(self, temp_wandb_dir):
        """Test run summary generation."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        run = analyzer.find_run("abc123")
        summary = analyzer.summarize_run(run)

        assert "abc123" in summary
        assert "10,000" in summary  # Step count
        assert "baseline-run" in summary  # Name
        assert "baseline, v1" in summary  # Tags

    def test_format_run_list(self, temp_wandb_dir):
        """Test run list formatting."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        runs = analyzer.list_runs()
        output = analyzer.format_run_list(runs)

        assert "abc123" in output
        assert "def456" in output
        assert "FINISHED" in output
        assert "baseline-run" in output  # Name shown

    def test_compare_runs(self, temp_wandb_dir):
        """Test run comparison."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        run_a = analyzer.find_run("abc123")
        run_b = analyzer.find_run("def456")
        comparison = analyzer.compare_runs(run_a, run_b)

        assert "abc123" in comparison
        assert "def456" in comparison
        assert "train/loss" in comparison
        assert "Delta" in comparison

    def test_compare_runs_with_filter(self, temp_wandb_dir):
        """Test run comparison with metric filter."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        run_a = analyzer.find_run("abc123")
        run_b = analyzer.find_run("def456")
        comparison = analyzer.compare_runs(run_a, run_b, filter_prefix="val")

        assert "val/loss" in comparison
        assert "train/loss" not in comparison

    def test_compare_runs_with_config_diff(self, temp_wandb_dir):
        """Test run comparison with config diff."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        run_a = analyzer.find_run("abc123")
        run_b = analyzer.find_run("def456")
        comparison = analyzer.compare_runs(run_a, run_b, show_config_diff=True)

        assert "CONFIG DIFFERENCES" in comparison
        assert "learning_rate" in comparison
        assert "batch_size" in comparison

    def test_get_run_context(self, temp_wandb_dir):
        """Test getting full run context."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        run = analyzer.find_run("abc123")
        context = analyzer.get_run_context(run)

        assert "baseline-run" in context
        assert "Testing baseline configuration" in context
        assert "baseline, v1" in context
        assert "initial-experiments" in context

    def test_get_config(self, temp_wandb_dir):
        """Test getting run config."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        run = analyzer.find_run("abc123")
        config_output = analyzer.get_config(run)

        assert "learning_rate" in config_output
        assert "0.001" in config_output
        assert "batch_size" in config_output

    def test_get_history(self, temp_wandb_dir):
        """Test getting downsampled history."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        run = analyzer.find_run("abc123")
        history = analyzer.get_history(run, ["train/loss", "train/accuracy"], samples=10)

        assert "step,train/loss,train/accuracy" in history
        lines = history.strip().split("\n")
        assert len(lines) <= 12  # Header + up to 10 samples + possible message

    def test_get_history_stats(self, temp_wandb_dir):
        """Test getting history statistics."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        run = analyzer.find_run("abc123")
        stats = analyzer.get_history_stats(run, ["train/loss"])

        assert "train/loss" in stats
        assert "Min" in stats
        assert "Max" in stats
        assert "Mean" in stats

    def test_list_available_keys(self, temp_wandb_dir):
        """Test listing available metric keys."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        run = analyzer.find_run("abc123")
        keys = analyzer.list_available_keys(run)

        assert "train/loss" in keys
        assert "train/accuracy" in keys

    def test_find_best_run(self, temp_wandb_dir):
        """Test finding best run by metric."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        # Lower is better (loss)
        best, ranked = analyzer.find_best_run("train/loss", limit=10, higher_is_better=False)
        assert best.run_id == "abc123"  # Has lower loss

        # Higher is better (accuracy)
        best, ranked = analyzer.find_best_run("train/accuracy", limit=10, higher_is_better=True)
        assert best.run_id == "abc123"  # Has higher accuracy

    def test_get_live_status(self, temp_wandb_dir):
        """Test getting live status."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)

        status = analyzer.get_live_status()
        assert "LIVE TRAINING STATUS" in status
        assert "Run ID: def456" in status  # Latest run ID shown


class TestLocalLogs:
    """Tests for local log file handling."""

    def test_list_local_logs(self, temp_logs_dir):
        """Test listing local log files."""
        config = RunwiseConfig(logs_dir=temp_logs_dir)
        analyzer = RunAnalyzer(config)

        logs = analyzer.list_local_logs()
        assert len(logs) == 1
        assert logs[0].name == "training.jsonl"

    def test_parse_local_log(self, temp_logs_dir):
        """Test parsing local log file."""
        config = RunwiseConfig(logs_dir=temp_logs_dir)
        analyzer = RunAnalyzer(config)

        log_file = temp_logs_dir / "training.jsonl"
        records = analyzer.parse_local_log(log_file)

        assert len(records) == 101  # 0 to 1000 by 10
        assert records[0]["step"] == 0
        assert records[-1]["step"] == 1000

    def test_summarize_local_log(self, temp_logs_dir):
        """Test summarizing local log file."""
        config = RunwiseConfig(logs_dir=temp_logs_dir)
        analyzer = RunAnalyzer(config)

        log_file = temp_logs_dir / "training.jsonl"
        summary = analyzer.summarize_local_log(log_file)

        assert "LOCAL LOG" in summary
        assert "training.jsonl" in summary
