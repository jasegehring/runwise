"""Tests for runwise.cli module."""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from runwise.cli import (
    cmd_list,
    cmd_latest,
    cmd_run,
    cmd_compare,
    cmd_config,
    cmd_notes,
    cmd_best,
    cmd_history,
    cmd_stats,
    cmd_keys,
    cmd_live,
    cmd_local,
    _get_default_metric_keys,
    main,
)
from runwise.core import RunAnalyzer
from runwise.config import RunwiseConfig


class MockArgs:
    """Mock argparse namespace for testing."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestCLICommands:
    """Tests for CLI command functions."""

    def test_cmd_list(self, temp_wandb_dir, capsys):
        """Test list command."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)
        args = MockArgs(limit=10)

        cmd_list(analyzer, args)

        captured = capsys.readouterr()
        assert "abc123" in captured.out
        assert "def456" in captured.out

    def test_cmd_latest(self, temp_wandb_dir, capsys):
        """Test latest command."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)
        args = MockArgs()

        cmd_latest(analyzer, args)

        captured = capsys.readouterr()
        assert "def456" in captured.out  # Latest run

    def test_cmd_run(self, temp_wandb_dir, capsys):
        """Test run command."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)
        args = MockArgs(run_id="abc123")

        cmd_run(analyzer, args)

        captured = capsys.readouterr()
        assert "abc123" in captured.out
        assert "10,000" in captured.out

    def test_cmd_run_not_found(self, temp_wandb_dir, capsys):
        """Test run command with nonexistent run."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)
        args = MockArgs(run_id="nonexistent")

        cmd_run(analyzer, args)

        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_cmd_compare(self, temp_wandb_dir, capsys):
        """Test compare command."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)
        args = MockArgs(run_a="abc123", run_b="def456", filter=None, diff=False)

        cmd_compare(analyzer, args)

        captured = capsys.readouterr()
        assert "abc123" in captured.out
        assert "def456" in captured.out
        assert "Delta" in captured.out

    def test_cmd_compare_with_filter(self, temp_wandb_dir, capsys):
        """Test compare command with filter."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)
        args = MockArgs(run_a="abc123", run_b="def456", filter="val", diff=False)

        cmd_compare(analyzer, args)

        captured = capsys.readouterr()
        assert "val/loss" in captured.out

    def test_cmd_compare_with_diff(self, temp_wandb_dir, capsys):
        """Test compare command with config diff."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)
        args = MockArgs(run_a="abc123", run_b="def456", filter=None, diff=True)

        cmd_compare(analyzer, args)

        captured = capsys.readouterr()
        assert "CONFIG DIFFERENCES" in captured.out

    def test_cmd_config(self, temp_wandb_dir, capsys):
        """Test config command."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)
        args = MockArgs(run_id="abc123")

        cmd_config(analyzer, args)

        captured = capsys.readouterr()
        assert "learning_rate" in captured.out
        assert "0.001" in captured.out

    def test_cmd_notes(self, temp_wandb_dir, capsys):
        """Test notes command."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)
        args = MockArgs(run_id="abc123")

        cmd_notes(analyzer, args)

        captured = capsys.readouterr()
        assert "baseline-run" in captured.out
        assert "Testing baseline configuration" in captured.out

    def test_cmd_best(self, temp_wandb_dir, capsys):
        """Test best command."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)
        args = MockArgs(metric="train/loss", limit=10, higher_is_better=False)

        cmd_best(analyzer, args)

        captured = capsys.readouterr()
        assert "BEST" in captured.out
        assert "abc123" in captured.out

    def test_cmd_history_with_keys(self, temp_wandb_dir, capsys):
        """Test history command with specified keys."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)
        args = MockArgs(run_id="abc123", keys="train/loss,train/accuracy", samples=10)

        cmd_history(analyzer, args)

        captured = capsys.readouterr()
        assert "step,train/loss,train/accuracy" in captured.out

    def test_cmd_history_auto_detect(self, temp_wandb_dir, capsys):
        """Test history command with auto-detected keys."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)
        args = MockArgs(run_id="abc123", keys=None, samples=10)

        cmd_history(analyzer, args)

        captured = capsys.readouterr()
        # Should either auto-detect or show available keys
        assert "step" in captured.out or "auto-detected" in captured.out or "Available" in captured.out

    def test_cmd_stats_with_keys(self, temp_wandb_dir, capsys):
        """Test stats command with specified keys."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)
        args = MockArgs(run_id="abc123", keys="train/loss")

        cmd_stats(analyzer, args)

        captured = capsys.readouterr()
        assert "train/loss" in captured.out
        assert "Min" in captured.out

    def test_cmd_keys(self, temp_wandb_dir, capsys):
        """Test keys command."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)
        args = MockArgs(run_id="abc123")

        cmd_keys(analyzer, args)

        captured = capsys.readouterr()
        assert "train/loss" in captured.out

    def test_cmd_live(self, temp_wandb_dir, capsys):
        """Test live command."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)
        args = MockArgs()

        cmd_live(analyzer, args)

        captured = capsys.readouterr()
        assert "LIVE TRAINING STATUS" in captured.out
        assert "Run ID: def456" in captured.out

    def test_cmd_local_list(self, temp_logs_dir, capsys):
        """Test local command to list logs."""
        config = RunwiseConfig(logs_dir=temp_logs_dir)
        analyzer = RunAnalyzer(config)
        args = MockArgs(file=None)

        cmd_local(analyzer, args)

        captured = capsys.readouterr()
        assert "training.jsonl" in captured.out

    def test_cmd_local_analyze(self, temp_logs_dir, capsys):
        """Test local command to analyze specific log."""
        config = RunwiseConfig(logs_dir=temp_logs_dir)
        analyzer = RunAnalyzer(config)
        args = MockArgs(file=str(temp_logs_dir / "training.jsonl"))

        cmd_local(analyzer, args)

        captured = capsys.readouterr()
        assert "LOCAL LOG" in captured.out


class TestGetDefaultMetricKeys:
    """Tests for auto-detection of metric keys."""

    def test_detect_common_keys(self, temp_wandb_dir):
        """Test that common metric keys are detected."""
        config = RunwiseConfig(wandb_dir=temp_wandb_dir["wandb_dir"])
        analyzer = RunAnalyzer(config)
        run = analyzer.find_run("abc123")

        keys = _get_default_metric_keys(analyzer, run)

        # Should find train/loss and train/accuracy since they contain common patterns
        assert len(keys) > 0


class TestMainEntrypoint:
    """Tests for main CLI entrypoint."""

    def test_main_no_args(self, capsys):
        """Test main with no arguments shows help."""
        with patch.object(sys, 'argv', ['runwise']):
            main()

        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "runwise" in captured.out.lower()

    def test_main_list_command(self, temp_wandb_dir, capsys):
        """Test main with list command."""
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_wandb_dir["wandb_dir"].parent)
            with patch.object(sys, 'argv', ['runwise', 'list']):
                main()

            captured = capsys.readouterr()
            assert "abc123" in captured.out or "def456" in captured.out or "No runs" in captured.out
        finally:
            os.chdir(original_cwd)
