"""
Core analysis logic for Runwise.

Provides the main RunAnalyzer class that handles:
- W&B run discovery and parsing
- Local log file parsing
- Token-efficient summary generation
- Downsampled history retrieval (for large files)
- Sparkline trend visualization
- Anomaly detection
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .anomalies import detect_anomalies, format_anomalies
from .config import RunwiseConfig
from .sparklines import sparkline, trend_indicator


@dataclass
class RunInfo:
    """Basic information about a training run."""
    run_id: str
    directory: Path
    date: str
    time: str
    final_step: int = 0
    metrics: dict = None
    config: dict = None
    state: str = "unknown"  # running, finished, crashed, unknown
    # User-provided context about the run
    name: str = ""  # Display name (e.g., "lr-sweep-0.001")
    notes: str = ""  # Natural language description of what's being tested
    tags: list = None  # Tags like ["sweep", "baseline", "ablation"]
    group: str = ""  # Group name for related runs (e.g., sweep name)

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.config is None:
            self.config = {}
        if self.tags is None:
            self.tags = []


class RunAnalyzer:
    """
    Main analyzer for ML training runs.

    Supports W&B runs and local log files, generating token-efficient
    summaries suitable for LLM analysis.
    """

    def __init__(self, config: Optional[RunwiseConfig] = None):
        """
        Initialize analyzer with configuration.

        Args:
            config: RunwiseConfig instance. If None, auto-detects from cwd.
        """
        self.config = config or RunwiseConfig.auto_detect()

    # ==================== Run Discovery ====================

    def list_runs(self, limit: int = 20) -> list[RunInfo]:
        """List recent W&B runs."""
        runs = []

        if not self.config.wandb_dir.exists():
            return runs

        for d in sorted(self.config.wandb_dir.iterdir(), reverse=True):
            if d.is_dir() and d.name.startswith("run-"):
                run_info = self._parse_run_dir(d)
                if run_info:
                    runs.append(run_info)
                    if len(runs) >= limit:
                        break

        return runs

    def get_latest_run(self) -> Optional[RunInfo]:
        """Get the latest/active run."""
        latest_link = self.config.wandb_dir / "latest-run"
        if latest_link.exists():
            target = latest_link.resolve()
            if target.is_dir():
                return self._parse_run_dir(target)
        return None

    def find_run(self, run_id: str) -> Optional[RunInfo]:
        """Find a specific run by ID."""
        for d in self.config.wandb_dir.iterdir():
            if d.is_dir() and (run_id in d.name):
                return self._parse_run_dir(d)
        return None

    def _parse_run_dir(self, directory: Path) -> Optional[RunInfo]:
        """Parse a W&B run directory."""
        # Format: run-20251212_191603-iustpqgf
        match = re.match(r'run-(\d{8})_(\d{6})-(\w+)', directory.name)
        if not match:
            return None

        date_str, time_str, wandb_id = match.groups()

        run_info = RunInfo(
            run_id=wandb_id,
            directory=directory,
            date=f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}",
            time=f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}",
        )

        # Load summary if available
        summary_file = directory / "files" / "wandb-summary.json"
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    summary = json.load(f)
                run_info.metrics = summary
                run_info.final_step = summary.get(
                    self.config.schema.step_key,
                    summary.get("_step", 0)
                )
            except Exception:
                pass

        # Load config if available
        config_file = directory / "files" / "config.yaml"
        if config_file.exists():
            run_info.config = self._parse_wandb_config(config_file)

        # Load metadata for run context (name, notes, tags, group)
        metadata_file = directory / "files" / "wandb-metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                # Extract user-provided run context
                run_info.name = metadata.get("displayName", "")
                run_info.notes = metadata.get("notes", "")
                run_info.tags = metadata.get("tags", [])
                run_info.group = metadata.get("group", "")
                # Extract args as fallback config
                if not run_info.config and "args" in metadata:
                    run_info.config["_args"] = metadata["args"]
            except Exception:
                pass

        # Detect run state
        run_info.state = self._detect_run_state(directory)

        return run_info

    def _parse_wandb_config(self, config_file: Path) -> dict:
        """Parse W&B config.yaml file, filtering out internal keys."""
        config = {}
        try:
            # W&B config.yaml has a specific format with 'value' keys
            # Example: learning_rate:\n  value: 0.001
            content = config_file.read_text()

            # Simple YAML parsing for W&B config format
            current_key = None
            for line in content.split('\n'):
                line = line.rstrip()
                if not line or line.startswith('#'):
                    continue

                # Top-level key (no indent)
                if not line.startswith(' ') and line.endswith(':'):
                    current_key = line[:-1]
                    # Skip internal W&B keys
                    if current_key.startswith('_') or current_key.startswith('wandb'):
                        current_key = None
                # Value line (indented)
                elif current_key and 'value:' in line:
                    value_str = line.split('value:', 1)[1].strip()
                    # Parse value
                    if value_str == 'null' or value_str == 'None':
                        config[current_key] = None
                    elif value_str == 'true':
                        config[current_key] = True
                    elif value_str == 'false':
                        config[current_key] = False
                    elif value_str.startswith("'") or value_str.startswith('"'):
                        config[current_key] = value_str[1:-1]
                    else:
                        try:
                            # Try int first, then float
                            if '.' in value_str or 'e' in value_str.lower():
                                config[current_key] = float(value_str)
                            else:
                                config[current_key] = int(value_str)
                        except ValueError:
                            config[current_key] = value_str
        except Exception:
            pass
        return config

    def _detect_run_state(self, directory: Path) -> str:
        """Detect if a run is running, finished, or crashed."""
        # Check for wandb-summary.json - indicates run logged something
        summary_file = directory / "files" / "wandb-summary.json"

        # Check for exit code or status files
        exit_file = directory / "files" / "wandb-metadata.json"
        if exit_file.exists():
            try:
                with open(exit_file) as f:
                    metadata = json.load(f)
                # Check for exit code
                if "exitcode" in metadata:
                    code = metadata["exitcode"]
                    if code == 0:
                        return "finished"
                    else:
                        return "crashed"
                # Check state field
                if "state" in metadata:
                    return metadata["state"].lower()
            except Exception:
                pass

        # Check if run is still active by looking at latest-run symlink
        latest_link = self.config.wandb_dir / "latest-run"
        if latest_link.exists():
            try:
                if latest_link.resolve() == directory.resolve():
                    # This is the latest run - check if process is likely still running
                    # by checking if files are being modified recently
                    history_file = directory / "files" / "wandb-history.jsonl"
                    if history_file.exists():
                        import time
                        mtime = history_file.stat().st_mtime
                        if time.time() - mtime < 300:  # Modified in last 5 minutes
                            return "running"
            except Exception:
                pass

        # If summary exists but no exit code, assume finished
        if summary_file.exists():
            return "finished"

        return "unknown"

    # ==================== Summary Generation ====================

    def summarize_run(
        self,
        run: RunInfo,
        include_anomalies: bool = True,
        include_sparklines: bool = True,
    ) -> str:
        """Generate a token-efficient summary of a run with anomaly detection."""
        lines = []
        summary = run.metrics
        schema = self.config.schema

        # Header
        step = run.final_step
        runtime = summary.get("_runtime", 0)
        runtime_hrs = runtime / 3600 if runtime else 0
        lines.append(f"=== {self.config.project_name} Run Summary ===")
        lines.append(f"Run: {run.run_id} | Step: {step:,} | Runtime: {runtime_hrs:.1f}h")

        # Get history for sparklines and anomaly detection
        history = []
        if include_anomalies or include_sparklines:
            try:
                history = self.get_history_data(run, samples=200)
            except Exception:
                pass

        # Anomaly detection (show first if there are issues)
        if include_anomalies and history:
            anomalies = detect_anomalies(
                history,
                loss_key=schema.loss_key,
                val_loss_key="val_loss",  # Common default
            )
            if anomalies:
                lines.append("")
                lines.append(format_anomalies(anomalies, compact=True))

        # Run context (name, notes, tags, group) - provides semantic context
        if run.name:
            lines.append(f"Name: {run.name}")
        if run.group:
            lines.append(f"Group: {run.group}")
        if run.tags:
            lines.append(f"Tags: {', '.join(run.tags)}")
        if run.notes:
            # Truncate long notes for token efficiency, preserve first ~200 chars
            notes_display = run.notes[:200] + "..." if len(run.notes) > 200 else run.notes
            # Replace newlines with spaces for compact display
            notes_display = notes_display.replace("\n", " ").strip()
            lines.append(f"Notes: {notes_display}")

        lines.append("")

        # Build sparkline lookup from history
        metric_sparklines = {}
        if include_sparklines and history:
            for key in set(k for record in history for k in record.keys()):
                values = [r.get(key) for r in history if key in r]
                if values and all(isinstance(v, (int, float)) for v in values if v is not None):
                    metric_sparklines[key] = sparkline(values, width=10)

        # Metric groups
        for group in schema.groups:
            lines.append(f"{group.display_name}:")
            for key, meta in group.metrics.items():
                if key in summary:
                    value = summary[key]
                    fmt = meta.get("format", ".4f")
                    display = meta.get("display", key)

                    # Handle percentage format
                    if fmt.endswith("%"):
                        formatted = f"{value * 100:{fmt[:-1]}}%"
                    else:
                        formatted = f"{value:{fmt}}"

                    # Add sparkline if available
                    spark = metric_sparklines.get(key, "")
                    if spark:
                        lines.append(f"  {display}: {formatted}  {spark}")
                    else:
                        lines.append(f"  {display}: {formatted}")
            lines.append("")

        # Per-step metrics (for iterative models)
        if schema.per_step_pattern and schema.num_steps > 0:
            lines.append("PER-STEP METRICS:")
            step_values = []
            for i in range(schema.num_steps):
                key = schema.per_step_pattern.format(i=i)
                if key in summary:
                    step_values.append((i, summary[key]))

            if step_values:
                for i, val in step_values:
                    marker = " (initial)" if i == 0 else " (final)" if i == len(step_values) - 1 else ""
                    lines.append(f"  Step {i}: {val:.4f}{marker}")

                # Calculate improvement
                if len(step_values) >= 2:
                    initial, final = step_values[0][1], step_values[-1][1]
                    if initial > 0:
                        improvement = (initial - final) / initial * 100
                        lines.append(f"  Improvement: {improvement:+.1f}%")

                        # Detect plateau
                        if len(step_values) > 2:
                            mid_vals = [v for _, v in step_values[1:-1]]
                            if mid_vals:
                                variance = sum((v - sum(mid_vals)/len(mid_vals))**2 for v in mid_vals) / len(mid_vals)
                                if variance < 0.0001:
                                    lines.append("  WARNING: Plateau detected in middle steps")
            lines.append("")

        # Validation metrics
        if schema.validation_sets:
            lines.append("VALIDATION:")
            for prefix, display_name in schema.validation_sets.items():
                # Try common patterns
                for acc_suffix in ["/accuracy", "/token_accuracy", "_accuracy", "_acc"]:
                    key = f"{prefix}{acc_suffix}"
                    if key in summary:
                        lines.append(f"  {display_name}: {summary[key]*100:.1f}%")
                        break
            lines.append("")

        return "\n".join(lines)

    def format_run_list(self, runs: list[RunInfo], show_sparklines: bool = True) -> str:
        """Format a list of runs as a compact table with optional sparklines."""
        schema = self.config.schema
        lines = [f"RECENT RUNS ({self.config.project_name}):", ""]

        # Header with trend column
        if show_sparklines:
            lines.append(f"{'ID':<12} {'State':<9} {'Date':<12} {'Steps':>8} {schema.primary_metric_name:>12} {'Trend':>12}")
            lines.append("-" * 72)
        else:
            lines.append(f"{'ID':<12} {'State':<9} {'Date':<12} {'Steps':>8} {schema.primary_metric_name:>12}")
            lines.append("-" * 60)

        for run in runs:
            primary_val = run.metrics.get(schema.primary_metric, 0)
            if isinstance(primary_val, float) and primary_val <= 1:
                primary_str = f"{primary_val*100:.1f}%"
            else:
                primary_str = f"{primary_val:.4f}"

            # Format state with visual indicator
            state_str = run.state.upper()[:8]

            # Get sparkline for primary metric
            trend_str = ""
            if show_sparklines:
                try:
                    history = self.get_history_data(
                        run,
                        keys=[schema.primary_metric, schema.loss_key],
                        samples=20
                    )
                    if history:
                        # Use loss for trend (decreasing is good)
                        loss_values = [r.get(schema.loss_key) for r in history if schema.loss_key in r]
                        if loss_values:
                            spark = sparkline(loss_values, width=8)
                            trend = trend_indicator(loss_values)
                            trend_str = f"{spark}{trend}"
                except Exception:
                    trend_str = ""

            if show_sparklines:
                lines.append(
                    f"{run.run_id:<12} "
                    f"{state_str:<9} "
                    f"{run.date:<12} "
                    f"{run.final_step:>8,} "
                    f"{primary_str:>12} "
                    f"{trend_str:>12}"
                )
            else:
                lines.append(
                    f"{run.run_id:<12} "
                    f"{state_str:<9} "
                    f"{run.date:<12} "
                    f"{run.final_step:>8,} "
                    f"{primary_str:>12}"
                )

            # Show run context on second line if available (token-efficient)
            context_parts = []
            if run.name:
                context_parts.append(run.name[:30])
            if run.tags:
                context_parts.append(f"[{', '.join(run.tags[:3])}]")
            if context_parts:
                lines.append(f"  └─ {' '.join(context_parts)}")

        return "\n".join(lines)

    def get_config(self, run: RunInfo) -> str:
        """Get hyperparameters/config for a run as formatted output."""
        if not run.config:
            return f"No config found for run {run.run_id}"

        lines = [f"CONFIG: {run.run_id}", ""]

        # Group config by category if possible
        for key in sorted(run.config.keys()):
            value = run.config[key]

            # Format value based on type
            if isinstance(value, float):
                if abs(value) < 0.001 or abs(value) > 10000:
                    value_str = f"{value:.2e}"
                else:
                    value_str = f"{value:.6g}"
            elif isinstance(value, bool):
                value_str = str(value).lower()
            elif isinstance(value, list):
                value_str = str(value)[:50]  # Truncate long lists
            else:
                value_str = str(value)

            lines.append(f"  {key}: {value_str}")

        return "\n".join(lines)

    def get_run_context(self, run: RunInfo) -> str:
        """
        Get full run context (name, notes, tags, group) for a run.

        This provides the complete user-provided description of what the run
        is testing, what variables are being swept, hypotheses, etc.
        """
        lines = [f"RUN CONTEXT: {run.run_id}", ""]

        if run.name:
            lines.append(f"Name: {run.name}")
        else:
            lines.append("Name: (not set)")

        if run.group:
            lines.append(f"Group: {run.group}")

        if run.tags:
            lines.append(f"Tags: {', '.join(run.tags)}")
        else:
            lines.append("Tags: (none)")

        lines.append("")

        if run.notes:
            lines.append("Notes:")
            # Preserve full notes with original formatting
            for line in run.notes.split("\n"):
                lines.append(f"  {line}")
        else:
            lines.append("Notes: (none)")
            lines.append("")
            lines.append("Tip: Set notes in W&B with:")
            lines.append("  wandb.init(notes='Testing lr=0.001 with new augmentation')")
            lines.append("  # or after init:")
            lines.append("  run.notes = 'Description of what this run tests'")

        return "\n".join(lines)

    def find_best_run(
        self,
        metric: str,
        limit: int = 10,
        higher_is_better: bool = False
    ) -> tuple[Optional[RunInfo], list[tuple[RunInfo, float]]]:
        """
        Find the best run by a given metric.

        Args:
            metric: Metric key to compare (e.g., 'val_loss', 'accuracy')
            limit: Number of recent runs to consider
            higher_is_better: If True, higher values are better (e.g., accuracy)

        Returns:
            Tuple of (best_run, ranked_list) where ranked_list is [(run, value), ...]
        """
        runs = self.list_runs(limit=limit)

        # Collect runs with the metric
        scored_runs = []
        for run in runs:
            value = run.metrics.get(metric)
            if value is not None and isinstance(value, (int, float)):
                scored_runs.append((run, value))

        if not scored_runs:
            return None, []

        # Sort by metric
        scored_runs.sort(key=lambda x: x[1], reverse=higher_is_better)

        return scored_runs[0][0], scored_runs

    def format_best_run(
        self,
        metric: str,
        limit: int = 10,
        higher_is_better: bool = False
    ) -> str:
        """Format best run comparison as a table."""
        best_run, ranked = self.find_best_run(metric, limit, higher_is_better)

        if not best_run:
            return f"No runs found with metric '{metric}'"

        lines = [f"BEST RUN BY {metric} (from last {limit} runs):", ""]

        # Highlight the winner
        best_value = ranked[0][1]
        if isinstance(best_value, float) and best_value <= 1:
            best_str = f"{best_value*100:.2f}%"
        else:
            best_str = f"{best_value:.6g}"

        lines.append(f"  BEST: {best_run.run_id} = {best_str}")
        lines.append("")

        # Show ranking
        lines.append(f"{'Rank':<6} {'Run ID':<12} {'State':<9} {metric:>15}")
        lines.append("-" * 45)

        for i, (run, value) in enumerate(ranked[:10], 1):
            if isinstance(value, float) and value <= 1:
                value_str = f"{value*100:.2f}%"
            else:
                value_str = f"{value:.6g}"

            marker = " *" if i == 1 else ""
            lines.append(f"{i:<6} {run.run_id:<12} {run.state:<9} {value_str:>15}{marker}")

        return "\n".join(lines)

    def compare_runs(
        self,
        run_a: RunInfo,
        run_b: RunInfo,
        filter_prefix: str = None,
        show_config_diff: bool = False
    ) -> str:
        """
        Compare two runs side-by-side.

        Args:
            run_a: First run to compare
            run_b: Second run to compare
            filter_prefix: Only show metrics starting with this prefix (e.g., 'val', 'train')
            show_config_diff: Include config differences at the end
        """
        lines = [f"COMPARISON: {run_a.run_id} vs {run_b.run_id}", ""]

        # Show run names if available
        if run_a.name or run_b.name:
            lines.append(f"  A: {run_a.name or '(unnamed)'}")
            lines.append(f"  B: {run_b.name or '(unnamed)'}")
            lines.append("")

        lines.append(f"{'Metric':<25} {'Run A':>12} {'Run B':>12} {'Delta':>10}")
        lines.append("-" * 65)

        # Collect all metrics from both runs
        all_keys = set(run_a.metrics.keys()) | set(run_b.metrics.keys())

        # Filter to numeric metrics
        matched_count = 0
        for key in sorted(all_keys):
            val_a = run_a.metrics.get(key)
            val_b = run_b.metrics.get(key)

            if not isinstance(val_a, (int, float)) or not isinstance(val_b, (int, float)):
                continue

            # Skip internal metrics
            if key.startswith("_"):
                continue

            # Apply filter if specified
            if filter_prefix:
                if not key.lower().startswith(filter_prefix.lower()):
                    continue

            matched_count += 1

            # Format values
            if isinstance(val_a, float) and val_a <= 1 and isinstance(val_b, float) and val_b <= 1:
                str_a = f"{val_a*100:.1f}%"
                str_b = f"{val_b*100:.1f}%"
                delta = (val_b - val_a) * 100
                delta_str = f"{delta:+.1f}%"
            else:
                str_a = f"{val_a:.4f}"
                str_b = f"{val_b:.4f}"
                delta = val_b - val_a
                delta_str = f"{delta:+.4f}"

            # Truncate key for display
            display_key = key[:25] if len(key) > 25 else key
            lines.append(f"{display_key:<25} {str_a:>12} {str_b:>12} {delta_str:>10}")

        if filter_prefix and matched_count == 0:
            lines.append(f"(no metrics matching filter '{filter_prefix}')")

        # Show config diff if requested
        if show_config_diff:
            config_diff = self._get_config_diff(run_a.config, run_b.config)
            if config_diff:
                lines.append("")
                lines.append("CONFIG DIFFERENCES:")
                lines.append(f"{'Parameter':<25} {'Run A':>15} {'Run B':>15}")
                lines.append("-" * 60)
                for key, (val_a, val_b) in config_diff.items():
                    display_key = key[:25] if len(key) > 25 else key
                    str_a = str(val_a)[:15] if val_a is not None else "(not set)"
                    str_b = str(val_b)[:15] if val_b is not None else "(not set)"
                    lines.append(f"{display_key:<25} {str_a:>15} {str_b:>15}")
            else:
                lines.append("")
                lines.append("CONFIG DIFFERENCES: (none - configs are identical)")

        return "\n".join(lines)

    def _get_config_diff(self, config_a: dict, config_b: dict) -> dict:
        """Get config parameters that differ between two runs."""
        diff = {}
        all_keys = set(config_a.keys()) | set(config_b.keys())

        for key in sorted(all_keys):
            # Skip internal keys
            if key.startswith("_"):
                continue

            val_a = config_a.get(key)
            val_b = config_b.get(key)

            if val_a != val_b:
                diff[key] = (val_a, val_b)

        return diff

    # ==================== History (Downsampled) ====================

    def get_history_data(
        self,
        run: RunInfo,
        keys: Optional[list[str]] = None,
        samples: int = 100,
    ) -> list[dict]:
        """
        Get downsampled history as list of dicts (for internal use).

        This is used by sparklines and anomaly detection. For CLI/MCP,
        use get_history() which returns CSV.

        Args:
            run: RunInfo object for the run
            keys: List of metric keys to fetch (None = all keys)
            samples: Number of data points to return

        Returns:
            List of metric dictionaries
        """
        history_file = run.directory / "files" / "wandb-history.jsonl"
        if not history_file.exists():
            return []

        # First pass: count lines
        total_lines = 0
        with open(history_file, 'r') as f:
            for _ in f:
                total_lines += 1

        if total_lines == 0:
            return []

        # Calculate which lines to sample
        if total_lines <= samples:
            sample_indices = set(range(total_lines))
        else:
            sample_indices = set()
            for i in range(samples):
                idx = int(i * (total_lines - 1) / (samples - 1))
                sample_indices.add(idx)

        # Second pass: read only sampled lines
        records = []
        with open(history_file, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num not in sample_indices:
                    continue
                try:
                    record = json.loads(line.strip())
                    if keys:
                        # Filter to requested keys plus step
                        filtered = {"_step": record.get("_step", record.get("step", line_num))}
                        for key in keys:
                            if key in record:
                                filtered[key] = record[key]
                        records.append(filtered)
                    else:
                        records.append(record)
                except json.JSONDecodeError:
                    continue

        return records

    def get_history(
        self,
        run: RunInfo,
        keys: list[str],
        samples: int = 500
    ) -> str:
        """
        Get downsampled training history as CSV.

        This is the key token-efficiency feature: a 1M step run returns
        exactly `samples` rows, keeping output under ~3000 tokens.

        Args:
            run: RunInfo object for the run
            keys: List of metric keys to fetch (e.g., ["loss", "val_loss"])
            samples: Number of data points to return (default 500)

        Returns:
            CSV-formatted string with step and requested metrics
        """
        # Try W&B history file first
        history_file = run.directory / "files" / "wandb-history.jsonl"
        if not history_file.exists():
            # Fall back to output.log parsing or return error
            return self._get_history_from_output_log(run, keys, samples)

        return self._downsample_jsonl(history_file, keys, samples)

    def _downsample_jsonl(
        self,
        file_path: Path,
        keys: list[str],
        samples: int
    ) -> str:
        """
        Downsample a JSONL file efficiently without loading it all into memory.

        Uses two-pass approach:
        1. Count total lines (fast, no parsing)
        2. Read only the lines we need based on calculated interval
        """
        # First pass: count lines
        total_lines = 0
        with open(file_path, 'r') as f:
            for _ in f:
                total_lines += 1

        if total_lines == 0:
            return "step," + ",".join(keys) + "\n(no data)"

        # Calculate which lines to sample
        if total_lines <= samples:
            # File is small enough, read all
            sample_indices = set(range(total_lines))
        else:
            # Evenly spaced samples including first and last
            sample_indices = set()
            for i in range(samples):
                idx = int(i * (total_lines - 1) / (samples - 1))
                sample_indices.add(idx)

        # Second pass: read only sampled lines
        rows = []
        available_keys = set()

        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num not in sample_indices:
                    continue

                try:
                    record = json.loads(line.strip())
                    available_keys.update(record.keys())

                    step = record.get("_step", record.get("step", line_num))
                    row = [str(step)]

                    for key in keys:
                        val = record.get(key, "")
                        if isinstance(val, float):
                            row.append(f"{val:.6g}")
                        else:
                            row.append(str(val) if val != "" else "")

                    rows.append(",".join(row))
                except json.JSONDecodeError:
                    continue

        if not rows:
            # No data found for requested keys, list available
            filtered_keys = [k for k in available_keys if not k.startswith("_")]
            return f"No data for keys: {keys}\nAvailable keys: {sorted(filtered_keys)[:20]}"

        # Build CSV output
        header = "step," + ",".join(keys)
        return header + "\n" + "\n".join(rows)

    def _get_history_from_output_log(
        self,
        run: RunInfo,
        keys: list[str],
        samples: int
    ) -> str:
        """
        Extract history from output.log when wandb-history.jsonl isn't available.

        Parses common logging patterns like:
        - "Step 1000 | loss: 0.5 | accuracy: 0.8"
        - "{'step': 1000, 'loss': 0.5}"
        """
        output_file = run.directory / "files" / "output.log"
        if not output_file.exists():
            return f"No history file found for run {run.run_id}"

        # First pass: find all lines with metrics
        metric_lines = []
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Check if line contains any of our keys
                if any(key in line.lower() for key in [k.lower() for k in keys]):
                    metric_lines.append(line)

        if not metric_lines:
            return f"No metrics matching {keys} found in output.log"

        # Downsample
        if len(metric_lines) > samples:
            indices = [int(i * (len(metric_lines) - 1) / (samples - 1)) for i in range(samples)]
            metric_lines = [metric_lines[i] for i in indices]

        # Parse and format as CSV
        rows = []
        for line in metric_lines:
            row_data = {"step": ""}

            # Try JSON parsing first
            try:
                data = json.loads(line)
                row_data["step"] = str(data.get("_step", data.get("step", "")))
                for key in keys:
                    if key in data:
                        val = data[key]
                        row_data[key] = f"{val:.6g}" if isinstance(val, float) else str(val)
            except json.JSONDecodeError:
                # Try regex patterns
                step_match = re.search(r'[Ss]tep[:\s]+(\d+)', line)
                if step_match:
                    row_data["step"] = step_match.group(1)

                for key in keys:
                    pattern = rf'{re.escape(key)}[:\s]+([\d.e+-]+)'
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        row_data[key] = match.group(1)

            if row_data["step"] or any(key in row_data for key in keys):
                row = [row_data.get("step", "")]
                for key in keys:
                    row.append(row_data.get(key, ""))
                rows.append(",".join(row))

        if not rows:
            return "Could not parse metrics from output.log"

        header = "step," + ",".join(keys)
        return header + "\n" + "\n".join(rows)

    def get_history_stats(
        self,
        run: RunInfo,
        keys: list[str]
    ) -> str:
        """
        Get statistical summary of history without returning raw data.

        Even more token-efficient than get_history - returns only:
        - min, max, mean, final value
        - NaN count (for divergence detection)
        - Trend direction
        """
        history_file = run.directory / "files" / "wandb-history.jsonl"
        if not history_file.exists():
            return f"No history file for run {run.run_id}"

        # Single pass through file collecting stats
        stats = {key: {"values": [], "nan_count": 0} for key in keys}
        total_steps = 0

        with open(history_file, 'r') as f:
            for line in f:
                total_steps += 1
                try:
                    record = json.loads(line.strip())
                    for key in keys:
                        if key in record:
                            val = record[key]
                            if val is None or (isinstance(val, float) and (val != val)):  # NaN check
                                stats[key]["nan_count"] += 1
                            elif isinstance(val, (int, float)):
                                stats[key]["values"].append(val)
                except json.JSONDecodeError:
                    continue

        # Format output
        lines = [f"HISTORY STATS: {run.run_id} ({total_steps:,} steps)", ""]
        lines.append(f"{'Metric':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Final':>10} {'NaN':>6}")
        lines.append("-" * 70)

        for key in keys:
            vals = stats[key]["values"]
            nan_count = stats[key]["nan_count"]

            if not vals:
                lines.append(f"{key:<20} {'--':>10} {'--':>10} {'--':>10} {'--':>10} {nan_count:>6}")
                continue

            min_v = min(vals)
            max_v = max(vals)
            mean_v = sum(vals) / len(vals)
            final_v = vals[-1]

            # Format based on magnitude
            def fmt(v):
                if abs(v) < 0.01 or abs(v) > 1000:
                    return f"{v:.2e}"
                return f"{v:.4f}"

            lines.append(f"{key:<20} {fmt(min_v):>10} {fmt(max_v):>10} {fmt(mean_v):>10} {fmt(final_v):>10} {nan_count:>6}")

        return "\n".join(lines)

    def list_available_keys(self, run: RunInfo) -> str:
        """
        List all available metric keys in a run's history.

        Useful when you don't know what metrics are logged.
        """
        history_file = run.directory / "files" / "wandb-history.jsonl"
        if not history_file.exists():
            # Try summary file
            if run.metrics:
                keys = [k for k in run.metrics.keys() if not k.startswith("_")]
                return "Available keys (from summary):\n" + "\n".join(sorted(keys))
            return f"No history or summary for run {run.run_id}"

        # Read first and last few lines to get representative keys
        keys = set()
        with open(history_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 5 or i % 1000 == 0:  # Sample first 5 and every 1000th
                    try:
                        record = json.loads(line.strip())
                        keys.update(k for k in record.keys() if not k.startswith("_"))
                    except json.JSONDecodeError:
                        continue
                if i > 10000:  # Don't scan entire huge file
                    break

        return "Available keys:\n" + "\n".join(sorted(keys))

    # ==================== Live Status ====================

    def get_live_status(self) -> str:
        """Get status of currently running training."""
        latest = self.get_latest_run()
        if not latest:
            return "No active run found"

        lines = ["LIVE TRAINING STATUS:", ""]
        lines.append(f"Run ID: {latest.run_id}")
        if latest.name:
            lines.append(f"Name: {latest.name}")
        lines.append(f"State: {latest.state}")
        lines.append("")

        output_file = latest.directory / "files" / "output.log"
        if not output_file.exists():
            return "\n".join(lines) + "\n(no output.log yet)"

        # Read last 100 lines
        with open(output_file) as f:
            recent_lines = f.readlines()[-100:]

        # Parse for latest metrics (generic patterns)
        latest_step = None
        latest_loss = None
        latest_acc = None

        for line in recent_lines:
            line = line.strip()

            # Try common patterns
            # Pattern 1: "Step X | Loss: Y | Accuracy: Z"
            match = re.search(r'[Ss]tep[:\s]+(\d+)', line)
            if match:
                latest_step = int(match.group(1))

            loss_match = re.search(r'[Ll]oss[:\s]+([\d.]+)', line)
            if loss_match:
                latest_loss = float(loss_match.group(1))

            acc_match = re.search(r'[Aa]cc(?:uracy)?[:\s]+([\d.]+)', line)
            if acc_match:
                latest_acc = float(acc_match.group(1))

        if latest_step is not None:
            lines.append(f"Current Step: {latest_step:,}")
        if latest_loss is not None:
            lines.append(f"Loss: {latest_loss:.4f}")
        if latest_acc is not None:
            if latest_acc <= 1:
                lines.append(f"Accuracy: {latest_acc*100:.1f}%")
            else:
                lines.append(f"Accuracy: {latest_acc:.1f}%")

        return "\n".join(lines)

    # ==================== Local Logs ====================

    def list_local_logs(self) -> list[Path]:
        """List available local log files."""
        if not self.config.logs_dir.exists():
            return []

        # Try common patterns
        patterns = ["*.jsonl", "metrics_*.json", "train_*.log"]
        logs = []
        for pattern in patterns:
            logs.extend(self.config.logs_dir.glob(pattern))

        return sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True)

    def parse_local_log(self, log_file: Path) -> list[dict]:
        """Parse a local JSONL log file."""
        records = []
        if not log_file.exists():
            return records

        with open(log_file) as f:
            for line in f:
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        return records

    def summarize_local_log(self, log_file: Path) -> str:
        """Generate summary from local log file."""
        records = self.parse_local_log(log_file)
        if not records:
            return f"No data in {log_file}"

        lines = [f"=== LOCAL LOG: {log_file.name} ===", ""]

        # Training curve (downsampled)
        schema = self.config.schema
        interval = self.config.downsample_interval

        lines.append("TRAINING CURVE:")
        lines.append(f"{'Step':>8} {'Loss':>8} {schema.primary_metric_name:>10}")
        lines.append("-" * 30)

        for r in records:
            step = r.get("step", r.get("_step", 0))
            if step % interval == 0:
                loss = r.get(schema.loss_key, r.get("loss", 0))
                primary = r.get(schema.primary_metric, r.get("accuracy", 0))

                if isinstance(primary, float) and primary <= 1:
                    primary_str = f"{primary*100:.1f}%"
                else:
                    primary_str = f"{primary:.4f}"

                lines.append(f"{step:>8} {loss:>8.4f} {primary_str:>10}")

        # Latest metrics
        if records:
            latest = records[-1]
            lines.append("")
            lines.append("LATEST METRICS:")
            for key, value in sorted(latest.items()):
                if isinstance(value, (int, float)) and not key.startswith("_"):
                    if isinstance(value, float):
                        lines.append(f"  {key}: {value:.4f}")
                    else:
                        lines.append(f"  {key}: {value}")

        return "\n".join(lines)
