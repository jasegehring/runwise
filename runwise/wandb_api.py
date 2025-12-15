"""
Optional W&B API support for Runwise.

This module provides W&B cloud API access for teams without local synced
directories. It requires the `wandb` package to be installed.

Usage:
    from runwise.wandb_api import WandbAPIClient, WANDB_API_AVAILABLE

    if WANDB_API_AVAILABLE:
        client = WandbAPIClient("entity/project")
        runs = client.list_runs()
    else:
        print("W&B API support requires: pip install wandb")
"""

from dataclasses import dataclass
from typing import Optional

# Try to import wandb - make it optional
WANDB_API_AVAILABLE = False
_wandb_error_message = ""

try:
    import wandb
    from wandb.apis.public import Api, Run

    WANDB_API_AVAILABLE = True
except ImportError as e:
    _wandb_error_message = str(e)
    wandb = None  # type: ignore
    Api = None  # type: ignore
    Run = None  # type: ignore


@dataclass
class APIRunInfo:
    """Information about a W&B run from the API."""

    run_id: str
    name: str
    state: str  # running, finished, crashed, failed
    created_at: str
    runtime: float  # seconds
    steps: int
    summary: dict
    config: dict
    tags: list[str]
    notes: str
    group: str
    url: str


class WandbAPIClient:
    """
    Client for accessing W&B runs via the cloud API.

    Provides similar interface to RunAnalyzer but fetches from W&B servers
    instead of local files.
    """

    def __init__(self, project: str, entity: Optional[str] = None):
        """
        Initialize W&B API client.

        Args:
            project: W&B project name
            entity: W&B entity (username or team). If None, uses default entity.

        Raises:
            ImportError: If wandb package is not installed
        """
        if not WANDB_API_AVAILABLE:
            raise ImportError(
                "W&B API support requires the wandb package. "
                "Install with: pip install wandb\n"
                f"Original error: {_wandb_error_message}"
            )

        self.api = Api()
        self.project = project
        self.entity = entity or self.api.default_entity
        self.project_path = f"{self.entity}/{project}"
        self._run_cache: dict[str, APIRunInfo] = {}

    def list_runs(
        self,
        limit: int = 20,
        filters: Optional[dict] = None,
        order: str = "-created_at",
    ) -> list[APIRunInfo]:
        """
        List runs from W&B project.

        Args:
            limit: Maximum number of runs to return
            filters: W&B filter dict (e.g., {"state": "running"})
            order: Sort order (default: newest first)

        Returns:
            List of APIRunInfo objects
        """
        runs = self.api.runs(
            path=self.project_path,
            filters=filters,
            order=order,
            per_page=limit,
        )

        result = []
        for run in runs:
            info = self._parse_run(run)
            if info:
                result.append(info)
                self._run_cache[info.run_id] = info
            if len(result) >= limit:
                break

        return result

    def get_run(self, run_id: str) -> Optional[APIRunInfo]:
        """Get a specific run by ID."""
        # Check cache first
        if run_id in self._run_cache:
            return self._run_cache[run_id]

        try:
            run = self.api.run(f"{self.project_path}/{run_id}")
            info = self._parse_run(run)
            if info:
                self._run_cache[run_id] = info
            return info
        except Exception:
            return None

    def get_latest_run(self) -> Optional[APIRunInfo]:
        """Get the most recent run."""
        runs = self.list_runs(limit=1)
        return runs[0] if runs else None

    def _parse_run(self, run: "Run") -> Optional[APIRunInfo]:
        """Parse W&B Run object into APIRunInfo."""
        try:
            return APIRunInfo(
                run_id=run.id,
                name=run.name or "",
                state=run.state,
                created_at=run.created_at,
                runtime=run.summary.get("_runtime", 0) if run.summary else 0,
                steps=run.summary.get("_step", 0) if run.summary else 0,
                summary=dict(run.summary) if run.summary else {},
                config=dict(run.config) if run.config else {},
                tags=list(run.tags) if run.tags else [],
                notes=run.notes or "",
                group=run.group or "",
                url=run.url,
            )
        except Exception:
            return None

    def get_history(
        self,
        run_id: str,
        keys: list[str],
        samples: int = 500,
    ) -> list[dict]:
        """
        Get run history (metrics over time).

        Args:
            run_id: Run ID
            keys: Metric keys to fetch
            samples: Number of samples (W&B handles downsampling)

        Returns:
            List of metric dictionaries
        """
        try:
            run = self.api.run(f"{self.project_path}/{run_id}")

            # W&B history() returns a dataframe-like object
            # Using samples parameter for server-side downsampling
            history = run.history(keys=["_step"] + keys, samples=samples)

            # Convert to list of dicts
            records = []
            for _, row in history.iterrows():
                record = {"_step": row.get("_step", 0)}
                for key in keys:
                    if key in row and row[key] is not None:
                        record[key] = row[key]
                records.append(record)

            return records
        except Exception:
            return []

    def summarize_run(
        self,
        run: APIRunInfo,
        include_sparklines: bool = True,
        keys: Optional[list[str]] = None,
    ) -> str:
        """
        Generate summary for a W&B API run.

        Args:
            run: APIRunInfo object
            include_sparklines: Include sparkline visualizations
            keys: Optional list of specific metric keys to show. If provided,
                  only these metrics will be displayed.
        """
        from .sparklines import sparkline

        lines = ["=== W&B Run Summary (API) ==="]
        lines.append(f"Run: {run.run_id} | Name: {run.name}")
        lines.append(f"State: {run.state} | Steps: {run.steps:,} | Runtime: {run.runtime / 3600:.1f}h")
        lines.append(f"URL: {run.url}")

        if run.notes:
            notes_display = run.notes[:200] + "..." if len(run.notes) > 200 else run.notes
            lines.append(f"Notes: {notes_display}")

        if run.tags:
            lines.append(f"Tags: {', '.join(run.tags)}")

        lines.append("")

        # Get sparkline data if needed
        sparklines_data = {}
        if include_sparklines and run.summary:
            metrics_for_spark = keys if keys else list(run.summary.keys())[:10]
            history = self.get_history(run.run_id, metrics_for_spark, samples=20)
            if history:
                for key in metrics_for_spark:
                    values = [r.get(key) for r in history if key in r]
                    if values:
                        sparklines_data[key] = sparkline(values, width=8)

        # If custom keys specified, show only those metrics
        if keys:
            lines.append("METRICS:")
            max_key_len = min(40, max(len(k) for k in keys))
            found_any = False
            for key in keys:
                if key in run.summary:
                    found_any = True
                    value = run.summary[key]
                    if isinstance(value, float):
                        if value <= 1 and value >= 0:
                            value_str = f"{value*100:.2f}%"
                        elif abs(value) < 0.001 or abs(value) > 10000:
                            value_str = f"{value:.2e}"
                        else:
                            value_str = f"{value:.4f}"
                    else:
                        value_str = str(value)

                    spark = sparklines_data.get(key, "")
                    if spark:
                        lines.append(f"  {key:<{max_key_len}}: {value_str:>12}  {spark}")
                    else:
                        lines.append(f"  {key:<{max_key_len}}: {value_str:>12}")
                else:
                    lines.append(f"  {key:<{max_key_len}}: (not found)")

            if not found_any:
                lines.append("")
                lines.append("Tip: Check available metrics with 'runwise api -p <project> -r <run_id>'")
            return "\n".join(lines)

        # Otherwise show all metrics (default behavior)
        if run.summary:
            lines.append("METRICS:")
            # Filter out internal metrics and show top metrics
            metrics = {k: v for k, v in run.summary.items()
                       if not k.startswith("_") and isinstance(v, (int, float))}

            # Calculate dynamic width
            max_key_len = min(40, max(len(k) for k in metrics.keys())) if metrics else 30

            for key, value in sorted(metrics.items())[:15]:
                if isinstance(value, float):
                    if abs(value) < 0.001 or abs(value) > 1000:
                        value_str = f"{value:.2e}"
                    elif value <= 1:
                        value_str = f"{value*100:.1f}%"
                    else:
                        value_str = f"{value:.4f}"
                else:
                    value_str = str(value)

                spark = sparklines_data.get(key, "")
                if spark:
                    lines.append(f"  {key:<{max_key_len}}: {value_str:>12}  {spark}")
                else:
                    lines.append(f"  {key:<{max_key_len}}: {value_str:>12}")

        # Run anomaly detection
        if run.summary:
            history = self.get_history(run.run_id, ["loss", "train/loss", "val_loss"], samples=100)
            if history:
                from .anomalies import detect_anomalies, format_anomalies

                anomalies = detect_anomalies(history)
                if anomalies:
                    lines.append("")
                    lines.append(format_anomalies(anomalies, compact=True))

        return "\n".join(lines)

    def format_run_list(self, runs: list[APIRunInfo]) -> str:
        """Format runs as a table."""
        lines = ["W&B RUNS (API):", ""]
        lines.append(f"{'ID':<12} {'Name':<20} {'State':<10} {'Steps':>10} {'Runtime':>10}")
        lines.append("-" * 70)

        for run in runs:
            runtime_str = f"{run.runtime / 3600:.1f}h" if run.runtime > 3600 else f"{run.runtime / 60:.1f}m"
            name_display = run.name[:20] if run.name else "(unnamed)"
            lines.append(
                f"{run.run_id:<12} "
                f"{name_display:<20} "
                f"{run.state:<10} "
                f"{run.steps:>10,} "
                f"{runtime_str:>10}"
            )

            # Show tags on second line if available
            if run.tags:
                tags_str = ", ".join(run.tags[:3])
                if len(run.tags) > 3:
                    tags_str += f" (+{len(run.tags) - 3})"
                lines.append(f"  └─ [{tags_str}]")

        return "\n".join(lines)

    def compare_runs(
        self,
        run_a: APIRunInfo,
        run_b: APIRunInfo,
        filter_prefix: Optional[str] = None,
        threshold: Optional[float] = None,
        group_by_prefix: bool = False,
        show_config_diff: bool = False,
    ) -> str:
        """
        Compare two runs side-by-side.

        Args:
            run_a: First run to compare
            run_b: Second run to compare
            filter_prefix: Only show metrics starting with this prefix (e.g., 'val', 'train')
            threshold: Only show metrics with delta > threshold % (e.g., 5 for 5%)
            group_by_prefix: Group metrics by their prefix (e.g., train/, val/)
            show_config_diff: Include config differences at the end
        """
        lines = [f"COMPARISON (API): {run_a.run_id} vs {run_b.run_id}", ""]

        if run_a.name or run_b.name:
            lines.append(f"  A: {run_a.name or '(unnamed)'}")
            lines.append(f"  B: {run_b.name or '(unnamed)'}")
            lines.append("")

        # Show step counts if different (important for fair comparison)
        if run_a.steps != run_b.steps:
            lines.append(f"  NOTE: Runs at different steps (A: {run_a.steps:,}, B: {run_b.steps:,})")
            lines.append("")

        # Collect all metrics from both runs
        all_keys = set(run_a.summary.keys()) | set(run_b.summary.keys())

        # Filter and collect metrics
        metrics_to_show = []
        for key in all_keys:
            val_a = run_a.summary.get(key)
            val_b = run_b.summary.get(key)

            if not isinstance(val_a, (int, float)) or not isinstance(val_b, (int, float)):
                continue

            if key.startswith("_"):
                continue

            # Apply filter if specified
            if filter_prefix:
                if not key.lower().startswith(filter_prefix.lower()):
                    continue

            # Calculate delta percentage
            if val_a != 0:
                delta_pct = abs((val_b - val_a) / val_a * 100)
            else:
                delta_pct = 100 if val_b != 0 else 0

            # Apply threshold filter if specified
            if threshold is not None and delta_pct < threshold:
                continue

            metrics_to_show.append((key, val_a, val_b, delta_pct))

        if not metrics_to_show:
            if filter_prefix:
                lines.append(f"(no metrics matching filter '{filter_prefix}')")
            elif threshold:
                lines.append(f"(no metrics with >{threshold}% change)")
            else:
                lines.append("(no comparable metrics found)")
            return "\n".join(lines)

        # Calculate dynamic column width based on actual key lengths
        max_key_len = min(45, max(len(k) for k, _, _, _ in metrics_to_show))
        max_key_len = max(max_key_len, 6)  # Minimum width for "Metric" header

        # Group by prefix if requested
        if group_by_prefix:
            grouped: dict[str, list] = {}
            for key, val_a, val_b, delta_pct in metrics_to_show:
                prefix = key.split("/")[0] if "/" in key else "(ungrouped)"
                if prefix not in grouped:
                    grouped[prefix] = []
                grouped[prefix].append((key, val_a, val_b, delta_pct))

            for prefix in sorted(grouped.keys()):
                lines.append(f"\n{prefix.upper()}:")
                lines.append(f"{'Metric':<{max_key_len}} {'Run A':>12} {'Run B':>12} {'Delta':>10}")
                lines.append("-" * (max_key_len + 40))
                for key, val_a, val_b, delta_pct in sorted(grouped[prefix]):
                    self._append_comparison_line(lines, key, val_a, val_b, max_key_len)
        else:
            lines.append(f"{'Metric':<{max_key_len}} {'Run A':>12} {'Run B':>12} {'Delta':>10}")
            lines.append("-" * (max_key_len + 40))
            for key, val_a, val_b, delta_pct in sorted(metrics_to_show):
                self._append_comparison_line(lines, key, val_a, val_b, max_key_len)

        # Show summary of what was filtered
        if threshold:
            lines.append("")
            lines.append(f"(showing {len(metrics_to_show)} metrics with >{threshold}% change)")

        # Show config diff if requested
        if show_config_diff:
            config_diff = self._get_config_diff(run_a.config, run_b.config)
            if config_diff:
                lines.append("")
                lines.append("CONFIG DIFFERENCES:")
                max_config_key = min(40, max(len(k) for k in config_diff.keys()))
                lines.append(f"{'Parameter':<{max_config_key}} {'Run A':>20} {'Run B':>20}")
                lines.append("-" * (max_config_key + 45))
                for key, (c_val_a, c_val_b) in config_diff.items():
                    display_key = key[:max_config_key] if len(key) > max_config_key else key
                    str_a = str(c_val_a)[:20] if c_val_a is not None else "(not set)"
                    str_b = str(c_val_b)[:20] if c_val_b is not None else "(not set)"
                    lines.append(f"{display_key:<{max_config_key}} {str_a:>20} {str_b:>20}")
            else:
                lines.append("")
                lines.append("CONFIG DIFFERENCES: (none - configs are identical)")

        return "\n".join(lines)

    def _append_comparison_line(
        self,
        lines: list[str],
        key: str,
        val_a: float,
        val_b: float,
        key_width: int
    ) -> None:
        """Append a formatted comparison line to the output."""
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

        display_key = key[:key_width] if len(key) > key_width else key
        lines.append(f"{display_key:<{key_width}} {str_a:>12} {str_b:>12} {delta_str:>10}")

    def _get_config_diff(self, config_a: dict, config_b: dict) -> dict:
        """Get config parameters that differ between two runs."""
        diff = {}
        all_keys = set(config_a.keys()) | set(config_b.keys())

        for key in sorted(all_keys):
            if key.startswith("_"):
                continue
            val_a = config_a.get(key)
            val_b = config_b.get(key)
            if val_a != val_b:
                diff[key] = (val_a, val_b)

        return diff


def check_wandb_api_available() -> tuple[bool, str]:
    """
    Check if W&B API is available.

    Returns:
        Tuple of (is_available, message)
    """
    if WANDB_API_AVAILABLE:
        return True, "W&B API support is available"
    else:
        return False, (
            "W&B API support requires the wandb package.\n"
            "Install with: pip install wandb"
        )
