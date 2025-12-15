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
    ) -> str:
        """
        Generate summary for a W&B API run.

        Args:
            run: APIRunInfo object
            include_sparklines: Include sparkline visualizations
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

        # Show key metrics from summary
        if run.summary:
            lines.append("METRICS:")
            # Filter out internal metrics and show top metrics
            metrics = {k: v for k, v in run.summary.items()
                       if not k.startswith("_") and isinstance(v, (int, float))}

            # Get history for sparklines if requested
            sparklines_data = {}
            if include_sparklines and metrics:
                history = self.get_history(run.run_id, list(metrics.keys())[:10], samples=20)
                if history:
                    for key in metrics:
                        values = [r.get(key) for r in history if key in r]
                        if values:
                            sparklines_data[key] = sparkline(values, width=8)

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
                    lines.append(f"  {key[:30]:<30}: {value_str:>12}  {spark}")
                else:
                    lines.append(f"  {key[:30]:<30}: {value_str:>12}")

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

    def compare_runs(self, run_a: APIRunInfo, run_b: APIRunInfo) -> str:
        """Compare two runs side-by-side."""
        lines = [f"COMPARISON (API): {run_a.run_id} vs {run_b.run_id}", ""]

        if run_a.name or run_b.name:
            lines.append(f"  A: {run_a.name or '(unnamed)'}")
            lines.append(f"  B: {run_b.name or '(unnamed)'}")
            lines.append("")

        lines.append(f"{'Metric':<25} {'Run A':>12} {'Run B':>12} {'Delta':>10}")
        lines.append("-" * 65)

        # Collect all metrics from both runs
        all_keys = set(run_a.summary.keys()) | set(run_b.summary.keys())

        for key in sorted(all_keys):
            val_a = run_a.summary.get(key)
            val_b = run_b.summary.get(key)

            if not isinstance(val_a, (int, float)) or not isinstance(val_b, (int, float)):
                continue

            if key.startswith("_"):
                continue

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

            display_key = key[:25]
            lines.append(f"{display_key:<25} {str_a:>12} {str_b:>12} {delta_str:>10}")

        return "\n".join(lines)


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
