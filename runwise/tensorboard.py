"""
Optional TensorBoard support for Runwise.

This module provides TensorBoard event file parsing. It requires the
`tensorboard` package to be installed. Users who work with TensorBoard
typically already have it installed.

Usage:
    from runwise.tensorboard import TensorBoardParser, TENSORBOARD_AVAILABLE

    if TENSORBOARD_AVAILABLE:
        parser = TensorBoardParser("/path/to/logs")
        runs = parser.list_runs()
    else:
        print("TensorBoard support requires: pip install tensorboard")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Try to import TensorBoard - make it optional
TENSORBOARD_AVAILABLE = False
_tb_error_message = ""

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError as e:
    _tb_error_message = str(e)
    EventAccumulator = None  # type: ignore


@dataclass
class TBRunInfo:
    """Information about a TensorBoard run."""

    run_id: str
    directory: Path
    tags: list[str]  # Available scalar tags
    steps: int
    wall_time: float  # Total wall time in seconds


class TensorBoardParser:
    """
    Parser for TensorBoard event files.

    Handles tfevents files and provides similar interface to RunAnalyzer.
    """

    def __init__(self, log_dir: Path | str):
        """
        Initialize TensorBoard parser.

        Args:
            log_dir: Path to TensorBoard log directory

        Raises:
            ImportError: If tensorboard package is not installed
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError(
                "TensorBoard support requires the tensorboard package. "
                "Install with: pip install tensorboard\n"
                f"Original error: {_tb_error_message}"
            )

        self.log_dir = Path(log_dir)
        self._accumulators: dict[str, EventAccumulator] = {}

    def list_runs(self) -> list[TBRunInfo]:
        """
        List available TensorBoard runs.

        Scans for directories containing tfevents files.
        """
        runs = []

        # Find all directories with tfevents files
        for event_file in self.log_dir.rglob("events.out.tfevents.*"):
            run_dir = event_file.parent

            # Create unique run_id from relative path
            try:
                rel_path = run_dir.relative_to(self.log_dir)
                run_id = str(rel_path).replace("/", "_") or "root"
            except ValueError:
                run_id = run_dir.name

            # Skip if already processed
            if any(r.run_id == run_id for r in runs):
                continue

            try:
                info = self._parse_run_dir(run_dir, run_id)
                if info:
                    runs.append(info)
            except Exception:
                continue

        return sorted(runs, key=lambda r: r.wall_time, reverse=True)

    def _parse_run_dir(self, directory: Path, run_id: str) -> Optional[TBRunInfo]:
        """Parse a TensorBoard run directory."""
        accumulator = self._get_accumulator(directory)
        if not accumulator:
            return None

        # Get available scalar tags
        tags = accumulator.Tags().get("scalars", [])

        # Get step count from any tag
        steps = 0
        wall_time = 0.0
        for tag in tags:
            events = accumulator.Scalars(tag)
            if events:
                steps = max(steps, max(e.step for e in events))
                wall_time = max(wall_time, events[-1].wall_time - events[0].wall_time)

        return TBRunInfo(
            run_id=run_id,
            directory=directory,
            tags=tags,
            steps=steps,
            wall_time=wall_time,
        )

    def _get_accumulator(self, directory: Path) -> Optional[EventAccumulator]:
        """Get or create EventAccumulator for a directory."""
        dir_str = str(directory)

        if dir_str not in self._accumulators:
            try:
                acc = EventAccumulator(dir_str)
                acc.Reload()
                self._accumulators[dir_str] = acc
            except Exception:
                return None

        return self._accumulators.get(dir_str)

    def get_scalars(
        self,
        run: TBRunInfo,
        tags: list[str],
        samples: int = 500,
    ) -> dict[str, list[tuple[int, float]]]:
        """
        Get scalar values for specified tags.

        Args:
            run: TBRunInfo for the run
            tags: List of scalar tag names
            samples: Max number of samples to return (downsampled if needed)

        Returns:
            Dict mapping tag name to list of (step, value) tuples
        """
        accumulator = self._get_accumulator(run.directory)
        if not accumulator:
            return {}

        result = {}
        for tag in tags:
            if tag not in run.tags:
                continue

            try:
                events = accumulator.Scalars(tag)
                if not events:
                    continue

                # Downsample if needed
                if len(events) > samples:
                    indices = [
                        int(i * (len(events) - 1) / (samples - 1))
                        for i in range(samples)
                    ]
                    events = [events[i] for i in indices]

                result[tag] = [(e.step, e.value) for e in events]
            except Exception:
                continue

        return result

    def get_history_data(
        self,
        run: TBRunInfo,
        tags: Optional[list[str]] = None,
        samples: int = 100,
    ) -> list[dict]:
        """
        Get history as list of dicts (compatible with anomaly detection).

        Args:
            run: TBRunInfo for the run
            tags: List of tags to include (None = all)
            samples: Number of samples

        Returns:
            List of dicts with step and metric values
        """
        if tags is None:
            tags = run.tags

        scalars = self.get_scalars(run, tags, samples)

        if not scalars:
            return []

        # Merge all scalars by step
        step_data: dict[int, dict] = {}
        for tag, values in scalars.items():
            for step, value in values:
                if step not in step_data:
                    step_data[step] = {"_step": step}
                step_data[step][tag] = value

        return [step_data[s] for s in sorted(step_data.keys())]

    def format_run_list(self, runs: list[TBRunInfo]) -> str:
        """Format runs as a table."""
        lines = ["TENSORBOARD RUNS:", ""]
        lines.append(f"{'Run ID':<30} {'Steps':>10} {'Runtime':>12} {'Tags':>8}")
        lines.append("-" * 65)

        for run in runs:
            runtime_str = f"{run.wall_time / 3600:.1f}h" if run.wall_time > 3600 else f"{run.wall_time / 60:.1f}m"
            lines.append(
                f"{run.run_id[:30]:<30} "
                f"{run.steps:>10,} "
                f"{runtime_str:>12} "
                f"{len(run.tags):>8}"
            )

        return "\n".join(lines)

    def summarize_run(
        self,
        run: TBRunInfo,
        loss_tag: str = "loss",
        include_sparklines: bool = True,
    ) -> str:
        """
        Generate summary for a TensorBoard run.

        Args:
            run: TBRunInfo for the run
            loss_tag: Tag name for loss metric
            include_sparklines: Include sparkline visualizations
        """
        from .sparklines import sparkline

        lines = ["=== TensorBoard Run Summary ==="]
        lines.append(f"Run: {run.run_id}")
        lines.append(f"Steps: {run.steps:,} | Runtime: {run.wall_time / 3600:.1f}h")
        lines.append("")

        # Get scalar data
        history = self.get_history_data(run, samples=100)

        if not history:
            lines.append("(no scalar data)")
            return "\n".join(lines)

        # Show metrics with sparklines
        lines.append("METRICS:")
        for tag in run.tags[:10]:  # Limit to first 10 tags
            values = [r.get(tag) for r in history if tag in r]
            if not values:
                continue

            # Get final value
            final = values[-1]
            if abs(final) < 0.001 or abs(final) > 1000:
                final_str = f"{final:.2e}"
            else:
                final_str = f"{final:.4f}"

            if include_sparklines:
                spark = sparkline(values, width=10)
                lines.append(f"  {tag[:25]:<25}: {final_str:>12}  {spark}")
            else:
                lines.append(f"  {tag[:25]:<25}: {final_str:>12}")

        # Run anomaly detection
        from .anomalies import detect_anomalies, format_anomalies

        anomalies = detect_anomalies(history, loss_key=loss_tag)
        if anomalies:
            lines.append("")
            lines.append(format_anomalies(anomalies, compact=True))

        return "\n".join(lines)


def check_tensorboard_available() -> tuple[bool, str]:
    """
    Check if TensorBoard is available.

    Returns:
        Tuple of (is_available, message)
    """
    if TENSORBOARD_AVAILABLE:
        return True, "TensorBoard support is available"
    else:
        return False, (
            "TensorBoard support requires the tensorboard package.\n"
            "Install with: pip install tensorboard"
        )
