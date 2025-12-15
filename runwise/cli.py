#!/usr/bin/env python3
"""
Command-line interface for Runwise - Token-efficient ML training run analysis.

IMPORTANT FOR AI AGENTS:
- This tool reads from LOCAL wandb/ directories, NOT from W&B cloud API
- For W&B cloud access without local files, use: runwise api -p <project>
- For local JSONL logs (not W&B), use: runwise local <file>

DATA SOURCES:
1. W&B Local (default): Reads wandb/run-*/files/ directories
2. W&B Cloud API: Use 'runwise api' (requires: pip install wandb)
3. Local JSONL: Use 'runwise local' for standalone log files
4. TensorBoard: Use 'runwise tb' (requires: pip install tensorboard)

COMMON WORKFLOWS:

  # W&B runs (reads from local wandb/ directory)
  runwise list                        # List recent runs
  runwise latest                      # Analyze latest run
  runwise run <ID>                    # Analyze specific run
  runwise compare <A> <B>             # Compare two runs
  runwise history -k loss,val_loss    # Get training history as CSV
  runwise keys                        # List available metric keys

  # Local JSONL logs (standalone files, not W&B)
  runwise local                       # List logs in logs/ directory
  runwise local <file> --keys         # List available keys in file
  runwise local <file> --history -k loss,val_loss  # Get history CSV
  runwise local <file> --stats -k loss,val_loss    # Get statistics

  # W&B Cloud API (when no local files)
  runwise api -p my-project           # List runs from W&B cloud

  # Output formats
  runwise latest --format md          # Markdown for GitHub/Notion
"""

import argparse
from pathlib import Path

from .config import MetricGroup, MetricSchema, RunwiseConfig
from .core import RunAnalyzer
from .formatters.markdown import MarkdownFormatter, to_markdown


def _no_wandb_runs_message(analyzer: RunAnalyzer) -> str:
    """Generate helpful message when no W&B runs are found."""
    lines = [
        "No W&B runs found in wandb/ directory.",
        "",
    ]

    # Check for local logs
    local_logs = analyzer.list_local_logs()
    if local_logs:
        lines.append("Found local JSONL logs instead. Use 'runwise local' commands:")
        lines.append(f"  runwise local                    # List {len(local_logs)} available logs")
        lines.append(f"  runwise local {local_logs[0].name} --keys")
        lines.append(f"  runwise local {local_logs[0].name} --history -k <metrics>")
        lines.append(f"  runwise local {local_logs[0].name} --stats -k <metrics>")
    else:
        lines.append("Options:")
        lines.append("  1. Ensure you're in a directory with wandb/ folder")
        lines.append("  2. For local JSONL logs: runwise local <file>")
        lines.append("  3. For W&B cloud: runwise api -p <project> (requires: pip install wandb)")

    return "\n".join(lines)


def _run_not_found_message(run_id: str, analyzer: RunAnalyzer) -> str:
    """Generate helpful message when a specific run is not found."""
    lines = [f"Run '{run_id}' not found.", ""]

    # List available runs
    runs = analyzer.list_runs(limit=5)
    if runs:
        lines.append("Available W&B runs:")
        for run in runs:
            lines.append(f"  {run.run_id}")
    else:
        lines.append(_no_wandb_runs_message(analyzer))

    return "\n".join(lines)


def cmd_list(analyzer: RunAnalyzer, args):
    """List recent W&B runs from local wandb/ directory."""
    runs = analyzer.list_runs(limit=args.limit)
    if runs:
        show_sparklines = not getattr(args, 'no_spark', False)
        output = analyzer.format_run_list(runs, show_sparklines=show_sparklines)

        # Apply format if requested
        fmt = getattr(args, 'format', None)
        if fmt == 'md':
            output = to_markdown(output, "list")

        print(output)
    else:
        print(_no_wandb_runs_message(analyzer))


def cmd_latest(analyzer: RunAnalyzer, args):
    """Summarize latest W&B run."""
    run = analyzer.get_latest_run()
    if run:
        include_anomalies = not getattr(args, 'no_anomalies', False)
        include_sparklines = not getattr(args, 'no_spark', False)
        output = analyzer.summarize_run(
            run,
            include_anomalies=include_anomalies,
            include_sparklines=include_sparklines
        )

        # Apply format if requested
        fmt = getattr(args, 'format', None)
        if fmt == 'md':
            formatter = MarkdownFormatter()
            output = formatter.format_run_summary(run, output)

        print(output)
    else:
        print(_no_wandb_runs_message(analyzer))


def cmd_run(analyzer: RunAnalyzer, args):
    """Summarize specific W&B run."""
    run = analyzer.find_run(args.run_id)
    if run:
        include_anomalies = not getattr(args, 'no_anomalies', False)
        include_sparklines = not getattr(args, 'no_spark', False)
        output = analyzer.summarize_run(
            run,
            include_anomalies=include_anomalies,
            include_sparklines=include_sparklines
        )

        # Apply format if requested
        fmt = getattr(args, 'format', None)
        if fmt == 'md':
            formatter = MarkdownFormatter()
            output = formatter.format_run_summary(run, output)

        print(output)
    else:
        print(_run_not_found_message(args.run_id, analyzer))


def cmd_compare(analyzer: RunAnalyzer, args):
    """Compare two W&B runs."""
    run_a = analyzer.find_run(args.run_a)
    run_b = analyzer.find_run(args.run_b)

    if not run_a:
        print(_run_not_found_message(args.run_a, analyzer))
        return
    if not run_b:
        print(_run_not_found_message(args.run_b, analyzer))
        return

    output = analyzer.compare_runs(
        run_a,
        run_b,
        filter_prefix=args.filter,
        show_config_diff=args.diff
    )

    # Apply format if requested
    fmt = getattr(args, 'format', None)
    if fmt == 'md':
        formatter = MarkdownFormatter()
        output = formatter.format_comparison(output)

    print(output)


def cmd_config(analyzer: RunAnalyzer, args):
    """Show config/hyperparameters for a W&B run."""
    run = analyzer.find_run(args.run_id) if args.run_id else analyzer.get_latest_run()

    if not run:
        if args.run_id:
            print(_run_not_found_message(args.run_id, analyzer))
        else:
            print(_no_wandb_runs_message(analyzer))
        return

    print(analyzer.get_config(run))


def cmd_notes(analyzer: RunAnalyzer, args):
    """Show run context (name, notes, tags, group) for a W&B run."""
    run = analyzer.find_run(args.run_id) if args.run_id else analyzer.get_latest_run()

    if not run:
        if args.run_id:
            print(_run_not_found_message(args.run_id, analyzer))
        else:
            print(_no_wandb_runs_message(analyzer))
        return

    print(analyzer.get_run_context(run))


def cmd_best(analyzer: RunAnalyzer, args):
    """Find the best run by a metric."""
    print(analyzer.format_best_run(
        metric=args.metric,
        limit=args.limit,
        higher_is_better=args.higher_is_better
    ))


def cmd_history(analyzer: RunAnalyzer, args):
    """Get downsampled training history from a W&B run."""
    run = analyzer.find_run(args.run_id) if args.run_id else analyzer.get_latest_run()

    if not run:
        if args.run_id:
            print(_run_not_found_message(args.run_id, analyzer))
        else:
            print(_no_wandb_runs_message(analyzer))
        return

    # If no keys specified, try to auto-detect common metrics
    if not args.keys:
        keys = _get_default_metric_keys(analyzer, run)
        if not keys:
            print("No keys specified. Use -k/--keys or see available keys:")
            print(analyzer.list_available_keys(run))
            return
        print(f"(auto-detected keys: {', '.join(keys)})\n")
    else:
        keys = [k.strip() for k in args.keys.split(",")]

    print(analyzer.get_history(run, keys, samples=args.samples))


def cmd_stats(analyzer: RunAnalyzer, args):
    """Get history statistics from a W&B run (even more compact than history)."""
    run = analyzer.find_run(args.run_id) if args.run_id else analyzer.get_latest_run()

    if not run:
        if args.run_id:
            print(_run_not_found_message(args.run_id, analyzer))
        else:
            print(_no_wandb_runs_message(analyzer))
        return

    # If no keys specified, try to auto-detect common metrics
    if not args.keys:
        keys = _get_default_metric_keys(analyzer, run)
        if not keys:
            print("No keys specified. Use -k/--keys or see available keys:")
            print(analyzer.list_available_keys(run))
            return
        print(f"(auto-detected keys: {', '.join(keys)})\n")
    else:
        keys = [k.strip() for k in args.keys.split(",")]

    print(analyzer.get_history_stats(run, keys))


def _get_default_metric_keys(analyzer: RunAnalyzer, run) -> list[str]:
    """Try to auto-detect common metric keys from run history."""
    # Common metric patterns to look for
    common_patterns = [
        "loss", "train/loss", "train_loss",
        "val_loss", "val/loss", "validation_loss",
        "accuracy", "train/accuracy", "acc",
        "val_accuracy", "val/accuracy", "val_acc",
        "lr", "learning_rate",
    ]

    # Get available keys from the run
    available = set()
    history_file = run.directory / "files" / "wandb-history.jsonl"
    if history_file.exists():
        import json
        with open(history_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Only check first few lines
                    break
                try:
                    record = json.loads(line.strip())
                    available.update(k for k in record.keys() if not k.startswith("_"))
                except Exception:
                    continue

    # Find matching keys
    found_keys = []
    for pattern in common_patterns:
        if pattern in available:
            found_keys.append(pattern)
        # Also check for partial matches
        for key in available:
            if pattern in key.lower() and key not in found_keys:
                found_keys.append(key)

    # Limit to reasonable number
    return found_keys[:6]


def cmd_keys(analyzer: RunAnalyzer, args):
    """List available metric keys in a W&B run."""
    run = analyzer.find_run(args.run_id) if args.run_id else analyzer.get_latest_run()

    if not run:
        if args.run_id:
            print(_run_not_found_message(args.run_id, analyzer))
        else:
            print(_no_wandb_runs_message(analyzer))
        return

    print(analyzer.list_available_keys(run))


def cmd_stability(analyzer: RunAnalyzer, args):
    """Analyze training stability using rolling standard deviation."""
    run = analyzer.find_run(args.run_id) if args.run_id else analyzer.get_latest_run()

    if not run:
        if args.run_id:
            print(_run_not_found_message(args.run_id, analyzer))
        else:
            print(_no_wandb_runs_message(analyzer))
        return

    # Keys are required for stability analysis
    if not args.keys:
        print("Error: -k/--keys is required for stability analysis")
        print("")
        print("First, list available keys:")
        print("  runwise keys")
        print("")
        print("Then analyze stability:")
        print("  runwise stability -k loss,val_loss --window 100")
        return

    keys = [k.strip() for k in args.keys.split(",")]
    window = args.window

    if args.csv:
        # Output rolling stats as CSV
        print(analyzer.get_stability_csv(run, keys, window=window, samples=args.samples))
    else:
        # Output summary report
        print(analyzer.get_stability_analysis(run, keys, window=window))


def cmd_live(analyzer: RunAnalyzer, args):
    """Show live training status."""
    print(analyzer.get_live_status())


def cmd_local(analyzer: RunAnalyzer, args):
    """List or analyze local JSONL log files."""
    # If no file specified, list available logs
    if not args.file:
        logs = analyzer.list_local_logs()
        if logs:
            print("LOCAL LOGS (in logs/ directory):")
            print("")
            for log in logs[:15]:
                records = analyzer.parse_local_log(log)
                if records:
                    max_step = max(r.get("step", r.get("_step", 0)) for r in records)
                    # Show some available keys
                    sample_keys = [k for k in records[0].keys() if not k.startswith("_")][:5]
                    keys_hint = ", ".join(sample_keys)
                    print(f"  {log.name}: {len(records)} records, step {max_step}")
                    print(f"    keys: {keys_hint}...")
                else:
                    print(f"  {log.name}: empty")
            print("")
            print("Usage: runwise local <file> [--keys | --history -k <keys> | --stats -k <keys>]")
        else:
            print("No local logs found in logs/ directory")
            print("")
            print("Expected location: logs/*.jsonl")
            print("Or specify path directly: runwise local /path/to/file.jsonl")
        return

    # Find the log file
    log_file = analyzer.find_local_log(args.file)
    if not log_file:
        print(f"Log file not found: {args.file}")
        print("")
        print("Try: runwise local  (to list available logs)")
        return

    # Handle different modes
    if getattr(args, 'list_keys', False):
        # List available keys
        print(analyzer.list_local_log_keys(log_file))
        return

    if getattr(args, 'history', False):
        # Get downsampled history
        if not args.keys:
            print("Error: --history requires -k/--keys to specify which metrics")
            print("")
            print("First, list available keys:")
            print(f"  runwise local {args.file} --keys")
            print("")
            print("Then get history:")
            print(f"  runwise local {args.file} --history -k loss,val_loss")
            return
        keys = [k.strip() for k in args.keys.split(",")]
        samples = getattr(args, 'samples', 500)
        print(analyzer.get_local_history(log_file, keys, samples))
        return

    if getattr(args, 'stats', False):
        # Get statistics
        if not args.keys:
            print("Error: --stats requires -k/--keys to specify which metrics")
            print("")
            print("First, list available keys:")
            print(f"  runwise local {args.file} --keys")
            print("")
            print("Then get stats:")
            print(f"  runwise local {args.file} --stats -k loss,val_loss")
            return
        keys = [k.strip() for k in args.keys.split(",")]
        print(analyzer.get_local_history_stats(log_file, keys))
        return

    if getattr(args, 'stability', False):
        # Get stability analysis
        if not args.keys:
            print("Error: --stability requires -k/--keys to specify which metrics")
            print("")
            print("First, list available keys:")
            print(f"  runwise local {args.file} --keys")
            print("")
            print("Then analyze stability:")
            print(f"  runwise local {args.file} --stability -k loss,val_loss --window 100")
            return
        keys = [k.strip() for k in args.keys.split(",")]
        window = getattr(args, 'window', 100)
        print(analyzer.get_local_stability_analysis(log_file, keys, window=window))
        return

    # Default: show summary
    print(analyzer.summarize_local_log(log_file))


def cmd_init(analyzer: RunAnalyzer, args):
    """Initialize runwise.json configuration."""
    config_path = Path("runwise.json")
    if config_path.exists() and not args.force:
        print("runwise.json already exists. Use --force to overwrite.")
        return

    # Create default config
    config = RunwiseConfig(
        project_name=args.name or "ML Project",
        wandb_dir=Path("wandb"),
        logs_dir=Path("logs"),
        schema=MetricSchema(
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
            validation_sets={"val": "Validation"},
        )
    )

    config.save(config_path)
    print(f"Created {config_path}")
    print("Edit this file to customize metrics for your project.")


def cmd_tb(analyzer: RunAnalyzer, args):
    """List or analyze TensorBoard runs."""
    from .tensorboard import TENSORBOARD_AVAILABLE, TensorBoardParser

    if not TENSORBOARD_AVAILABLE:
        print("TensorBoard support requires the tensorboard package.")
        print("Install with: pip install tensorboard")
        return

    log_dir = Path(args.log_dir) if args.log_dir else Path(".")
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        return

    try:
        parser = TensorBoardParser(log_dir)
        runs = parser.list_runs()

        if not runs:
            print(f"No TensorBoard runs found in {log_dir}")
            return

        if args.run_id:
            # Find and summarize specific run
            run = next((r for r in runs if args.run_id in r.run_id), None)
            if run:
                print(parser.summarize_run(run))
            else:
                print(f"Run '{args.run_id}' not found")
        else:
            # List all runs
            print(parser.format_run_list(runs))

    except Exception as e:
        print(f"Error reading TensorBoard logs: {e}")


def cmd_api(analyzer: RunAnalyzer, args):
    """Access W&B runs via cloud API."""
    from .wandb_api import WANDB_API_AVAILABLE, WandbAPIClient

    if not WANDB_API_AVAILABLE:
        print("W&B API support requires the wandb package.")
        print("Install with: pip install wandb")
        return

    if not args.project:
        print("Error: --project is required for API access")
        print("Usage: runwise api --project my-project [--entity my-team]")
        return

    try:
        client = WandbAPIClient(project=args.project, entity=args.entity)

        if args.run_id:
            # Summarize specific run
            run = client.get_run(args.run_id)
            if run:
                print(client.summarize_run(run))
            else:
                print(f"Run '{args.run_id}' not found")
        elif args.compare:
            # Compare two runs
            run_ids = args.compare.split(",")
            if len(run_ids) != 2:
                print("Error: --compare requires two run IDs separated by comma")
                return
            run_a = client.get_run(run_ids[0].strip())
            run_b = client.get_run(run_ids[1].strip())
            if not run_a:
                print(f"Run '{run_ids[0]}' not found")
                return
            if not run_b:
                print(f"Run '{run_ids[1]}' not found")
                return
            print(client.compare_runs(run_a, run_b))
        else:
            # List runs
            filters = None
            if args.state:
                filters = {"state": args.state}
            runs = client.list_runs(limit=args.limit, filters=filters)
            if runs:
                print(client.format_run_list(runs))
            else:
                print("No runs found")

    except Exception as e:
        error_str = str(e).lower()
        # Detect common authentication/authorization errors
        if any(term in error_str for term in [
            "unauthorized", "403", "401", "authentication", "api key",
            "not logged in", "login", "permission", "forbidden", "netrc"
        ]):
            print("W&B API authentication error.")
            print("")
            print("You need to log in to W&B first:")
            print("  wandb login")
            print("")
            print("This will open a browser to get your API key, or you can")
            print("paste it from: https://wandb.ai/authorize")
            print("")
            print(f"Original error: {e}")
        elif "not found" in error_str or "404" in error_str:
            print(f"Project or entity not found: {args.project}")
            print("")
            print("Check that:")
            print(f"  1. Project '{args.project}' exists")
            if args.entity:
                print(f"  2. Entity '{args.entity}' is correct")
            else:
                print("  2. You have access to this project")
            print("  3. You're logged in: wandb login")
        else:
            print(f"Error accessing W&B API: {e}")
            print("")
            print("Common fixes:")
            print("  1. Check your internet connection")
            print("  2. Verify you're logged in: wandb login")
            print("  3. Check project/entity names are correct")


def main():
    parser = argparse.ArgumentParser(
        description="Runwise: Token-efficient ML training run analysis for AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DATA SOURCES (choose based on your setup):

  W&B Local (default) - reads from local wandb/ directory:
    runwise list                # List runs
    runwise latest              # Analyze latest run
    runwise run <id>            # Analyze specific run
    runwise history -k loss     # Get training history

  Local JSONL - for standalone log files (not W&B):
    runwise local               # List files in logs/
    runwise local <file> --keys # List available metrics
    runwise local <file> --history -k loss,val_loss

  W&B Cloud API - when no local files (requires: pip install wandb):
    runwise api -p <project>    # List runs from cloud

  TensorBoard - for tfevents files (requires: pip install tensorboard):
    runwise tb                  # List TB runs

TIPS FOR AI AGENTS:
  - Always run 'runwise keys' or 'runwise local <file> --keys' first
    to discover available metric names before querying history/stats
  - Use --format md for output suitable for GitHub issues or documentation
  - The 'history' and 'stats' commands require -k to specify metric keys
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list
    p_list = subparsers.add_parser("list", help="List recent runs with sparkline trends")
    p_list.add_argument("-n", "--limit", type=int, default=15, help="Number of runs to show")
    p_list.add_argument("--no-spark", action="store_true", help="Disable sparklines (faster)")
    p_list.add_argument("--format", choices=["text", "md"], default="text",
                        help="Output format (default: text, md for Markdown)")

    # latest
    p_latest = subparsers.add_parser("latest", help="Summarize latest run with anomaly detection")
    p_latest.add_argument("--no-spark", action="store_true", help="Disable sparklines")
    p_latest.add_argument("--no-anomalies", action="store_true", help="Disable anomaly detection")
    p_latest.add_argument("--format", choices=["text", "md"], default="text",
                          help="Output format (default: text, md for Markdown)")

    # run
    p_run = subparsers.add_parser("run", help="Summarize specific run")
    p_run.add_argument("run_id", help="Run ID to analyze")
    p_run.add_argument("--no-spark", action="store_true", help="Disable sparklines")
    p_run.add_argument("--no-anomalies", action="store_true", help="Disable anomaly detection")
    p_run.add_argument("--format", choices=["text", "md"], default="text",
                       help="Output format (default: text, md for Markdown)")

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare two runs")
    p_compare.add_argument("run_a", help="First run ID")
    p_compare.add_argument("run_b", help="Second run ID")
    p_compare.add_argument("-f", "--filter", help="Filter metrics by prefix (e.g., 'val', 'train')")
    p_compare.add_argument("-d", "--diff", action="store_true", help="Show config differences")
    p_compare.add_argument("--format", choices=["text", "md"], default="text",
                           help="Output format (default: text, md for Markdown)")

    # config
    p_config = subparsers.add_parser("config", help="Show hyperparameters/config for a run")
    p_config.add_argument("run_id", nargs="?", help="Run ID (uses latest if omitted)")

    # notes
    p_notes = subparsers.add_parser("notes", help="Show run context (name, notes, tags, group)")
    p_notes.add_argument("run_id", nargs="?", help="Run ID (uses latest if omitted)")

    # best
    p_best = subparsers.add_parser("best", help="Find the best run by a metric")
    p_best.add_argument("metric", help="Metric to compare (e.g., 'val_loss', 'accuracy')")
    p_best.add_argument("-n", "--limit", type=int, default=10, help="Number of runs to consider (default: 10)")
    p_best.add_argument("--max", dest="higher_is_better", action="store_true",
                        help="Higher values are better (default: lower is better)")

    # history (downsampled)
    p_history = subparsers.add_parser("history", help="Get downsampled training history (CSV)")
    p_history.add_argument("run_id", nargs="?", help="Run ID (uses latest if omitted)")
    p_history.add_argument("-k", "--keys", help="Comma-separated metric keys (auto-detects if omitted)")
    p_history.add_argument("-n", "--samples", type=int, default=500, help="Number of samples (default: 500)")

    # stats (even more compact)
    p_stats = subparsers.add_parser("stats", help="Get history statistics (min/max/mean)")
    p_stats.add_argument("run_id", nargs="?", help="Run ID (uses latest if omitted)")
    p_stats.add_argument("-k", "--keys", help="Comma-separated metric keys (auto-detects if omitted)")

    # stability (rolling std dev analysis)
    p_stability = subparsers.add_parser(
        "stability",
        help="Analyze training stability (rolling std dev)",
        description="""Analyze training stability using rolling standard deviation.

Measures how stable/noisy training is over time by calculating
local standard deviation within a sliding window.

Useful for:
  - Identifying noisy training periods
  - Confirming stable convergence (decreasing std dev)
  - Detecting instability before it causes problems

Examples:
  runwise stability -k loss,val_loss              # Default 100-step window
  runwise stability -k loss --window 50           # Smaller window (more sensitive)
  runwise stability -k loss --window 200          # Larger window (smoother)
  runwise stability -k loss --csv                 # Output as CSV for plotting
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p_stability.add_argument("run_id", nargs="?", help="Run ID (uses latest if omitted)")
    p_stability.add_argument("-k", "--keys", required=False,
                             help="Comma-separated metric keys (required)")
    p_stability.add_argument("-w", "--window", type=int, default=100,
                             help="Rolling window size in steps (default: 100)")
    p_stability.add_argument("--csv", action="store_true",
                             help="Output rolling mean/std as CSV instead of summary")
    p_stability.add_argument("-n", "--samples", type=int, default=100,
                             help="Number of CSV rows when using --csv (default: 100)")

    # keys (list available)
    p_keys = subparsers.add_parser("keys", help="List available metric keys in a run")
    p_keys.add_argument("run_id", nargs="?", help="Run ID (uses latest if omitted)")

    # live
    subparsers.add_parser("live", help="Show live training status")

    # local
    p_local = subparsers.add_parser(
        "local",
        help="Analyze local JSONL log files (not W&B runs)",
        description="""Analyze standalone JSONL log files in logs/ directory.

For W&B runs, use 'runwise list' and 'runwise run <id>' instead.
This command is for local log files like metrics_*.jsonl.

Examples:
  runwise local                              # List available local logs
  runwise local metrics.jsonl                # Show summary of log file
  runwise local metrics.jsonl --keys         # List available metric keys
  runwise local metrics.jsonl --history -k loss,val_loss  # Get history CSV
  runwise local metrics.jsonl --stats -k loss,val_loss    # Get statistics
  runwise local metrics.jsonl --stability -k loss --window 100  # Stability analysis
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p_local.add_argument("file", nargs="?", help="Log file name or path (omit to list available)")
    p_local.add_argument("--keys", dest="list_keys", action="store_true",
                         help="List available metric keys in the log file")
    p_local.add_argument("--history", action="store_true",
                         help="Get downsampled history as CSV (requires -k)")
    p_local.add_argument("--stats", action="store_true",
                         help="Get statistics (min/max/mean/final) for metrics (requires -k)")
    p_local.add_argument("--stability", action="store_true",
                         help="Analyze training stability with rolling std dev (requires -k)")
    p_local.add_argument("-k", "--metrics", dest="keys",
                         help="Comma-separated metric keys (e.g., loss,val_loss,accuracy)")
    p_local.add_argument("-n", "--samples", type=int, default=500,
                         help="Number of samples for --history (default: 500)")
    p_local.add_argument("-w", "--window", type=int, default=100,
                         help="Rolling window size for --stability (default: 100)")

    # init
    p_init = subparsers.add_parser("init", help="Initialize configuration")
    p_init.add_argument("--name", help="Project name")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing config")

    # TensorBoard
    p_tb = subparsers.add_parser("tb", help="List/analyze TensorBoard runs (requires tensorboard)")
    p_tb.add_argument("--log-dir", "-d", help="TensorBoard log directory (default: current)")
    p_tb.add_argument("--run", "-r", dest="run_id", help="Specific run to analyze")

    # W&B API
    p_api = subparsers.add_parser("api", help="Access W&B runs via cloud API (requires wandb)")
    p_api.add_argument("--project", "-p", help="W&B project name (required)")
    p_api.add_argument("--entity", "-e", help="W&B entity (username or team)")
    p_api.add_argument("--run", "-r", dest="run_id", help="Specific run to analyze")
    p_api.add_argument("--compare", "-c", help="Compare two runs (comma-separated IDs)")
    p_api.add_argument("--state", "-s", help="Filter by state (running, finished, crashed)")
    p_api.add_argument("-n", "--limit", type=int, default=15, help="Number of runs to show")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize analyzer
    try:
        config = RunwiseConfig.auto_detect()
    except Exception as e:
        print(f"Warning: Could not auto-detect config: {e}")
        config = RunwiseConfig()

    analyzer = RunAnalyzer(config)

    # Dispatch command
    commands = {
        "list": cmd_list,
        "latest": cmd_latest,
        "run": cmd_run,
        "compare": cmd_compare,
        "config": cmd_config,
        "notes": cmd_notes,
        "best": cmd_best,
        "history": cmd_history,
        "stats": cmd_stats,
        "stability": cmd_stability,
        "keys": cmd_keys,
        "live": cmd_live,
        "local": cmd_local,
        "init": cmd_init,
        "tb": cmd_tb,
        "api": cmd_api,
    }

    if args.command in commands:
        commands[args.command](analyzer, args)


if __name__ == "__main__":
    main()
