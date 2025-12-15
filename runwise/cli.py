#!/usr/bin/env python3
"""
Command-line interface for Runwise.

Usage:
    runwise list                        # List recent runs (with state)
    runwise latest                      # Summarize latest run
    runwise run <ID>                    # Summarize specific run
    runwise compare <A> <B>             # Compare two runs
    runwise config [ID]                 # Show hyperparameters/config
    runwise notes [ID]                  # Show run context (name, notes, tags)
    runwise best <metric>               # Find best run by metric
    runwise history -k loss,val_loss    # Get downsampled history (CSV)
    runwise stats -k loss,val_loss      # Get history statistics
    runwise keys [ID]                   # List available metric keys
    runwise live                        # Show live training status
    runwise local                       # List/analyze local logs
    runwise init                        # Initialize runwise.json config
"""

import argparse
from pathlib import Path

from .config import MetricGroup, MetricSchema, RunwiseConfig
from .core import RunAnalyzer


def cmd_list(analyzer: RunAnalyzer, args):
    """List recent runs."""
    runs = analyzer.list_runs(limit=args.limit)
    if runs:
        print(analyzer.format_run_list(runs))
    else:
        print("No runs found")


def cmd_latest(analyzer: RunAnalyzer, args):
    """Summarize latest run."""
    run = analyzer.get_latest_run()
    if run:
        print(analyzer.summarize_run(run))
    else:
        print("No latest run found")


def cmd_run(analyzer: RunAnalyzer, args):
    """Summarize specific run."""
    run = analyzer.find_run(args.run_id)
    if run:
        print(analyzer.summarize_run(run))
    else:
        print(f"Run '{args.run_id}' not found")


def cmd_compare(analyzer: RunAnalyzer, args):
    """Compare two runs."""
    run_a = analyzer.find_run(args.run_a)
    run_b = analyzer.find_run(args.run_b)

    if not run_a:
        print(f"Run '{args.run_a}' not found")
        return
    if not run_b:
        print(f"Run '{args.run_b}' not found")
        return

    print(analyzer.compare_runs(
        run_a,
        run_b,
        filter_prefix=args.filter,
        show_config_diff=args.diff
    ))


def cmd_config(analyzer: RunAnalyzer, args):
    """Show config/hyperparameters for a run."""
    run = analyzer.find_run(args.run_id) if args.run_id else analyzer.get_latest_run()

    if not run:
        print("Run not found")
        return

    print(analyzer.get_config(run))


def cmd_notes(analyzer: RunAnalyzer, args):
    """Show run context (name, notes, tags, group)."""
    run = analyzer.find_run(args.run_id) if args.run_id else analyzer.get_latest_run()

    if not run:
        print("Run not found")
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
    """Get downsampled training history."""
    run = analyzer.find_run(args.run_id) if args.run_id else analyzer.get_latest_run()

    if not run:
        print("Run not found")
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
    """Get history statistics (even more compact than history)."""
    run = analyzer.find_run(args.run_id) if args.run_id else analyzer.get_latest_run()

    if not run:
        print("Run not found")
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
    """List available metric keys in a run."""
    run = analyzer.find_run(args.run_id) if args.run_id else analyzer.get_latest_run()

    if not run:
        print("Run not found")
        return

    print(analyzer.list_available_keys(run))


def cmd_live(analyzer: RunAnalyzer, args):
    """Show live training status."""
    print(analyzer.get_live_status())


def cmd_local(analyzer: RunAnalyzer, args):
    """List or analyze local logs."""
    if args.file:
        log_file = Path(args.file)
        if not log_file.exists():
            log_file = analyzer.config.logs_dir / args.file
        print(analyzer.summarize_local_log(log_file))
    else:
        logs = analyzer.list_local_logs()
        if logs:
            print("LOCAL LOGS:")
            for log in logs[:10]:
                records = analyzer.parse_local_log(log)
                if records:
                    max_step = max(r.get("step", r.get("_step", 0)) for r in records)
                    print(f"  {log.name}: {len(records)} records, max step {max_step}")
                else:
                    print(f"  {log.name}: empty")
        else:
            print("No local logs found")


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


def main():
    parser = argparse.ArgumentParser(
        description="Runwise: Token-efficient ML training run analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  runwise list              # List recent W&B runs
  runwise latest            # Summarize latest run
  runwise run abc123        # Summarize specific run
  runwise compare abc def   # Compare two runs
  runwise live              # Show live training status
  runwise local             # List local log files
  runwise local train.jsonl # Analyze specific log
  runwise init              # Create runwise.json config
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list
    p_list = subparsers.add_parser("list", help="List recent runs")
    p_list.add_argument("-n", "--limit", type=int, default=15, help="Number of runs to show")

    # latest
    subparsers.add_parser("latest", help="Summarize latest run")

    # run
    p_run = subparsers.add_parser("run", help="Summarize specific run")
    p_run.add_argument("run_id", help="Run ID to analyze")

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare two runs")
    p_compare.add_argument("run_a", help="First run ID")
    p_compare.add_argument("run_b", help="Second run ID")
    p_compare.add_argument("-f", "--filter", help="Filter metrics by prefix (e.g., 'val', 'train')")
    p_compare.add_argument("-d", "--diff", action="store_true", help="Show config differences")

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

    # keys (list available)
    p_keys = subparsers.add_parser("keys", help="List available metric keys in a run")
    p_keys.add_argument("run_id", nargs="?", help="Run ID (uses latest if omitted)")

    # live
    subparsers.add_parser("live", help="Show live training status")

    # local
    p_local = subparsers.add_parser("local", help="List/analyze local logs")
    p_local.add_argument("file", nargs="?", help="Log file to analyze")

    # init
    p_init = subparsers.add_parser("init", help="Initialize configuration")
    p_init.add_argument("--name", help="Project name")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing config")

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
        "keys": cmd_keys,
        "live": cmd_live,
        "local": cmd_local,
        "init": cmd_init,
    }

    if args.command in commands:
        commands[args.command](analyzer, args)


if __name__ == "__main__":
    main()
