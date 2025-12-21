# CLAUDE.md - Development Guide for Runwise

## Project Overview

**Runwise** is a token-efficient ML training run analysis tool designed for AI agents. It parses W&B and local training logs, generating condensed summaries optimized for LLM context windows.

**Core Philosophy**: The "Zoom-In Strategy" - hierarchical retrieval to minimize token usage:
1. **Macro View**: List runs and status (low cost)
2. **Summary View**: Single-point metrics like "best val_loss" (low cost)
3. **Sampled View**: Downsampled history curves (fixed token cost)
4. **Micro View**: Raw data only when absolutely necessary

## Repository Structure

```
runwise/
├── runwise/              # Main package
│   ├── __init__.py       # Exports: RunAnalyzer, RunwiseConfig, MetricSchema, sparklines, anomalies
│   ├── cli.py            # CLI entry point (runwise command)
│   ├── config.py         # Configuration and metric schema definitions
│   ├── core.py           # RunAnalyzer - main analysis logic
│   ├── sparklines.py     # Sparkline generation for trend visualization
│   ├── anomalies.py      # Anomaly detection (spikes, overfitting, plateaus)
│   ├── tensorboard.py    # Optional TensorBoard support (requires tensorboard package)
│   ├── formatters/       # Output formatters (placeholder for expansion)
│   └── parsers/          # Log parsers (placeholder for expansion)
├── mcp_server/
│   └── server.py         # MCP server for Claude Code integration
├── .claude/
│   └── skills/
│       └── runwise/      # Claude Code skill (recommended for Claude Code users)
│           ├── SKILL.md      # Main instructions and command reference
│           └── examples.md   # Common workflow examples
├── tests/                # pytest tests (94 tests)
├── examples/
│   ├── training_log.jsonl        # Sample JSONL training log
│   ├── runwise_config_example.json # Example configuration
│   ├── expected_outputs.md       # Example command outputs
│   └── README.md                 # Examples documentation
├── pyproject.toml        # Package configuration
├── README.md             # User documentation
├── ROADMAP.md            # Feature roadmap
└── runwise_spec.md       # Original design specification
```

## Key Components

### RunAnalyzer (core.py)
The main class providing all analysis functionality:
- `list_runs(limit)` - Enumerate W&B runs from local directory
- `get_latest_run()` - Get the most recent/active run
- `find_run(run_id)` - Find specific run by ID
- `summarize_run(run, include_anomalies, include_sparklines)` - Generate token-efficient summary with optional anomaly detection and sparklines
- `compare_runs(run_a, run_b)` - Side-by-side metric comparison
- `get_run_context(run)` - Get full run context (name, notes, tags, group)
- `get_history_data(run, keys, samples)` - Get history as list of dicts (for anomaly detection/sparklines)
- `get_stability_analysis(run, keys, window)` - Analyze training stability with rolling std dev
- `get_stability_csv(run, keys, window, samples)` - Stability analysis as CSV output
- `get_live_status()` - Parse output.log for live training status
- Local log methods for JSONL files (including stability analysis)

### Sparklines (sparklines.py)
Token-efficient trend visualization using Unicode block characters:
- `sparkline(values, width)` - Convert values to sparkline string (e.g., "▁▂▄▆█")
- `sparkline_with_stats(values, width)` - Sparkline with first/last values
- `trend_indicator(values)` - Simple trend arrow (↑/↓/→/~)
- `calculate_slope(values, steps)` - Robust slope via Theil-Sen estimator (handles noisy data)
- `calculate_windowed_slopes(values, steps, window)` - Slope over sliding windows
- `format_metric_with_spark(name, values)` - Full metric line with sparkline

### Anomaly Detection (anomalies.py)
Lightweight, deterministic heuristics for detecting training issues:
- `detect_anomalies(history, config)` - Returns list of Anomaly objects (empty if healthy)
- `format_anomalies(anomalies)` - Format for output (zero tokens if no anomalies)
- Detection types: loss spikes (MAD-based), overfitting (val/train ratio), plateaus, gradient issues, NaN/Inf, throughput drops
- `AnomalyConfig` - Configurable thresholds for all detectors

### TensorBoard Support (tensorboard.py)
Optional TensorBoard event file parsing (requires `pip install tensorboard`):
- `TensorBoardParser(log_dir)` - Parse tfevents files
- `list_runs()` - List available TB runs
- `summarize_run(run)` - Generate summary with sparklines and anomaly detection
- Import with: `from runwise.tensorboard import TensorBoardParser, TENSORBOARD_AVAILABLE`

### W&B API Support (wandb_api.py)
Optional W&B cloud API access (requires `pip install wandb`):
- `WandbAPIClient(project, entity)` - Connect to W&B cloud
- `list_runs(limit, filters)` - List runs from cloud
- `get_run(run_id)` - Get specific run
- `get_history(run_id, keys, samples)` - Get downsampled history
- `summarize_run(run)` - Generate summary with sparklines
- Import with: `from runwise.wandb_api import WandbAPIClient, WANDB_API_AVAILABLE`

### Markdown Formatter (formatters/markdown.py)
Convert output to GitHub-flavored markdown:
- `MarkdownFormatter()` - Formatter instance with configurable options
- `format_run_summary(run, text)` - Convert run summary to markdown
- `format_run_list(text)` - Convert run list to markdown table
- `format_comparison(text)` - Convert comparison to markdown
- `to_markdown(text, type)` - Convenience function for auto-detection
- CLI flag: `--format md` on list, latest, run, compare commands

### RunwiseConfig (config.py)
Configuration with auto-detection:
- Looks for `runwise.json` in project root
- Falls back to environment variables (RUNWISE_WANDB_DIR, RUNWISE_LOGS_DIR)
- Auto-detects wandb/ and logs/ directories

### MetricSchema (config.py)
Defines how to interpret project-specific metrics:
- `loss_key`, `step_key` - Core metric names
- `primary_metric` - Main metric for run list display
- `groups` - Metric groupings for detailed analysis
- `per_step_pattern` - For iterative models (e.g., "train/loss_step_{i}")
- `validation_sets` - Validation set prefixes

## CLI Commands

```bash
# List and summarize runs
runwise list                          # List runs with sparkline trends
runwise list --no-spark               # List without sparklines (faster)
runwise latest                        # Summarize with anomaly detection + sparklines
runwise latest -k loss,val_loss       # Show only specific metrics (NEW v0.4.0)
runwise latest --no-anomalies         # Disable anomaly detection
runwise run <ID>                      # Summarize specific run
runwise run <ID> -k loss,accuracy     # Show only specific metrics (NEW v0.4.0)
runwise run <ID> --no-spark           # Without sparklines

# Compare runs (improved in v0.4.0)
runwise compare <A> <B>               # Compare two runs (full metric names now shown)
runwise compare <A> <B> -f val        # Filter to validation metrics only
runwise compare <A> <B> -g            # Group metrics by prefix (train/, val/, etc.)
runwise compare <A> <B> -t 5          # Only show metrics with >5% delta
runwise compare <A> <B> -d            # Include config differences
runwise compare <A> <B> -f val -t 5 -g -d  # Combine all filters

# Step-matched comparison (NEW - essential for curriculum learning)
runwise compare run1@50000 run2@50000        # Compare at same training step
runwise compare run1@10000 run2@20000        # Compare at different steps
runwise compare run1@50000 run2              # Compare run1@50000 vs run2's final
runwise compare run1@50000 run2@50000 -f val # Step-matched, validation only

# Other local commands
runwise notes [ID]                    # Show run context (name, notes, tags)
runwise history [ID]                  # Get downsampled history (auto-detects keys)
runwise history -k loss,val_loss      # Specify keys explicitly
runwise stats [ID]                    # Get history statistics (auto-detects keys)
runwise stats -k loss,val_loss        # Specify keys explicitly
runwise keys [ID]                     # List available metric keys
runwise live                          # Show live training status (includes run ID)
runwise stability -k loss,val_loss    # Analyze training stability (rolling std dev)
runwise stability -w 50               # Custom window size (default: 100)
runwise stability --csv               # Output as CSV for further analysis

# Local JSONL logs
runwise local                         # List local JSONL logs in logs/ directory
runwise local <file> --keys           # List available metric keys in local log
runwise local <file> --history -k loss,val_loss  # Get history CSV from local log
runwise local <file> --stats -k loss,val_loss    # Get statistics from local log
runwise local <file> --stability -k loss,val_loss  # Stability analysis for local log

# TensorBoard (requires: pip install tensorboard)
runwise tb                            # List TensorBoard runs
runwise tb -r <run_id>                # Summarize specific TensorBoard run

# W&B Cloud API (requires: pip install wandb) - Enhanced in v0.4.0
runwise api -p <project>              # List runs (auto-detects project from wandb/)
runwise api -p <project> -r <run_id>  # Summarize specific run from cloud
runwise api -p <project> -r <run_id> -k loss,val_loss  # Show only specific metrics
runwise api -p <project> --best val/accuracy --max     # Find best run by metric
runwise api -p <project> -r <run_id> --history -k loss,val_loss  # Get metric history
runwise api -p <project> -c run1,run2                  # Compare two runs
runwise api -p <project> -c run1,run2 -f val -t 5 -g   # Compare with filtering
runwise api -p <project> --state running               # Filter by state

# Configuration
runwise init [--name]                 # Initialize runwise.json
runwise <cmd> --format md             # Output as markdown (list, latest, run, compare)
```

## Data Sources

**IMPORTANT FOR AI AGENTS**: Understand which data source to use:

| Source | Command | When to Use |
|--------|---------|-------------|
| W&B Local | `runwise list`, `run`, `history` | Default. Reads from local `wandb/` directory |
| Local JSONL | `runwise local <file>` | For standalone log files (not W&B) |
| W&B Cloud API | `runwise api -p <project>` | When no local files, need cloud access |
| TensorBoard | `runwise tb` | For tfevents files |

### Finding Available Metrics

Before querying history or stats, discover available metric keys:
```bash
# For W&B runs
runwise keys              # List keys in latest run
runwise keys <run_id>     # List keys in specific run

# For local JSONL files
runwise local <file> --keys
```

## Design Decisions

### Local-First Architecture
The current implementation reads from local `wandb/` directories rather than using the W&B API directly. This provides:
- **Zero dependencies** (stdlib only)
- **Offline operation** - No API key needed
- **Fast** - No network latency
- **Privacy** - Data stays local

This differs from the original spec which suggested using `wandb.Api()`. Both approaches are valid; the local approach is better for the MCP use case where you're already in the project directory.

### Token Efficiency
Output is designed to minimize tokens:
- Compact table formatting
- Aligned columns without excessive whitespace
- Percentage formatting for 0-1 values
- Internal metrics (keys starting with `_`) filtered out
- CSV-style output for history data
- **Sparklines**: Convey trend in ~10 tokens vs 50+ for raw numbers (e.g., "▁▂▄▆█")
- **Zero-token anomalies**: Anomaly detection returns empty string for healthy runs

### Downsampling for Large Files
The `get_history()` method handles arbitrarily large files efficiently:
1. **Two-pass approach**: First counts lines (fast), then reads only sampled lines
2. **Never loads full file**: Memory usage stays constant regardless of file size
3. **CSV output**: Most token-dense format for tabular data
4. **Configurable samples**: Default 500, adjustable via `--samples` flag

A 10GB history file with 10 million steps produces the same ~3000 token output as a 1MB file with 1000 steps.

## Areas for Future Development

### Completed
- ~~**Server-side downsampling**~~ - DONE: Implemented local downsampling in `get_history()`
- ~~**CSV output mode**~~ - DONE: `history` command outputs CSV
- ~~**Available keys hint**~~ - DONE: `list_available_keys()` and shown when keys not found
- ~~**Unit tests**~~ - DONE: 94 tests with pytest
- ~~**Anomaly detection**~~ - DONE: Spikes (MAD-based), overfitting, plateaus, gradient issues, NaN/Inf
- ~~**TensorBoard support**~~ - DONE: Optional, requires `pip install tensorboard`
- ~~**Sparkline visualizations**~~ - DONE: Unicode block characters for trend display
- ~~**Stability analysis**~~ - DONE: Rolling std dev for training stability (`runwise stability`)

### High Priority
1. **Optional W&B API support** - For remote run access (not just local files)
2. Improve error handling with user-friendly messages

### Medium Priority
1. Auto-detect common metric patterns
2. Add `--verbose` and `--quiet` flags
3. Schema auto-generation from W&B run
4. MLflow support

### Lower Priority
1. ASCII charts for training curves (beyond sparklines)
2. GitHub Action for PR comments

## Run Context (Names, Notes, Tags)

Runwise supports user-provided run descriptions to give AI agents semantic context about each run. This is extracted from W&B's local metadata.

**How users provide context in W&B:**
```python
import wandb

# Set at init time
wandb.init(
    name="lr-sweep-0.001",  # Short descriptive name
    notes="Testing learning rate 0.001 with new augmentation pipeline. Hypothesis: lower LR will reduce overfitting.",
    tags=["sweep", "lr-test", "augmentation"],
    group="lr-sweep"  # Group related runs together
)

# Or update during/after run
wandb.run.notes = "Updated: found better results with warmup"
```

**Accessing run context:**
- CLI: `runwise notes [run_id]` - Get full run context
- CLI: `runwise latest` or `runwise run <ID>` - Summaries include context if available
- MCP: `get_run_context` tool - Returns name, notes, tags, group
- MCP: `list_runs` - Shows names and tags in run list

## MCP Server

The MCP server (`mcp_server/server.py`) exposes these tools:

**Start here:**
- `health_check` - Quick "how's training going?" tool. Combines status, key metrics, sparklines, and anomaly detection in one call. **START HERE for most queries.**

**Discovery (use first):**
- `list_runs` - List recent training runs with sparkline trends
- `list_keys` - **IMPORTANT: Call FIRST before get_history/get_sparkline** to discover available metrics

**Analysis:**
- `analyze_run` - Detailed analysis with anomaly detection + sparklines (supports `-k keys` for custom metrics)
- `analyze_latest` - Analyze latest/active run
- `compare_runs` - Compare two runs (supports `@step` syntax for step-matched comparison)
- `detect_anomalies` - Run anomaly detection (returns empty if healthy - token-efficient)

**History (call list_keys first!):**
- `get_history` - Get downsampled training history as CSV
- `get_history_stats` - Statistical summary (min/max/mean/final)
- `get_sparkline` - Compact sparkline visualization

**Metadata:**
- `get_config` - Hyperparameters and configuration
- `get_run_context` - Run context (name, notes, tags, group)
- `live_status` - Live training status from output.log
- `find_best_run` - Find best run by a metric

Configure in Claude Code settings:
```json
{
    "mcpServers": {
        "runwise": {
            "command": "python",
            "args": ["-m", "mcp_server.server"],
            "cwd": "/path/to/runwise",
            "env": {
                "RUNWISE_PROJECT_ROOT": "/path/to/ml/project"
            }
        }
    }
}
```

## Claude Code Skill (Recommended)

For Claude Code users, the skill is more token-efficient than MCP (~500 tokens vs ~2000+). The skill invokes CLI commands via Bash.

**Location:** `.claude/skills/runwise/`

**Files:**
- `SKILL.md` - Main instructions, command reference, zoom-in strategy
- `examples.md` - Common workflow examples (loaded on-demand)

**How it works:**
1. Claude auto-discovers the skill based on its description
2. When user asks about training, Claude loads the skill
3. Skill guides Claude to invoke CLI commands (e.g., `runwise latest`)
4. CLI output is already optimized for token efficiency

**When to use Skill vs MCP:**

| Use Case | Best Option |
|----------|-------------|
| Claude Code users | Skill (lower tokens, auto-discovery) |
| Other MCP-compatible tools | MCP server |
| Programmatic/API access | MCP server |

**To use in your project:**
```bash
cp -r .claude/skills/runwise /path/to/your/project/.claude/skills/
```

## Testing

Run tests with pytest:
```bash
pytest tests/ -v           # Full test suite (94 tests)
pytest tests/ -q           # Quick summary
pytest tests/test_sparklines.py  # Just sparkline tests
pytest tests/test_anomalies.py   # Just anomaly detection tests
```

Test coverage:
- `test_core.py` - RunAnalyzer functionality
- `test_config.py` - Configuration and schema loading
- `test_cli.py` - CLI command parsing
- `test_sparklines.py` - Sparkline generation
- `test_anomalies.py` - Anomaly detection heuristics

## Publishing Checklist

Before PyPI release:
- [x] Update URLs in pyproject.toml
- [x] Add LICENSE file
- [x] Verify all imports work after pip install
- [x] Test CLI entry point
- [x] Add unit tests (94 tests)
- [x] Add py.typed marker for type hints
- [x] Add GitHub Actions CI/CD
- [x] Sparklines for trend visualization
- [x] Anomaly detection (spikes, overfitting, plateaus)
- [x] Optional TensorBoard support
- [x] Optional W&B API support
- [x] Markdown export (--format md)
- [x] Example files in /examples
- [x] README badges (PyPI, CI, License)
- [x] Claude Code skill for token-efficient integration

Build and publish:
```bash
pip install build twine
python -m build
twine upload dist/*
```

## Common Patterns

### Adding a New CLI Command
1. Add handler function in `cli.py`: `def cmd_newcmd(analyzer, args)`
2. Add subparser in `main()`
3. Add to `commands` dispatch dict

### Adding a New MCP Tool
1. Add tool definition in `_list_tools()` in `server.py`
2. Add handler in `_call_tool()`

### Adding a New Metric Schema Feature
1. Add field to `MetricSchema` dataclass
2. Update `from_file()` and `default()` class methods
3. Update `RunwiseConfig.save()` to serialize it
4. Use the new field in `RunAnalyzer.summarize_run()`

## Quick Reference

| File | Purpose |
|------|---------|
| `runwise/core.py:24` | RunInfo dataclass (includes name, notes, tags, group) |
| `runwise/core.py:49` | RunAnalyzer class |
| `runwise/core.py:68` | list_runs() |
| `runwise/core.py:253` | summarize_run() (with anomaly detection + sparklines) |
| `runwise/core.py:630` | get_history_data() - history as list of dicts |
| `runwise/core.py:694` | get_history() - downsampled history as CSV |
| `runwise/core.py` | get_stability_analysis() - rolling std dev analysis |
| `runwise/core.py` | get_local_stability_analysis() - stability for local JSONL |
| `runwise/sparklines.py:28` | sparkline() - value-to-sparkline conversion |
| `runwise/sparklines.py:77` | trend_indicator() - simple trend arrows |
| `runwise/anomalies.py:42` | detect_anomalies() - main detection function |
| `runwise/anomalies.py:245` | format_anomalies() - output formatting |
| `runwise/tensorboard.py:47` | TensorBoardParser class |
| `runwise/wandb_api.py:51` | WandbAPIClient class |
| `runwise/formatters/markdown.py:26` | MarkdownFormatter class |
| `runwise/config.py:23` | MetricSchema |
| `runwise/config.py:101` | RunwiseConfig |
| `runwise/cli.py:402` | CLI main() |
| `mcp_server/server.py:32` | MCPServer class |
