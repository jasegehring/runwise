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
│   ├── __init__.py       # Exports: RunAnalyzer, RunwiseConfig, MetricSchema
│   ├── cli.py            # CLI entry point (runwise command)
│   ├── config.py         # Configuration and metric schema definitions
│   ├── core.py           # RunAnalyzer - main analysis logic
│   ├── formatters/       # Output formatters (placeholder for expansion)
│   └── parsers/          # Log parsers (placeholder for expansion)
├── mcp_server/
│   └── server.py         # MCP server for Claude Code integration
├── examples/
│   └── peptrm_config.json # Example configuration
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
- `summarize_run(run)` - Generate token-efficient summary
- `compare_runs(run_a, run_b)` - Side-by-side metric comparison
- `get_live_status()` - Parse output.log for live training status
- Local log methods for JSONL files

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
runwise list                          # List recent runs
runwise latest                        # Summarize latest run
runwise run <ID>                      # Summarize specific run
runwise compare <A> <B>               # Compare two runs
runwise history -k loss,val_loss      # Get downsampled history (CSV)
runwise history <ID> -k loss -n 100   # Specific run, 100 samples
runwise stats -k loss,val_loss        # Get history statistics only
runwise keys [ID]                     # List available metric keys
runwise live                          # Show live training status
runwise local [file]                  # List/analyze local logs
runwise init [--name]                 # Initialize runwise.json
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

### Downsampling for Large Files
The `get_history()` method handles arbitrarily large files efficiently:
1. **Two-pass approach**: First counts lines (fast), then reads only sampled lines
2. **Never loads full file**: Memory usage stays constant regardless of file size
3. **CSV output**: Most token-dense format for tabular data
4. **Configurable samples**: Default 500, adjustable via `--samples` flag

A 10GB history file with 10 million steps produces the same ~3000 token output as a 1MB file with 1000 steps.

## Areas for Future Development

### High Priority (Spec Alignment)
1. ~~**Server-side downsampling**~~ - DONE: Implemented local downsampling in `get_history()`
2. ~~**CSV output mode**~~ - DONE: `history` command outputs CSV
3. ~~**Available keys hint**~~ - DONE: `list_available_keys()` and shown when keys not found
4. **Optional W&B API support** - For remote run access (not just local files)

### Medium Priority (Roadmap v0.1.x)
1. Add unit tests (pytest)
2. Improve error handling with user-friendly messages
3. Auto-detect common metric patterns
4. Add `--verbose` and `--quiet` flags
5. Schema auto-generation from W&B run

### Lower Priority (Roadmap v0.2+)
1. Anomaly detection (plateaus, divergence, overfitting)
2. TensorBoard/MLflow support
3. ASCII charts for training curves
4. Run tagging and notes

## MCP Server

The MCP server (`mcp_server/server.py`) exposes these tools:
- `list_runs` - List recent training runs
- `analyze_run` - Detailed analysis of specific run
- `analyze_latest` - Analyze latest/active run
- `compare_runs` - Compare two runs
- `live_status` - Live training status
- `analyze_local_log` - Analyze local log file

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

## Testing

Currently no tests. When adding tests:
- Use pytest
- Mock W&B directory structure with sample data
- Test metric parsing, formatting, schema loading
- Test CLI commands

## Publishing Checklist

Before PyPI release:
- [ ] Update URLs in pyproject.toml (replace "yourusername")
- [ ] Add LICENSE file
- [ ] Verify all imports work after pip install
- [ ] Test CLI entry point
- [ ] Consider adding py.typed marker

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
| `runwise/core.py:35` | RunAnalyzer class |
| `runwise/core.py:54` | list_runs() |
| `runwise/core.py:121` | summarize_run() |
| `runwise/core.py:225` | compare_runs() |
| `runwise/core.py:266` | get_history() - downsampled history |
| `runwise/core.py:294` | _downsample_jsonl() - efficient sampling |
| `runwise/core.py:434` | get_history_stats() - stats only |
| `runwise/core.py:498` | list_available_keys() |
| `runwise/config.py:23` | MetricSchema |
| `runwise/config.py:101` | RunwiseConfig |
| `runwise/cli.py:164` | CLI main() |
| `mcp_server/server.py:32` | MCPServer class |
