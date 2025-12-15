# Runwise

**Token-efficient ML training run analysis for AI agents.**

Runwise parses W&B and local training logs, generating condensed summaries optimized for LLM context windows. Designed for collaborative debugging between humans and AI assistants.

## Features

- **Token-efficient output**: Summaries designed to fit in LLM context without wasting tokens
- **Sparkline visualizations**: Unicode trend graphs in ~10 tokens (e.g., `loss: ▇▆▅▃▂▁ ↓`)
- **Anomaly detection**: Automatically flags spikes, overfitting, plateaus, gradient issues
- **W&B integration**: Parse wandb-summary.json, output.log, and run directories
- **TensorBoard support**: Optional parsing of tfevents files
- **Local log support**: JSONL and other common log formats
- **Configurable schemas**: Define metrics relevant to your project
- **MCP server**: Direct integration with Claude Code and other MCP-compatible assistants
- **CLI tool**: Quick analysis from the command line

## Installation

```bash
pip install runwise

# Or from source:
git clone https://github.com/jasegehring/runwise
cd runwise
pip install -e .
```

## Quick Start

### CLI Usage

```bash
# List recent runs with sparkline trends
runwise list
runwise list --no-spark                        # Without sparklines (faster)

# Analyze latest run (includes anomaly detection + sparklines)
runwise latest
runwise latest --no-anomalies                  # Skip anomaly detection
runwise latest --no-spark                      # Skip sparklines

# Analyze specific run
runwise run abc123xyz
runwise run abc123xyz --no-spark --no-anomalies

# Compare two runs
runwise compare run_a run_b
runwise compare run_a run_b -f val             # Filter to validation metrics only
runwise compare run_a run_b -d                 # Show config differences

# Show hyperparameters/config
runwise config                                 # Latest run
runwise config abc123                          # Specific run

# Show run context (name, notes, tags)
runwise notes                                  # Latest run
runwise notes abc123                           # Specific run

# Find best run by metric
runwise best val_loss                          # Lower is better (default)
runwise best accuracy --max                    # Higher is better
runwise best val_loss -n 20                    # Consider last 20 runs

# Get downsampled history (handles million-step runs efficiently)
runwise history                               # Auto-detects common metrics
runwise history -k loss,val_loss              # Specify keys explicitly
runwise history abc123 -k loss -n 100         # Specific run, 100 samples

# Get history statistics (even more compact)
runwise stats                                 # Auto-detects common metrics
runwise stats -k loss,val_loss,grad_norm      # Specify keys explicitly

# List available metric keys
runwise keys                                   # Latest run
runwise keys abc123                            # Specific run

# Live training status
runwise live

# TensorBoard support (requires: pip install tensorboard)
runwise tb                                     # List TB runs
runwise tb -r train_1                          # Summarize specific TB run

# Analyze local log
runwise local training.jsonl
```

### Python API

```python
from runwise import RunAnalyzer, RunwiseConfig

# Auto-detect configuration from project
analyzer = RunAnalyzer()

# List runs
runs = analyzer.list_runs(limit=10)
print(analyzer.format_run_list(runs))

# Analyze latest
run = analyzer.get_latest_run()
print(analyzer.summarize_run(run))

# Compare runs
run_a = analyzer.find_run("abc123")
run_b = analyzer.find_run("def456")
print(analyzer.compare_runs(run_a, run_b))

# Get hyperparameters/config
print(analyzer.get_config(run))

# Find best run by metric
best, ranked = analyzer.find_best_run("val_loss", limit=10, higher_is_better=False)
print(analyzer.format_best_run("val_loss"))

# Get downsampled history (efficient for large runs)
history_csv = analyzer.get_history(run, ["loss", "val_loss"], samples=500)
print(history_csv)

# Get statistics only (most token-efficient)
stats = analyzer.get_history_stats(run, ["loss", "val_loss"])
print(stats)

# List available keys when you don't know what's logged
print(analyzer.list_available_keys(run))
```

### MCP Server (Claude Code Integration)

Add to your Claude Code MCP settings (`~/.claude/settings.json`):

```json
{
    "mcpServers": {
        "runwise": {
            "command": "python",
            "args": ["-m", "runwise.mcp_server"],
            "env": {
                "RUNWISE_PROJECT_ROOT": "/path/to/your/project"
            }
        }
    }
}
```

Then in Claude Code, you can ask:
- "Show me the latest training run"
- "Compare runs abc and def"
- "What's the live training status?"

## Configuration

Create a `runwise.json` in your project root:

```json
{
    "project_name": "My ML Project",
    "wandb_dir": "wandb",
    "logs_dir": "logs",
    "downsample_interval": 1000,
    "schema_inline": {
        "loss_key": "train/loss",
        "step_key": "_step",
        "primary_metric": "train/accuracy",
        "primary_metric_name": "Accuracy",
        "per_step_pattern": "train/loss_step_{i}",
        "num_steps": 8,
        "validation_sets": {
            "val": "Validation",
            "test": "Test"
        },
        "groups": [
            {
                "name": "training",
                "display_name": "TRAINING",
                "metrics": {
                    "train/loss": {"display": "Loss", "format": ".4f", "higher_is_better": false},
                    "train/accuracy": {"display": "Accuracy", "format": ".1%", "higher_is_better": true}
                }
            }
        ]
    }
}
```

Or initialize with defaults:

```bash
runwise init --name "My Project"
```

## Example Output

### Run List (with sparkline trends)
```
RECENT RUNS (My ML Project):

ID           State     Date            Steps     Accuracy        Trend
------------------------------------------------------------------------
xyz789       RUNNING   2025-12-14      5,000       82.3%    ▇▆▅▄▃▂▁▁↓
abc123       FINISHED  2025-12-14     50,000       95.2%    ▇▅▃▂▁▁▁▁↓
def456       CRASHED   2025-12-13      2,341       45.0%    ▁▁▂▅▇▇▇▇↑
```

### Run Summary (with anomaly detection)
```
=== My ML Project Run Summary ===
Run: abc123xyz | Step: 50,000 | Runtime: 12.5h

ANOMALIES:
  ! Overfitting: val/train ratio +35% vs baseline

TRAINING:
  Loss: 0.2341  ▇▆▅▄▃▂▂▁▁▁
  Accuracy: 87.3%  ▁▂▃▄▅▆▇▇▇█

VALIDATION:
  Validation: 85.2%
  Test: 83.7%
```

### Config/Hyperparameters
```
CONFIG: abc123

  batch_size: 64
  dropout: 0.15
  learning_rate: 5.00e-04
  model: transformer
  num_layers: 8
```

### Best Run
```
BEST RUN BY val_loss (from last 10 runs):

  BEST: abc123 = 0.1234

Rank   Run ID       State            val_loss
---------------------------------------------
1      abc123       finished           0.1234 *
2      def456       finished           0.1567
3      ghi789       crashed            0.8901
```

## Why "Token-Efficient"?

When collaborating with AI assistants on ML debugging, you often need to share training logs. Raw W&B exports or verbose logs can consume thousands of tokens, leaving less context for actual analysis.

Runwise generates summaries that:
- **Prioritize actionable metrics**: Loss, accuracy, per-step breakdowns
- **Detect anomalies**: Plateaus, regressions, divergence
- **Use compact formatting**: Tables, aligned columns, minimal whitespace
- **Skip redundant data**: No repeated headers, timestamps, or metadata

### Handling Large Runs

The `history` command implements intelligent downsampling:

```bash
# A 1,000,000 step run returns exactly 500 data points
runwise history -k loss,val_loss -n 500
```

**How it works:**
1. First pass: Count total lines (fast, no parsing)
2. Calculate evenly-spaced sample indices
3. Second pass: Parse only the sampled lines
4. Output as CSV (most token-dense format)

This means a 10GB log file with millions of steps produces ~3000 tokens of output, regardless of size. The LLM never sees the raw file.

## Contributing

Contributions welcome! Areas of interest:
- Additional log format parsers (MLflow, etc.)
- W&B API support for remote runs
- GitHub Action for PR comments
- Integration with other AI assistants

## License

MIT
