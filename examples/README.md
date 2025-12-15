# Runwise Examples

This folder contains example files to help you understand Runwise's input formats and expected outputs.

## Files

### `training_log.jsonl`
A sample JSONL training log file showing the expected format for local log analysis. Each line is a JSON object with metrics:

```json
{"_step": 100, "loss": 1.8234, "train/accuracy": 0.35, "val/loss": 1.9123, "lr": 0.001}
```

**Key fields:**
- `_step` or `step`: Training step number
- Metric keys (e.g., `loss`, `train/loss`, `val/accuracy`)
- Can include any numeric metrics

**Usage:**
```bash
runwise local training_log.jsonl
```

### `runwise_config_example.json`
Example configuration file showing how to customize Runwise for your project. Copy this to your project root as `runwise.json`.

**Key sections:**
- `project_name`: Display name for your project
- `wandb_dir`: Path to W&B local directory (default: `wandb`)
- `logs_dir`: Path to local log files (default: `logs`)
- `schema_inline`: Metric configuration
  - `loss_key`: Primary loss metric name
  - `step_key`: Step counter key
  - `primary_metric`: Main metric for run lists
  - `groups`: Metric groupings for summaries
  - `validation_sets`: Validation set prefixes

**Usage:**
```bash
# Copy to project root
cp runwise_config_example.json /path/to/project/runwise.json
# Edit to match your metrics
```

### `expected_outputs.md`
Reference document showing example outputs from various Runwise commands. Useful for understanding what each command produces.

## Quick Start

1. **For W&B projects:**
   ```bash
   cd /path/to/your/project  # Must have wandb/ directory
   runwise list              # List recent runs
   runwise latest            # Summarize latest run
   ```

2. **For local JSONL logs:**
   ```bash
   runwise local path/to/training.jsonl
   ```

3. **For W&B cloud (no local files):**
   ```bash
   pip install wandb  # If not already installed
   runwise api --project my-project --entity my-team
   ```

4. **For TensorBoard:**
   ```bash
   pip install tensorboard  # If not already installed
   runwise tb --log-dir ./runs
   ```

## Log Format Tips

### JSONL (JSON Lines)
Each line is a complete JSON object. Metrics are logged as key-value pairs.

```json
{"_step": 0, "train/loss": 2.45, "train/accuracy": 0.12}
{"_step": 100, "train/loss": 1.82, "train/accuracy": 0.35}
```

### Common Metric Naming Conventions

| Convention | Example |
|------------|---------|
| Slash-separated | `train/loss`, `val/accuracy` |
| Underscore-separated | `train_loss`, `val_accuracy` |
| Simple | `loss`, `accuracy` |

Runwise auto-detects common metric patterns like `loss`, `val_loss`, `accuracy`, `lr`, etc.

### W&B Local Files

When using W&B, Runwise reads from these files:
- `wandb/run-*/files/wandb-summary.json` - Final metrics
- `wandb/run-*/files/wandb-history.jsonl` - Full training history
- `wandb/run-*/files/config.yaml` - Hyperparameters
- `wandb/run-*/files/wandb-metadata.json` - Run context (name, notes, tags)
