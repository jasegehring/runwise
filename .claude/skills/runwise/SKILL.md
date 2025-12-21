---
name: runwise
description: Analyze ML training runs from W&B or local logs. Use when user asks about training progress, metrics, run comparison, anomalies, or "how is training going". Provides token-efficient summaries with sparklines and anomaly detection.
allowed-tools: Bash, Read, Glob
---

# Runwise - ML Training Run Analysis

Token-efficient analysis of ML training runs. Uses the "Zoom-In Strategy": start broad, drill down only when needed.

## Quick Start

**"How is training going?"** - Start here:
```bash
runwise latest
```

**List recent runs:**
```bash
runwise list
```

**Compare two runs:**
```bash
runwise compare <run_a> <run_b>
```

## Zoom-In Strategy (Token Efficiency)

Always start with the cheapest view and drill down:

| Level | Command | Token Cost | Use When |
|-------|---------|------------|----------|
| 1. Status | `runwise list` | ~200 | Overview of all runs |
| 2. Summary | `runwise latest` or `runwise run <ID>` | ~500 | Details on one run |
| 3. Metrics | `runwise history -k loss,val_loss` | ~1000 | Need actual values |
| 4. Raw | Read the wandb files directly | High | Last resort |

## Command Reference

### Discovery
```bash
runwise list                    # List runs with sparkline trends
runwise keys [ID]               # List available metric keys (CALL FIRST before history)
runwise live                    # Live training status from output.log
```

### Analysis
```bash
runwise latest                  # Analyze latest run (anomalies + sparklines)
runwise latest -k loss,val_loss # Filter to specific metrics
runwise run <ID>                # Analyze specific run
runwise run <ID> --no-anomalies # Skip anomaly detection
```

### Comparison
```bash
runwise compare <A> <B>              # Basic comparison
runwise compare <A> <B> -f val       # Filter to validation metrics
runwise compare <A> <B> -t 5         # Only metrics with >5% delta
runwise compare <A> <B> -g           # Group by prefix (train/, val/)
runwise compare run1@50000 run2@50000  # Step-matched comparison
```

### History & Stats
```bash
runwise keys                         # ALWAYS call first to discover metrics
runwise history -k loss,val_loss     # Downsampled CSV (default 500 samples)
runwise stats -k loss,val_loss       # Min/max/mean/final statistics
runwise stability -k loss,val_loss   # Training stability (rolling std dev)
```

### Local JSONL Logs
```bash
runwise local                        # List local log files
runwise local <file> --keys          # List available keys
runwise local <file> --history -k loss,val_loss
runwise local <file> --stats -k loss,val_loss
```

### Markdown Export
Add `--format md` to any command for GitHub-flavored markdown:
```bash
runwise list --format md
runwise compare <A> <B> --format md
```

## Anomaly Detection

Runwise automatically detects:
- **Loss spikes** - Sudden increases (MAD-based detection)
- **Overfitting** - Val loss diverging from train loss
- **Plateaus** - Training stalled
- **Gradient issues** - NaN/Inf values
- **Throughput drops** - Training slowdowns

Anomalies are only reported when detected (zero tokens if healthy).

## Common Workflows

### "How is my training going?"
```bash
runwise latest
```
Returns: status, key metrics with sparklines, any anomalies.

### "Compare my last two runs"
```bash
runwise list                    # Get run IDs
runwise compare <latest> <previous> -t 5  # Show >5% differences
```

### "Is my model overfitting?"
```bash
runwise latest -k loss,val_loss
```
Look for: val_loss trend diverging from loss, overfitting anomaly warning.

### "What metrics are available?"
```bash
runwise keys
```

### "Show me the loss curve"
```bash
runwise history -k loss --samples 100
```
Returns CSV that can be analyzed or plotted.

### "Compare at the same training step"
```bash
runwise compare run1@50000 run2@50000
```
Essential for curriculum learning or different training speeds.

## Output Interpretation

### Sparklines
```
loss:     ▆▅▄▃▂▂▁▁  0.892 → 0.234  # Decreasing (good for loss)
val_acc:  ▁▂▃▄▅▆▇█  0.65 → 0.94   # Increasing (good for accuracy)
```

### Trend Indicators
- `↓` Decreasing
- `↑` Increasing
- `→` Stable
- `~` Noisy/unclear

### Status
- `running` - Currently training
- `finished` - Completed successfully
- `crashed` - Terminated with error
- `failed` - Failed validation

## Tips

1. **Always run `runwise keys` first** before requesting specific metrics
2. **Use `-k` flag** to filter output and save tokens
3. **Step-matched comparison** (`@step`) is essential when comparing runs with different training lengths
4. **Check for anomalies** before diving into raw data - they surface issues automatically
5. **Use `--format md`** when you need to include output in documentation
