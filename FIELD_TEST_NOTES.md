# Runwise Field Test Notes
**Date**: 2024-12-21
**Test Project**: pepTRM (peptide sequencing with recursive transformers)
**Runs Tested**: R004 (3be9af2m, finished), R008A (p8ceeyc1, running at ~86k steps)

## Status: RESOLVED (v0.5.1)
Most issues from this field test have been addressed. See below for details.

---

## What Worked Well

### 1. Run Discovery
- `runwise list` correctly enumerated all W&B runs
- `runwise live` correctly extracted current step/loss/accuracy from output.log of running run
- Run ID lookup worked correctly once I knew the W&B IDs

### 2. Key Discovery
- `runwise keys <run_id>` correctly listed all 60+ metrics from the summary file
- This is essential for complex projects with custom metric names

### 3. Custom Key Filtering
- `-k` flag worked well: `runwise run 3be9af2m -k train/token_accuracy,val_nine_species/token_accuracy`
- Showed exactly the metrics I asked for

### 4. Summary from wandb-summary.json
- Final values were correctly extracted
- Runtime and step counts accurate

### 5. Config Extraction
- `runwise config <run_id>` correctly extracted all hyperparameters from config.yaml
- Clean, alphabetized output

### 6. Best Run Finder
- `runwise best val_proteometools/token_accuracy --max` worked perfectly
- Found R003 (7xoyc9m1) as best with 24.4% token accuracy
- Correctly ranked 7 runs with the metric

### 7. Markdown Export
- `--format md` produced clean GitHub-flavored markdown
- Included collapsible hyperparameters section
- Good for pasting into issues/docs

---

## Issues Found (Bugs/Limitations)

### 1. **CRITICAL: Missing wandb-history.jsonl**
Both runs lacked the history file, causing multiple features to fail:
- `runwise history` → "No metrics matching X found in output.log"
- `runwise stability` → "No history file for run"
- `runwise compare @step` → "no common metrics found"
- No sparklines in run summaries

**Root cause**: wandb.finish() wasn't called or run was killed. This is documented in README but the error messages don't clearly explain the fix.

**Suggestion**:
- Better error message: "No wandb-history.jsonl found. Run `wandb sync <run_dir>` to recover history data."
- Add `runwise sync <run_id>` command that calls wandb sync

### 2. **Running run has no summary**
`runwise run p8ceeyc1` showed all metrics as "(not found)" because wandb-summary.json doesn't exist until run finishes.

**Current workaround**: `runwise live` parses output.log for basic metrics.

**Suggestion**: Fall back to parsing output.log when summary is missing, or parse the W&B internal files (there are .wandb files with real-time data).

### 3. **`runwise list` shows W&B IDs, not run names**
The list shows `p8ceeyc1` but user knows runs as "R004", "R008A". Had to manually look up the mapping in project docs.

**Suggestion**:
- Show run name in list if set: `runwise list` → shows `Name` column
- Allow search by name: `runwise find "R008"` or `runwise run R008A` fuzzy matching

### 4. **No sparklines without history**
Even though we have summary data (final values), sparklines require history.

**Suggestion**: For finished runs without history, show a single-point indicator instead of full sparkline: `train/loss: 1.96 (no trend data)`

### 5. **output.log has rich data that's not parsed**
The log contains validation metrics on real datasets:
```
Val (ProteomeTools) | Token: 14.0% | Gain: +0.6% | Pos: N=7% M=9% C=47% | Best AA: K (62%)
Val (Nine-Species) | Token: 7.6% | Gain: -0.5% | Pos: N=5% M=8% C=9% | Best AA: G (42%)
```

But `runwise live` only extracts: Step, Loss, Accuracy

**Suggestion**: Parse structured validation lines from output.log, not just training lines.

### 6. **Default schema doesn't match project**
`runwise run <id>` without `-k` showed sparse output because the default schema (loss, accuracy) didn't match actual keys (train/loss, train/token_accuracy).

**Suggestion**: Auto-detect key patterns from summary (e.g., if `train/loss` exists, use that instead of `loss`).

---

## Feature Wishes

### 1. **Run name/tag search**
```bash
runwise find "R008"              # Find runs matching name pattern
runwise run R008A                # Resolve name to W&B ID automatically
runwise list --filter "tag:experiment"
```

### 2. **Parse output.log as fallback**
When wandb-history.jsonl is missing, parse output.log for history:
```bash
runwise history --from-log       # Parse output.log instead of history.jsonl
```

### 3. **Real-time streaming for running runs**
```bash
runwise watch <run_id>           # Tail output.log with live metrics
runwise watch --interval 30      # Refresh every 30s
```

### 4. **Sync helper**
```bash
runwise sync <run_id>            # Calls wandb sync on the run directory
runwise sync --all               # Sync all unsynced runs
```

### 5. **Project-specific schema auto-detection**
Scan first run's keys and auto-generate schema:
```bash
runwise init --auto              # Auto-detect schema from existing runs
```

### 6. **Comparison at matching curriculum stage**
For curriculum learning, comparing at same step isn't ideal - compare at same curriculum stage:
```bash
runwise compare run1@stage:3 run2@stage:3
```

### 7. **Export comparison to CSV/JSON**
```bash
runwise compare run1 run2 --format csv > comparison.csv
runwise compare run1 run2 --format json
```

### 8. **Multi-run aggregation**
```bash
runwise aggregate R001,R002,R003 -k val/accuracy  # Mean/std across runs
runwise best val/accuracy --top 5                  # Show top 5 runs
```

### 9. **Anomaly detection on output.log**
Since history is often missing, run anomaly detection on parsed output.log:
```bash
runwise anomalies --from-log
```

### 10. **Better "no data" messages**
Instead of generic "not found", explain:
- Why data is missing (no history file, run still active, key doesn't exist)
- How to fix it (sync, wait for run to finish, check key names)

---

## Summary

**Token efficiency**: Good when data exists. The `-k` filtering and compact output work well.

**Main blocker**: Missing wandb-history.jsonl files. Without history, most advanced features (sparklines, trends, comparison, stability) fail. This is a W&B artifact issue, not runwise's fault, but runwise could:
1. Better explain the issue and fix
2. Fall back to parsing output.log
3. Provide sync helpers

**Skill vs MCP**: For this test, the skill's CLI-based approach worked fine. The main issue was data availability, not the interface.

**Priority fixes**:
1. Better error messages with sync instructions
2. Parse output.log as history fallback
3. Auto-detect metric key patterns
4. Show run names in list

---

## Resolutions (v0.5.1)

### Implemented
| Issue | Solution |
|-------|----------|
| Missing wandb-history.jsonl | Added `runwise sync` command to recover files |
| Error messages unclear | All "no history file" errors now include `runwise sync <id>` hint |
| output.log not parsed | `runwise live` now parses validation lines with structured output |
| No run name search | Added `runwise find "pattern"` to search names, tags, notes |
| No run names in list | `runwise list` now shows Name column when runs have names set |
| No live tailing | Added `runwise watch` with color highlighting |
| Key auto-detection | `summarize_run` and `format_run_list` now auto-detect metric keys |

### Partially Addressed
| Issue | Status |
|-------|--------|
| output.log as history fallback | Works but key names must match log format (e.g., "loss" not "train/loss") |

### Not Yet Implemented (Future Work)
| Feature | Notes |
|---------|-------|
| Multi-run aggregation | `runwise aggregate` command |
| Anomaly detection on output.log | Would require different detection logic |
| Comparison at curriculum stage | Complex to implement reliably |
