# Runwise Development Roadmap

**Vision**: The go-to tool for AI-assisted ML training analysis. Token-efficient, extensible, and seamlessly integrated with AI coding assistants.

**Current Version**: 0.4.0

---

## Completed Features

### v0.1.0 - MVP
- [x] W&B run discovery and parsing
- [x] Local JSONL log parsing
- [x] Configurable metric schemas
- [x] Token-efficient summary generation
- [x] CLI interface
- [x] MCP server (basic)

### v0.2.x - Anomaly Detection & Stability
- [x] **Anomaly detection**: Loss spikes (MAD-based), overfitting, plateaus, gradient issues, NaN/Inf
- [x] **Sparkline visualizations**: Unicode trend graphs in ~10 tokens
- [x] **Stability analysis**: Rolling std dev with `runwise stability`
- [x] **Unit tests**: 94 tests covering core functionality
- [x] Configurable anomaly thresholds

### v0.3.x - Multi-Platform & Export
- [x] **TensorBoard support**: Parse tfevents files (requires `tensorboard`)
- [x] **W&B API support**: Remote run access (requires `wandb`)
- [x] **Markdown export**: `--format md` for GitHub/Notion
- [x] **Examples directory**: Sample configs and expected outputs

### v0.4.0 - Usability & Filtering (Current)
- [x] **Step-matched comparison**: `run1@50000` syntax for curriculum learning
- [x] **Custom keys for summaries**: `-k loss,val_loss` for focused output
- [x] **Comparison filtering**: `-f val`, `-t 5` (threshold %), `-g` (group by prefix)
- [x] **API history**: `--history -k` for metric trajectories
- [x] **API best run**: `--best metric --max` to find best runs
- [x] **Project auto-detection**: Reads from `wandb/` metadata
- [x] **Dynamic column widths**: No more truncated metric names
- [x] **MCP health_check tool**: Quick "how's training going?" check
- [x] **MCP JSON-RPC fix**: Proper response format
- [x] **Improved MCP descriptions**: Emphasizes list_keys-first pattern

---

## Upcoming Features

### v0.4.1 - Schema & Usability (Next)
- [ ] Schema auto-generation from W&B run (`runwise init --from-run`)
- [ ] Better error handling with user-friendly messages
- [ ] `--verbose` and `--quiet` flags
- [ ] Built-in schema templates for common frameworks

### v0.5.x - Smart Analysis
- [ ] **Executive summary**: 2-3 sentence plain-English overview
- [ ] **Interactive queries**: "Show runs where val_acc > 80%"
- [ ] **Recommendations**: Actionable suggestions based on metrics
- [ ] **Learning rate analysis**: Detect LR issues

### v0.6.x - Collaboration
- [ ] **Slack notifications**: Post summaries to Slack
- [ ] **GitHub integration**: Link runs to commits/PRs
- [ ] **HTML reports**: Shareable standalone reports

---

## Lower Priority / Ideas

### Additional Platforms
- [ ] **MLflow support**: Connect to MLflow tracking server
- [ ] **Aim**: Parse Aim repositories
- [ ] **Neptune**: Neptune.ai support

### Advanced Features
- [ ] **Watch mode**: Monitor training and alert on events
- [ ] **ASCII charts**: Terminal plots for training curves
- [ ] **Confidence intervals**: Uncertainty on metrics
- [ ] **Cost estimation**: Estimate cloud compute costs

---

## Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Token efficiency | <500 tokens for run summary | Achieved |
| Parse time | <1s for 100k step run | Achieved |
| Platform coverage | 3+ tracking systems | W&B, TensorBoard, local JSONL |
| Zero dependencies | Core package | Achieved (optional: wandb, tensorboard) |
| Test coverage | 90+ tests | 94 tests |

---

## Contributing

Issues and PRs welcome for:
- New platform parsers (MLflow, Aim, Neptune)
- Schema templates for common frameworks
- Analysis heuristics and anomaly detectors
- Documentation improvements

---

**Last Updated**: December 15, 2024
