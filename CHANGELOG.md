# Changelog

All notable changes to Runwise will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2024-12-21

### Added
- **`runwise sync`**: Recover missing wandb-history.jsonl files
  - `runwise sync` lists runs needing sync
  - `runwise sync <run_id>` syncs specific run
  - `runwise sync --all` syncs all unsynced runs
- **`runwise find`**: Search runs by name, tag, group, or notes
  - `runwise find "R008"` searches across all metadata
- **`runwise watch`**: Live tail of training output with color highlighting
  - Validation lines highlighted in cyan
  - Errors in red, warnings in yellow
- **Validation parsing in `runwise live`**: Now extracts structured validation metrics
- **Run names in list**: `runwise list` shows Name column when runs have names
- **Auto-detection of metric keys**: Summaries auto-detect train/loss, train/accuracy patterns

### Changed
- All "no history file" errors now include sync instructions
- Improved error messages with actionable recovery steps

## [0.5.0] - 2024-12-21

### Added
- **Claude Code Skill**: Token-efficient alternative to MCP for Claude Code users
  - Auto-discovered when users ask about training progress
  - Invokes CLI commands via Bash (~500 tokens vs ~2000+ for MCP)
  - Includes command reference and workflow examples
  - Located in `.claude/skills/runwise/`

### Changed
- README now recommends Skill for Claude Code users, MCP for other integrations
- Updated CLAUDE.md with skill documentation and usage guide

## [0.4.0] - 2024-12-15

### Added
- **Step-matched comparison**: Compare runs at specific training steps using `@step` syntax
  - `runwise compare run1@50000 run2@50000` - Compare at same step
  - Essential for curriculum learning where final metrics are misleading
- **Custom keys for summaries**: `-k/--keys` flag to show only specific metrics
  - `runwise latest -k loss,val_loss,accuracy`
  - Works with `latest`, `run`, and API commands
- **Comparison filtering options**:
  - `-g/--group`: Group metrics by prefix (train/, val/, etc.)
  - `-t/--threshold`: Only show metrics with delta > N%
  - Dynamic column widths (no more truncated metric names)
  - Step mismatch warnings when comparing runs at different steps
- **W&B API enhancements**:
  - `--history -k`: Get metric trajectories from cloud
  - `--best metric --max`: Find best run by any metric
  - Project auto-detection from `wandb/` directory metadata
  - Comparison filtering (`-f`, `-t`, `-g`, `-d`) for API comparisons
- **MCP Server improvements**:
  - New `health_check` tool: Quick "how's training going?" combining status, metrics, sparklines, and anomaly detection
  - Fixed JSON-RPC response format (proper `{"result": ...}` wrapping)
  - Improved tool descriptions emphasizing "call list_keys first" pattern
  - Step-matched comparison support in `compare_runs` tool

### Changed
- Comparison output now uses dynamic column widths (up to 45 chars) instead of fixed 25-char truncation
- MCP server version now matches package version

## [0.3.1] - 2024-12-14

### Fixed
- Minor bug fixes and stability improvements

## [0.3.0] - 2024-12-14

### Added
- **Markdown export**: `--format md` flag for GitHub issues, Notion, documentation
- **W&B API support**: Access remote runs without local sync (requires `wandb` package)
- **Examples directory**: Sample configs and expected outputs
- Stability analysis with `runwise stability` command

## [0.2.3] - 2024-12-14

### Added
- Configurable anomaly detection thresholds in `runwise.json`

## [0.2.0] - 2024-12-14

### Added
- **Anomaly detection**: Loss spikes (MAD-based), overfitting, plateaus, gradient issues, NaN/Inf
- **Sparkline visualizations**: Unicode trend graphs in ~10 tokens
- **TensorBoard support**: Parse tfevents files (requires `tensorboard` package)
- 94 unit tests covering core functionality

## [0.1.0] - 2024-12-13

### Added
- Initial release
- W&B run discovery and parsing from local `wandb/` directory
- Local JSONL log parsing
- Configurable metric schemas
- Token-efficient summary generation
- CLI interface with commands: list, latest, run, compare, config, notes, history, stats, keys, live, best
- MCP server for Claude Code integration
- Downsampling for efficient handling of large runs
