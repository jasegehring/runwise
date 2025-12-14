# Runwise Development Roadmap

**Vision**: The go-to tool for AI-assisted ML training analysis. Token-efficient, extensible, and seamlessly integrated with AI coding assistants.

**Current Version**: 0.1.0 (MVP)

---

## Phase 1: Core Stability (v0.1.x)

**Goal**: Solid foundation that works reliably for W&B + local logs

### v0.1.0 - MVP (Current)
- [x] W&B run discovery and parsing
- [x] Local JSONL log parsing
- [x] Configurable metric schemas
- [x] Token-efficient summary generation
- [x] CLI interface
- [x] MCP server (basic)

### v0.1.1 - Polish
- [ ] Better error handling and user-friendly messages
- [ ] Auto-detect common metric patterns (loss, accuracy, etc.)
- [ ] Handle missing/corrupted log files gracefully
- [ ] Add `--verbose` and `--quiet` flags
- [ ] Unit tests for core functionality

### v0.1.2 - Schema Improvements
- [ ] Schema validation with helpful error messages
- [ ] Schema auto-generation from W&B run (inspect metrics, suggest schema)
- [ ] Built-in schema templates for common frameworks:
  - [ ] PyTorch Lightning
  - [ ] Hugging Face Transformers
  - [ ] Generic classification/regression

---

## Phase 2: Analysis Intelligence (v0.2.x)

**Goal**: Move from passive reporting to active insights

### v0.2.0 - Anomaly Detection
- [ ] **Plateau detection**: Identify when metrics stop improving
  - Configurable window size and threshold
  - Per-metric sensitivity
- [ ] **Regression alerts**: Flag when metrics get worse
- [ ] **Divergence detection**: Catch exploding/vanishing gradients
- [ ] **Training health score**: Single 0-100 score summarizing run health

### v0.2.1 - Trend Analysis
- [ ] **Learning rate finder integration**: Detect LR issues
- [ ] **Convergence estimation**: "At current rate, will reach X% in Y steps"
- [ ] **Comparison insights**: "Run B is 15% faster to converge than A"
- [ ] **Curriculum stage analysis**: Track performance across training phases

### v0.2.2 - Smart Summaries
- [ ] **Executive summary**: 2-3 sentence overview for quick status
- [ ] **Recommendations**: Actionable suggestions based on metrics
  - "Loss plateaued at step 20k, consider reducing LR"
  - "Val accuracy diverging from train - possible overfitting"
- [ ] **Key moments**: Automatically identify important events
  - Best checkpoint
  - Curriculum transitions
  - Anomalies

---

## Phase 3: Multi-Platform Support (v0.3.x)

**Goal**: Work with any ML tracking system

### v0.3.0 - TensorBoard Support
- [ ] Parse TensorBoard event files
- [ ] Handle TensorBoard directory structures
- [ ] Map TensorBoard scalars to Runwise schema

### v0.3.1 - MLflow Support
- [ ] Connect to MLflow tracking server
- [ ] Parse MLflow run artifacts
- [ ] Support MLflow experiments/runs hierarchy

### v0.3.2 - Additional Platforms
- [ ] **Aim**: Parse Aim repositories
- [ ] **ClearML**: ClearML task integration
- [ ] **Neptune**: Neptune.ai support
- [ ] **Custom**: Plugin architecture for custom backends

---

## Phase 4: Enhanced MCP Integration (v0.4.x)

**Goal**: Best-in-class AI assistant integration

### v0.4.0 - Rich MCP Tools
- [ ] **Streaming support**: Stream live training updates
- [ ] **Interactive queries**: "Show me runs where val_acc > 80%"
- [ ] **Chart generation**: ASCII charts for training curves
- [ ] **Diff tool**: Detailed config/hyperparameter diff between runs

### v0.4.1 - Context-Aware Analysis
- [ ] **Project memory**: Remember previous analyses in session
- [ ] **Run tagging**: Tag runs with notes via MCP
- [ ] **Favorites**: Mark important runs for quick access
- [ ] **Run groups**: Analyze groups of related runs together

### v0.4.2 - Proactive Notifications
- [ ] **Watch mode**: Monitor training and alert on events
- [ ] **Threshold alerts**: "Notify when val_acc > 90%"
- [ ] **Failure detection**: Alert on NaN, crash, stall

---

## Phase 5: Collaboration Features (v0.5.x)

**Goal**: Team-friendly features for shared ML projects

### v0.5.0 - Notes & Annotations
- [ ] **Run notes**: Add/edit notes on runs (stored locally or in W&B)
- [ ] **Hypothesis tracking**: Link runs to hypotheses being tested
- [ ] **Decision log**: Record why certain runs were continued/stopped

### v0.5.1 - Reports
- [ ] **Markdown report generation**: Export analysis as .md
- [ ] **HTML reports**: Shareable standalone HTML reports
- [ ] **Comparison tables**: Multi-run comparison exports

### v0.5.2 - Integration
- [ ] **Slack notifications**: Post summaries to Slack
- [ ] **GitHub integration**: Link runs to commits/PRs
- [ ] **Notion/Obsidian**: Export to knowledge bases

---

## Phase 6: Advanced Analysis (v0.6.x)

**Goal**: Deep insights for power users

### v0.6.0 - Statistical Analysis
- [ ] **Confidence intervals**: Uncertainty on metrics
- [ ] **Significance testing**: "Is run B significantly better than A?"
- [ ] **Hyperparameter importance**: Which HPs matter most?

### v0.6.1 - Visualization
- [ ] **Terminal plots**: Rich ASCII/Unicode charts
- [ ] **Export to matplotlib**: Generate publication-ready figures
- [ ] **Interactive HTML**: Zoomable, hoverable charts

### v0.6.2 - Predictive Features
- [ ] **Training time estimation**: Predict total training time
- [ ] **Resource usage**: GPU memory, compute trends
- [ ] **Cost estimation**: Estimate cloud compute costs

---

## Technical Debt & Infrastructure

### Testing
- [ ] Unit tests (pytest)
- [ ] Integration tests with mock W&B data
- [ ] MCP server contract tests
- [ ] CI/CD pipeline (GitHub Actions)

### Documentation
- [ ] API documentation (Sphinx/MkDocs)
- [ ] Tutorial: "Analyzing your first run"
- [ ] Tutorial: "Custom schemas for your project"
- [ ] Tutorial: "MCP setup with Claude Code"
- [ ] Video walkthrough

### Performance
- [ ] Lazy loading for large log files
- [ ] Caching for repeated queries
- [ ] Async I/O for MCP server
- [ ] Memory-efficient streaming for huge runs

### Distribution
- [ ] PyPI package
- [ ] Homebrew formula (macOS)
- [ ] Docker image
- [ ] Pre-built binaries (optional)

---

## Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Token efficiency | <500 tokens for run summary | Fit in LLM context |
| Parse time | <1s for 100k step run | Interactive feel |
| Platform coverage | 3+ tracking systems | Broad adoption |
| Zero dependencies | Core package | Easy install |

---

## Community & Governance

### Contributing
- Issues welcome for bug reports and feature requests
- PRs welcome for:
  - New platform parsers
  - Schema templates
  - Analysis heuristics
  - Documentation

### Versioning
- Semantic versioning (MAJOR.MINOR.PATCH)
- Breaking changes only in MAJOR versions
- Schema format stability guaranteed within MINOR versions

---

## Ideas Parking Lot

*Features that might be valuable but need more thought:*

- **Natural language queries**: "Show runs from last week with best accuracy"
- **Auto-labeling**: ML-based run categorization
- **Failure post-mortems**: Automated root cause analysis
- **A/B test analysis**: Statistical comparison framework
- **Distributed training support**: Multi-node run aggregation
- **Model card generation**: Auto-generate model documentation
- **Reproducibility checker**: Verify run can be reproduced

---

## Immediate Next Steps (Post-Extraction)

1. **Test in pepTRM**: Validate MCP integration works
2. **Fix any bugs**: Address issues found in real usage
3. **Add pepTRM schema as example**: Document real-world config
4. **v0.1.1 release**: Polish based on feedback
5. **Announce**: Share with community for feedback

---

**Last Updated**: December 14, 2024
**Maintainer**: TBD (after extraction)
