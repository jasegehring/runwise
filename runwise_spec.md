Specification: Autonomous Deep Learning Analysis Tooling
1. Executive Summary & Philosophy
Objective: Enable the AI agent to independently diagnose deep learning training runs, detect failure modes (divergence, overfitting), and compare experiments without human intervention.

Core Philosophy: "The Zoom-In Strategy" Deep learning logs are massive. Ingesting raw logs directly will saturate the context window and incur prohibitive token costs. Therefore, the agent must implement a hierarchical retrieval strategy:

Macro View: List runs and status (low cost).

Summary View: Single-point metrics like "best validation loss" (low cost).

Sampled View: Downsampled history curves (fixed token cost).

Micro View: Only fetch raw data for specific steps/keys when absolutely necessary.

Constraint: The solution must strictly limit token output. It should never dump full JSON logs into the console.

2. Technology Stack
Backend: Weights & Biases (W&B)

Interface: wandb Python API (wandb.Api())

Language: Python 3.x

Data Handling: Pandas (for efficient CSV serialization)

3. Functional Requirements
The agent is to implement a CLI-based utility script (suggested name: wandb_agent.py) that exposes the following specific capabilities via command-line arguments.

A. Run Enumeration (list_runs)
Goal: Allow the agent to orient itself and find relevant run IDs.

Input: Project name, limit (default 10).

Output: A minimalist JSON or Table containing Run ID, Run Name, State (finished/failed/running), and Created At.

Filtering: Must filter out "crashed" or "short" runs unless requested otherwise.

B. High-Level Summary (get_summary)
Goal: Quick diagnostic check of hyperparameters and final results.

Input: run_id.

Output: JSON object containing:

Config: Hyperparameters (learning rate, batch size, architecture).

Summary Metrics: Best values for tracked metrics (e.g., best_val_loss, accuracy).

Constraint: Must filter out internal W&B keys (keys starting with _ or wandb).

C. Token-Optimized History (get_history)
Goal: Analyze trends (loss curves, spikes) without reading the whole file.

Input: run_id, keys (list of metrics to fetch, e.g., "loss,val_loss"), samples (default 500).

Critical Implementation Detail:

Must use the run.history(samples=N) feature of the W&B API. This forces W&B to interpolate the data server-side and return exactly N rows, regardless of whether the run lasted 10 steps or 10 million steps.

Output: A CSV-formatted string. CSV is preferred over JSON here because it uses significantly fewer tokens (no repeated keys).

D. Comparison (compare_runs)
Goal: Compare a target run against a baseline.

Input: run_id_1, run_id_2, metric (e.g., "val_loss").

Output: A brief text summary or small table showing the delta in the summary metric and a correlation calculation if possible.

4. Implementation Guidelines for Claude
Authentication
The tool should assume WANDB_API_KEY is present in the environment or that the machine is already logged in via wandb login.

It should handle wandb.errors.CommError gracefully (e.g., network issues).

Data Serialization
Do not print pandas DataFrames using standard pretty-printing (which creates whitespace bloat).

Do print using df.to_csv(index=False) for maximum token density.

Error Handling
If a requested run_id does not exist, return a clear error message "Run [ID] not found" rather than a stack trace.

If a requested metric key is missing from the run, list the available keys to help the agent self-correct.

5. Agent Workflow Protocols
Define these rules for the agent to follow when using the tool:

Discovery First: Always start by listing runs to confirm the run_id. Never guess IDs.

Filter Noise: When fetching history, explicitely request only the relevant metrics (e.g., loss, val_loss, grad_norm). Do not fetch all columns by default.

Visualizing without Eyes: To "see" a curve:

Fetch sampled history (500 points).

Calculate simple statistics: min, max, mean, variance.

Check for NaN values to detect divergence.

(Optional) Use ASCII plotting if a visual overview is strictly required, but stats are usually preferred.

6. Success Criteria
The tool allows retrieving a full training curve for a 1M+ step run using fewer than 3,000 tokens of output context.

The agent can identify the "best" run from the last 5 attempts based on val_loss.

The agent can diagnose a "loss spike" by pinpointing the epoch where the gradient norm exploded.