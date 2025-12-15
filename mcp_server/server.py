#!/usr/bin/env python3
"""
MCP Server for Runwise.

Provides training run analysis tools to MCP-compatible AI assistants.

To use with Claude Code, add to your MCP settings:
{
    "mcpServers": {
        "runwise": {
            "command": "python",
            "args": ["/path/to/runwise/mcp_server/server.py"],
            "env": {
                "RUNWISE_PROJECT_ROOT": "/path/to/your/project"
            }
        }
    }
}
"""

import json
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from runwise import RunAnalyzer, RunwiseConfig, __version__


class MCPServer:
    """Simple MCP server implementation for Runwise."""

    def __init__(self):
        project_root = os.environ.get("RUNWISE_PROJECT_ROOT", os.getcwd())
        config = RunwiseConfig.auto_detect(Path(project_root))
        self.analyzer = RunAnalyzer(config)

    def handle_request(self, request: dict) -> dict:
        """Handle an MCP request. Returns the result to be wrapped in JSON-RPC format."""
        method = request.get("method", "")
        params = request.get("params", {})

        if method == "initialize":
            return {"result": self._initialize(params)}
        elif method == "tools/list":
            return {"result": self._list_tools()}
        elif method == "tools/call":
            return {"result": self._call_tool(params)}
        else:
            return {"error": {"code": -32601, "message": f"Method not found: {method}"}}

    def _initialize(self, params: dict) -> dict:
        """Handle initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "runwise",
                "version": __version__
            }
        }

    def _list_tools(self) -> dict:
        """List available tools with improved descriptions."""
        return {
            "tools": [
                # Quick health check - most common use case
                {
                    "name": "health_check",
                    "description": "Quick training health check - the 'how's training going?' tool. Combines run status, key metrics, sparklines, and anomaly detection in one call. START HERE for most queries.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "run_id": {
                                "type": "string",
                                "description": "Run ID (uses latest/active run if not specified)"
                            }
                        }
                    }
                },
                # Discovery tools - use these first
                {
                    "name": "list_runs",
                    "description": "List recent W&B training runs. Shows run IDs, names, state, and key metrics.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of runs to list",
                                "default": 15
                            }
                        }
                    }
                },
                {
                    "name": "list_keys",
                    "description": "IMPORTANT: Call this FIRST before get_history or get_sparkline to discover available metric keys. Lists all metrics logged in a run's history.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "run_id": {
                                "type": "string",
                                "description": "Run ID (uses latest if not specified)"
                            }
                        }
                    }
                },
                # Analysis tools
                {
                    "name": "analyze_run",
                    "description": "Detailed analysis of a specific run with metrics, sparklines, and anomaly detection.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "run_id": {
                                "type": "string",
                                "description": "W&B run ID to analyze"
                            },
                            "keys": {
                                "type": "string",
                                "description": "Comma-separated metric keys to show (optional, uses schema if not specified)"
                            }
                        },
                        "required": ["run_id"]
                    }
                },
                {
                    "name": "analyze_latest",
                    "description": "Analyze the latest/active training run. Same as analyze_run but auto-selects most recent run.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "keys": {
                                "type": "string",
                                "description": "Comma-separated metric keys to show (optional)"
                            }
                        }
                    }
                },
                {
                    "name": "compare_runs",
                    "description": "Compare metrics between two runs. Supports @step syntax for step-matched comparison (e.g., 'run1@50000' to compare at specific steps). Essential for curriculum learning.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "run_a": {
                                "type": "string",
                                "description": "First run ID (supports @step syntax, e.g., 'abc123@50000')"
                            },
                            "run_b": {
                                "type": "string",
                                "description": "Second run ID (supports @step syntax)"
                            },
                            "filter": {
                                "type": "string",
                                "description": "Filter metrics by prefix (e.g., 'val', 'train')"
                            },
                            "threshold": {
                                "type": "number",
                                "description": "Only show metrics with delta > threshold % (e.g., 5 for 5%)"
                            },
                            "show_config_diff": {
                                "type": "boolean",
                                "description": "Include hyperparameter differences",
                                "default": False
                            }
                        },
                        "required": ["run_a", "run_b"]
                    }
                },
                {
                    "name": "live_status",
                    "description": "Get live status of currently running training from output.log. Shows recent metrics and throughput.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                # History tools - call list_keys first!
                {
                    "name": "get_history",
                    "description": "Get downsampled training history as CSV. Call list_keys first to discover available metrics. Efficiently handles million-step runs.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "run_id": {
                                "type": "string",
                                "description": "Run ID (uses latest if not specified)"
                            },
                            "keys": {
                                "type": "string",
                                "description": "Comma-separated metric keys (REQUIRED - call list_keys first to discover)"
                            },
                            "samples": {
                                "type": "integer",
                                "description": "Number of data points (default: 500)",
                                "default": 500
                            }
                        },
                        "required": ["keys"]
                    }
                },
                {
                    "name": "get_history_stats",
                    "description": "Get statistical summary (min/max/mean/final). More token-efficient than get_history. Call list_keys first.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "run_id": {
                                "type": "string",
                                "description": "Run ID (uses latest if not specified)"
                            },
                            "keys": {
                                "type": "string",
                                "description": "Comma-separated metric keys (REQUIRED - call list_keys first)"
                            }
                        },
                        "required": ["keys"]
                    }
                },
                {
                    "name": "get_sparkline",
                    "description": "Compact sparkline visualization of metric trends. Shows trend in ~10 tokens. Call list_keys first.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "run_id": {
                                "type": "string",
                                "description": "Run ID (uses latest if not specified)"
                            },
                            "keys": {
                                "type": "string",
                                "description": "Comma-separated metric keys (REQUIRED - call list_keys first)"
                            },
                            "samples": {
                                "type": "integer",
                                "description": "Number of data points to sample (default: 50)",
                                "default": 50
                            }
                        },
                        "required": ["keys"]
                    }
                },
                # Metadata tools
                {
                    "name": "get_config",
                    "description": "Get hyperparameters and configuration (learning rate, batch size, model architecture, etc.)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "run_id": {
                                "type": "string",
                                "description": "Run ID (uses latest if not specified)"
                            }
                        }
                    }
                },
                {
                    "name": "get_run_context",
                    "description": "Get run context (name, notes, tags, group). User-provided descriptions of what the run is testing.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "run_id": {
                                "type": "string",
                                "description": "Run ID (uses latest if not specified)"
                            }
                        }
                    }
                },
                {
                    "name": "find_best_run",
                    "description": "Find the best run by a specific metric. Returns ranked list of runs.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "metric": {
                                "type": "string",
                                "description": "Metric to compare (call list_keys to discover available metrics)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of runs to consider (default: 10)",
                                "default": 10
                            },
                            "higher_is_better": {
                                "type": "boolean",
                                "description": "True for metrics like accuracy, false for loss",
                                "default": False
                            }
                        },
                        "required": ["metric"]
                    }
                },
                {
                    "name": "detect_anomalies",
                    "description": "Run anomaly detection. Detects loss spikes, overfitting, plateaus, gradient issues. Returns empty string if healthy (token-efficient).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "run_id": {
                                "type": "string",
                                "description": "Run ID (uses latest if not specified)"
                            },
                            "loss_key": {
                                "type": "string",
                                "description": "Key for loss metric (default: auto-detect)",
                                "default": "loss"
                            }
                        }
                    }
                },
                {
                    "name": "analyze_local_log",
                    "description": "Analyze a local JSONL training log file (not W&B).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "description": "Log file path (uses latest in logs/ if not specified)"
                            }
                        }
                    }
                }
            ]
        }

    def _parse_run_at_step(self, run_spec: str) -> tuple:
        """Parse run@step syntax. Returns (run_id, step) where step may be None."""
        if "@" in run_spec:
            parts = run_spec.rsplit("@", 1)
            run_id = parts[0]
            try:
                step = int(parts[1])
                return run_id, step
            except ValueError:
                return run_spec, None
        return run_spec, None

    def _call_tool(self, params: dict) -> dict:
        """Execute a tool call."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        try:
            # Health check - the quick "how's training going?" tool
            if tool_name == "health_check":
                from runwise.anomalies import detect_anomalies, format_anomalies
                from runwise.sparklines import sparkline, trend_indicator

                run_id = arguments.get("run_id")
                run = self.analyzer.find_run(run_id) if run_id else self.analyzer.get_latest_run()

                if not run:
                    result = "No runs found. Check that you're in a project with a wandb/ directory."
                else:
                    lines = ["TRAINING HEALTH CHECK", "=" * 40, ""]

                    # Basic info
                    state = "RUNNING" if run.state == "running" else run.state.upper()
                    lines.append(f"Run: {run.run_id} | State: {state}")
                    lines.append(f"Name: {run.name or '(unnamed)'}")
                    lines.append(f"Step: {run.final_step:,} | Runtime: {run.metrics.get('_runtime', 0)/3600:.1f}h")
                    lines.append("")

                    # Key metrics with sparklines
                    history = self.analyzer.get_history_data(run, samples=50)
                    if history:
                        lines.append("KEY METRICS:")
                        # Auto-detect common keys
                        common_keys = ["loss", "train/loss", "val_loss", "val/loss",
                                       "accuracy", "train/accuracy", "val/accuracy"]
                        found_keys = []
                        for key in common_keys:
                            if any(key in r for r in history):
                                found_keys.append(key)

                        for key in found_keys[:5]:  # Top 5 metrics
                            values = [r.get(key) for r in history if key in r]
                            if values:
                                spark = sparkline(values, width=12)
                                trend = trend_indicator(values)
                                first = values[0] if values else 0
                                last = values[-1] if values else 0
                                lines.append(f"  {key}: {spark} {trend} ({first:.4g} → {last:.4g})")
                        lines.append("")

                        # Anomaly detection
                        anomalies = detect_anomalies(
                            history,
                            config=self.analyzer.config.anomaly_config,
                            loss_key=self.analyzer.config.schema.loss_key
                        )
                        if anomalies:
                            lines.append("ANOMALIES DETECTED:")
                            lines.append(format_anomalies(anomalies, compact=True))
                        else:
                            lines.append("STATUS: Healthy - no anomalies detected")
                    else:
                        lines.append("(no history data available yet)")

                    result = "\n".join(lines)

            elif tool_name == "list_runs":
                limit = arguments.get("limit", 15)
                runs = self.analyzer.list_runs(limit=limit)
                result = self.analyzer.format_run_list(runs) if runs else "No runs found"

            elif tool_name == "analyze_run":
                run_id = arguments.get("run_id")
                run = self.analyzer.find_run(run_id)
                if run:
                    keys = None
                    if arguments.get("keys"):
                        keys = [k.strip() for k in arguments["keys"].split(",")]
                    result = self.analyzer.summarize_run(run, keys=keys)
                else:
                    result = f"Run '{run_id}' not found"

            elif tool_name == "analyze_latest":
                run = self.analyzer.get_latest_run()
                if run:
                    keys = None
                    if arguments.get("keys"):
                        keys = [k.strip() for k in arguments["keys"].split(",")]
                    result = self.analyzer.summarize_run(run, keys=keys)
                else:
                    result = "No latest run found"

            elif tool_name == "compare_runs":
                # Parse @step syntax
                run_spec_a = arguments.get("run_a", "")
                run_spec_b = arguments.get("run_b", "")
                run_id_a, step_a = self._parse_run_at_step(run_spec_a)
                run_id_b, step_b = self._parse_run_at_step(run_spec_b)

                run_a = self.analyzer.find_run(run_id_a)
                run_b = self.analyzer.find_run(run_id_b)

                if not run_a:
                    result = f"Run '{run_id_a}' not found"
                elif not run_b:
                    result = f"Run '{run_id_b}' not found"
                elif step_a is not None or step_b is not None:
                    # Step-matched comparison
                    if step_a is None:
                        step_a = run_a.final_step
                    if step_b is None:
                        step_b = run_b.final_step
                    result = self.analyzer.compare_runs_at_step(
                        run_a,
                        run_b,
                        step_a=step_a,
                        step_b=step_b,
                        filter_prefix=arguments.get("filter"),
                        threshold=arguments.get("threshold"),
                        show_config_diff=arguments.get("show_config_diff", False)
                    )
                else:
                    # Regular comparison
                    result = self.analyzer.compare_runs(
                        run_a,
                        run_b,
                        filter_prefix=arguments.get("filter"),
                        show_config_diff=arguments.get("show_config_diff", False),
                        threshold=arguments.get("threshold"),
                    )

            elif tool_name == "live_status":
                result = self.analyzer.get_live_status()

            elif tool_name == "analyze_local_log":
                file_arg = arguments.get("file")
                if file_arg:
                    log_file = Path(file_arg)
                    if not log_file.exists():
                        log_file = self.analyzer.config.logs_dir / file_arg
                else:
                    logs = self.analyzer.list_local_logs()
                    log_file = logs[0] if logs else None

                if log_file and log_file.exists():
                    result = self.analyzer.summarize_local_log(log_file)
                else:
                    result = "No log file found"

            elif tool_name == "get_history":
                run_id = arguments.get("run_id")
                run = self.analyzer.find_run(run_id) if run_id else self.analyzer.get_latest_run()
                if not run:
                    result = "Run not found"
                else:
                    keys = [k.strip() for k in arguments.get("keys", "").split(",")]
                    samples = arguments.get("samples", 500)
                    result = self.analyzer.get_history(run, keys, samples=samples)

            elif tool_name == "get_history_stats":
                run_id = arguments.get("run_id")
                run = self.analyzer.find_run(run_id) if run_id else self.analyzer.get_latest_run()
                if not run:
                    result = "Run not found"
                else:
                    keys = [k.strip() for k in arguments.get("keys", "").split(",")]
                    result = self.analyzer.get_history_stats(run, keys)

            elif tool_name == "list_keys":
                run_id = arguments.get("run_id")
                run = self.analyzer.find_run(run_id) if run_id else self.analyzer.get_latest_run()
                if not run:
                    result = "Run not found"
                else:
                    result = self.analyzer.list_available_keys(run)

            elif tool_name == "get_config":
                run_id = arguments.get("run_id")
                run = self.analyzer.find_run(run_id) if run_id else self.analyzer.get_latest_run()
                if not run:
                    result = "Run not found"
                else:
                    result = self.analyzer.get_config(run)

            elif tool_name == "get_run_context":
                run_id = arguments.get("run_id")
                run = self.analyzer.find_run(run_id) if run_id else self.analyzer.get_latest_run()
                if not run:
                    result = "Run not found"
                else:
                    result = self.analyzer.get_run_context(run)

            elif tool_name == "find_best_run":
                metric = arguments.get("metric")
                limit = arguments.get("limit", 10)
                higher_is_better = arguments.get("higher_is_better", False)
                result = self.analyzer.format_best_run(metric, limit, higher_is_better)

            elif tool_name == "detect_anomalies":
                from runwise.anomalies import detect_anomalies, format_anomalies

                run_id = arguments.get("run_id")
                run = self.analyzer.find_run(run_id) if run_id else self.analyzer.get_latest_run()
                if not run:
                    result = "Run not found"
                else:
                    loss_key = arguments.get("loss_key", self.analyzer.config.schema.loss_key)
                    history = self.analyzer.get_history_data(run, samples=200)
                    if not history:
                        result = "No history data for anomaly detection"
                    else:
                        anomalies = detect_anomalies(
                            history,
                            config=self.analyzer.config.anomaly_config,
                            loss_key=loss_key
                        )
                        if anomalies:
                            result = format_anomalies(anomalies, compact=True)
                        else:
                            result = "No anomalies detected - run appears healthy"

            elif tool_name == "get_sparkline":
                from runwise.sparklines import sparkline, trend_indicator

                run_id = arguments.get("run_id")
                run = self.analyzer.find_run(run_id) if run_id else self.analyzer.get_latest_run()
                if not run:
                    result = "Run not found"
                else:
                    keys = [k.strip() for k in arguments.get("keys", "").split(",")]
                    samples = arguments.get("samples", 50)
                    history = self.analyzer.get_history_data(run, keys=keys, samples=samples)

                    if not history:
                        result = f"No history data for keys: {keys}"
                    else:
                        lines = [f"SPARKLINES: {run.run_id}"]
                        for key in keys:
                            values = [r.get(key) for r in history if key in r]
                            if values:
                                spark = sparkline(values, width=20)
                                trend = trend_indicator(values)
                                first = values[0] if values else 0
                                last = values[-1] if values else 0
                                lines.append(f"  {key}: {spark} {trend} ({first:.4g}→{last:.4g})")
                            else:
                                lines.append(f"  {key}: (no data)")
                        result = "\n".join(lines)

            else:
                return {"error": {"code": -32602, "message": f"Unknown tool: {tool_name}"}}

            return {
                "content": [
                    {
                        "type": "text",
                        "text": result
                    }
                ]
            }

        except Exception as e:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error executing {tool_name}: {str(e)}"
                    }
                ],
                "isError": True
            }

    def run(self):
        """Run the MCP server (stdio transport)."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                request = json.loads(line)
                handler_response = self.handle_request(request)

                # Construct proper JSON-RPC 2.0 response
                # The response should be: {"jsonrpc": "2.0", "id": X, "result": {...}}
                # OR for errors: {"jsonrpc": "2.0", "id": X, "error": {...}}
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id")
                }

                # handler_response contains either {"result": ...} or {"error": ...}
                if "error" in handler_response:
                    response["error"] = handler_response["error"]
                else:
                    response["result"] = handler_response.get("result", handler_response)

                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

            except json.JSONDecodeError:
                continue
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32603, "message": str(e)}
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()


def main():
    server = MCPServer()
    server.run()


if __name__ == "__main__":
    main()
