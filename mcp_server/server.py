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

from runwise import RunAnalyzer, RunwiseConfig


class MCPServer:
    """Simple MCP server implementation for Runwise."""

    def __init__(self):
        project_root = os.environ.get("RUNWISE_PROJECT_ROOT", os.getcwd())
        config = RunwiseConfig.auto_detect(Path(project_root))
        self.analyzer = RunAnalyzer(config)

    def handle_request(self, request: dict) -> dict:
        """Handle an MCP request."""
        method = request.get("method", "")
        params = request.get("params", {})

        if method == "initialize":
            return self._initialize(params)
        elif method == "tools/list":
            return self._list_tools()
        elif method == "tools/call":
            return self._call_tool(params)
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
                "version": "0.1.0"
            }
        }

    def _list_tools(self) -> dict:
        """List available tools."""
        return {
            "tools": [
                {
                    "name": "list_runs",
                    "description": "List recent W&B training runs with basic metrics",
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
                    "name": "analyze_run",
                    "description": "Get detailed analysis of a specific training run",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "run_id": {
                                "type": "string",
                                "description": "W&B run ID to analyze"
                            }
                        },
                        "required": ["run_id"]
                    }
                },
                {
                    "name": "analyze_latest",
                    "description": "Get detailed analysis of the latest/active training run",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "compare_runs",
                    "description": "Compare metrics between two training runs",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "run_a": {
                                "type": "string",
                                "description": "First run ID"
                            },
                            "run_b": {
                                "type": "string",
                                "description": "Second run ID"
                            },
                            "filter": {
                                "type": "string",
                                "description": "Filter metrics by prefix (e.g., 'val', 'train')"
                            },
                            "show_config_diff": {
                                "type": "boolean",
                                "description": "Include config differences in output",
                                "default": false
                            }
                        },
                        "required": ["run_a", "run_b"]
                    }
                },
                {
                    "name": "live_status",
                    "description": "Get live status of currently running training",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                },
                {
                    "name": "analyze_local_log",
                    "description": "Analyze a local training log file (JSONL format)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "description": "Log file name or path (optional, uses latest if not specified)"
                            }
                        }
                    }
                },
                {
                    "name": "get_history",
                    "description": "Get downsampled training history as CSV. Efficiently handles million-step runs by returning only N sampled points.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "run_id": {
                                "type": "string",
                                "description": "Run ID (uses latest if not specified)"
                            },
                            "keys": {
                                "type": "string",
                                "description": "Comma-separated metric keys (e.g., 'loss,val_loss,grad_norm')"
                            },
                            "samples": {
                                "type": "integer",
                                "description": "Number of data points to return (default: 500)",
                                "default": 500
                            }
                        },
                        "required": ["keys"]
                    }
                },
                {
                    "name": "get_history_stats",
                    "description": "Get statistical summary of training history (min/max/mean/final/NaN count). Even more token-efficient than get_history.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "run_id": {
                                "type": "string",
                                "description": "Run ID (uses latest if not specified)"
                            },
                            "keys": {
                                "type": "string",
                                "description": "Comma-separated metric keys"
                            }
                        },
                        "required": ["keys"]
                    }
                },
                {
                    "name": "list_keys",
                    "description": "List available metric keys in a run's history. Use this when you don't know what metrics are logged.",
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
                    "name": "get_config",
                    "description": "Get hyperparameters and configuration for a training run. Essential for debugging (learning rate, batch size, model architecture, etc.)",
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
                    "description": "Get run context (name, notes, tags, group). Returns user-provided descriptions of what the run is testing, variable sweeps, hypotheses, etc. Use this to understand the intent behind a run.",
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
                                "description": "Metric to compare (e.g., 'val_loss', 'accuracy', 'train/loss')"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of recent runs to consider (default: 10)",
                                "default": 10
                            },
                            "higher_is_better": {
                                "type": "boolean",
                                "description": "If true, higher values are better (e.g., accuracy). Default: false (lower is better, e.g., loss)",
                                "default": false
                            }
                        },
                        "required": ["metric"]
                    }
                }
            ]
        }

    def _call_tool(self, params: dict) -> dict:
        """Execute a tool call."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        try:
            if tool_name == "list_runs":
                limit = arguments.get("limit", 15)
                runs = self.analyzer.list_runs(limit=limit)
                result = self.analyzer.format_run_list(runs) if runs else "No runs found"

            elif tool_name == "analyze_run":
                run_id = arguments.get("run_id")
                run = self.analyzer.find_run(run_id)
                result = self.analyzer.summarize_run(run) if run else f"Run '{run_id}' not found"

            elif tool_name == "analyze_latest":
                run = self.analyzer.get_latest_run()
                result = self.analyzer.summarize_run(run) if run else "No latest run found"

            elif tool_name == "compare_runs":
                run_a = self.analyzer.find_run(arguments.get("run_a"))
                run_b = self.analyzer.find_run(arguments.get("run_b"))
                if not run_a:
                    result = f"Run '{arguments.get('run_a')}' not found"
                elif not run_b:
                    result = f"Run '{arguments.get('run_b')}' not found"
                else:
                    result = self.analyzer.compare_runs(
                        run_a,
                        run_b,
                        filter_prefix=arguments.get("filter"),
                        show_config_diff=arguments.get("show_config_diff", False)
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
                    result = f"Run not found"
                else:
                    keys = [k.strip() for k in arguments.get("keys", "").split(",")]
                    samples = arguments.get("samples", 500)
                    result = self.analyzer.get_history(run, keys, samples=samples)

            elif tool_name == "get_history_stats":
                run_id = arguments.get("run_id")
                run = self.analyzer.find_run(run_id) if run_id else self.analyzer.get_latest_run()
                if not run:
                    result = f"Run not found"
                else:
                    keys = [k.strip() for k in arguments.get("keys", "").split(",")]
                    result = self.analyzer.get_history_stats(run, keys)

            elif tool_name == "list_keys":
                run_id = arguments.get("run_id")
                run = self.analyzer.find_run(run_id) if run_id else self.analyzer.get_latest_run()
                if not run:
                    result = f"Run not found"
                else:
                    result = self.analyzer.list_available_keys(run)

            elif tool_name == "get_config":
                run_id = arguments.get("run_id")
                run = self.analyzer.find_run(run_id) if run_id else self.analyzer.get_latest_run()
                if not run:
                    result = f"Run not found"
                else:
                    result = self.analyzer.get_config(run)

            elif tool_name == "get_run_context":
                run_id = arguments.get("run_id")
                run = self.analyzer.find_run(run_id) if run_id else self.analyzer.get_latest_run()
                if not run:
                    result = f"Run not found"
                else:
                    result = self.analyzer.get_run_context(run)

            elif tool_name == "find_best_run":
                metric = arguments.get("metric")
                limit = arguments.get("limit", 10)
                higher_is_better = arguments.get("higher_is_better", False)
                result = self.analyzer.format_best_run(metric, limit, higher_is_better)

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
                response = self.handle_request(request)

                # Add JSON-RPC fields
                response["jsonrpc"] = "2.0"
                response["id"] = request.get("id")

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
