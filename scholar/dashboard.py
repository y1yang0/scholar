#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scholar Dashboard - Web interface for training control, monitoring, and inference.
Run: python dashboard.py
Open: http://localhost:5002
"""

import io
import os
import sys
import json
import base64
import hashlib
import signal
import subprocess
import logging
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, request, Response

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

# Add parent directory to path for importing scholar modules
SCRIPT_DIR = Path(__file__).parent

# ============== CONFIG ==============
charts = [
    {
        "files": ["../train.log"],
        "fields": ["trainLoss"],
        "xlabel": "Iteration",
        "ylabel": "Loss",
        "title": "Train Loss",
    },
    {
        "files": ["../train.log"],
        "fields": ["valLoss"],
        "xlabel": "Iteration",
        "ylabel": "Loss",
        "title": "Val Loss",
    },
    {
        "files": ["../train.log"],
        "fields": ["valPPL", "trainPPL"],
        "xlabel": "Iteration",
        "ylabel": "PPL",
        "title": "Perplexity",
    },
    {
        "files": ["../train.log"],
        "fields": ["elapsed"],
        "xlabel": "Epoch",
        "ylabel": "seconds",
        "title": "Elapsed",
    },
    {
        "files": ["../train.log"],
        "fields": ["gradNorm"],
        "xlabel": "Iteration",
        "ylabel": "Gradient Norm",
        "title": "Gradient Norm",
    },
    {
        "files": ["../train.log"],
        "fields": ["currentLR"],
        "xlabel": "Iteration",
        "ylabel": "Learning Rate",
        "title": "Learning Rate",
    },
]
metrics = [
    {"file": "../train.log", "type": "uptime", "title": "Uptime"},
    {"file": "../train.log", "event": "config"},
]
port = 5002
refresh_interval = 3
# ====================================

app = Flask(__name__)
LOGO_PATH = SCRIPT_DIR.parent / "misc" / "logo.png"
LOGO_PURE_PATH = SCRIPT_DIR.parent / "misc" / "logo_pure.png"

# Global state
train_process = None
scholar_instance = None
current_weight_path = None


# ============== UTILITY FUNCTIONS ==============


def get_logo_base64():
    if not LOGO_PATH.exists():
        return None
    try:
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except:
        return None


def get_logo_pure_base64():
    if not LOGO_PURE_PATH.exists():
        return None
    try:
        with open(LOGO_PURE_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except:
        return None


def get_gpu_stats():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "utilization": int(parts[2]),
                        "memory_used": int(parts[3]),
                        "memory_total": int(parts[4]),
                        "temperature": int(parts[5]),
                    }
                )
        return gpus
    except Exception:
        return []


def parse_log_file(log_path: str, fields: list):
    result = {field: [] for field in fields}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    for field in fields:
                        if field in data:
                            result[field].append(data[field])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return result


def draw_single_chart(chart_config: dict, ratio: int = 100):
    files = chart_config.get("files", [])
    fields = chart_config.get("fields", [])
    xlabel = chart_config.get("xlabel", "Iteration")
    ylabel = chart_config.get("ylabel", "Value")
    title = chart_config.get("title", "")

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    has_data = False
    color_idx = 0
    stats_lines = []

    for file_path in files:
        path = Path(file_path)
        if not path.is_absolute():
            path = SCRIPT_DIR / path
        if not path.exists():
            continue
        field_data = parse_log_file(str(path), fields)
        for field in fields:
            values = field_data[field]
            if not values:
                continue
            
            # Apply ratio: keep only the last N% of data
            if ratio < 100:
                start_idx = int(len(values) * (100 - ratio) / 100)
                values = values[start_idx:]
            
            if not values:
                continue
            has_data = True
            indices = list(range(len(values)))
            color = colors[color_idx % len(colors)]
            color_idx += 1
            label = field if len(files) == 1 else f"{path.stem}:{field}"
            ax.plot(indices, values, "-", linewidth=1.5, color=color, label=label)
            min_val, max_val, cur_val = min(values), max(values), values[-1]
            min_idx, max_idx = values.index(min_val), values.index(max_val)
            ax.scatter(
                [min_idx],
                [min_val],
                color=color,
                s=40,
                zorder=5,
                edgecolors="white",
                linewidths=1,
            )
            ax.scatter(
                [max_idx],
                [max_val],
                color=color,
                s=40,
                zorder=5,
                edgecolors="white",
                linewidths=1,
                marker="^",
            )
            stats_lines.append(
                f"{label}: min={min_val:.4g}  max={max_val:.4g}  cur={cur_val:.4g}"
            )

    if not has_data:
        ax.text(
            0.5,
            0.5,
            "No Data",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
            transform=ax.transAxes,
        )

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title or " & ".join(fields), fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    if has_data:
        ax.legend(loc="upper right", fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if stats_lines:
        fig.text(
            0.5, 0.02, "  |  ".join(stats_lines), ha="center", fontsize=10, color="#333"
        )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


def draw_all_charts_combined():
    import math

    n = len(charts)
    if n == 0:
        return None
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    for i, chart_config in enumerate(charts):
        ax = axes[i]
        files = chart_config.get("files", [])
        fields = chart_config.get("fields", [])
        xlabel = chart_config.get("xlabel", "Iteration")
        ylabel = chart_config.get("ylabel", "Value")
        title = chart_config.get("title", "")
        has_data = False
        color_idx = 0

        for file_path in files:
            path = Path(file_path)
            if not path.is_absolute():
                path = SCRIPT_DIR / path
            if not path.exists():
                continue
            field_data = parse_log_file(str(path), fields)
            for field in fields:
                values = field_data[field]
                if not values:
                    continue
                has_data = True
                indices = list(range(len(values)))
                color = colors[color_idx % len(colors)]
                color_idx += 1
                label = field if len(files) == 1 else f"{path.stem}:{field}"
                ax.plot(indices, values, "-", linewidth=1.5, color=color, label=label)
                min_val, max_val = min(values), max(values)
                min_idx, max_idx = values.index(min_val), values.index(max_val)
                ax.scatter(
                    [min_idx],
                    [min_val],
                    color=color,
                    s=30,
                    zorder=5,
                    edgecolors="white",
                    linewidths=1,
                )
                ax.scatter(
                    [max_idx],
                    [max_val],
                    color=color,
                    s=30,
                    zorder=5,
                    edgecolors="white",
                    linewidths=1,
                    marker="^",
                )

        if not has_data:
            ax.text(
                0.5,
                0.5,
                "No Data",
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
                transform=ax.transAxes,
            )
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title or " & ".join(fields), fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)
        if has_data:
            ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_data = buf.read()
    plt.close(fig)
    return img_data


def get_file_hash(file_path: Path):
    if not file_path.exists():
        return None
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None


def get_data_hash():
    hashes = []
    for chart in charts:
        for file_path in chart.get("files", []):
            path = Path(file_path)
            if not path.is_absolute():
                path = SCRIPT_DIR / path
            h = get_file_hash(path)
            if h:
                hashes.append(h)
    return hashlib.md5("".join(hashes).encode()).hexdigest() if hashes else ""


def format_value(val):
    if isinstance(val, float):
        return f"{val:.2e}" if val < 0.01 else f"{val:.4g}"
    if isinstance(val, int):
        if val >= 1_000_000:
            return f"{val/1_000_000:.1f}M"
        if val >= 1_000:
            return f"{val/1_000:.1f}K"
    return str(val)


def get_metrics_data():
    import time

    result = []
    for m in metrics:
        mtype = m.get("type", "")
        file_path = m.get("file", "")
        path = Path(file_path)
        if not path.is_absolute():
            path = SCRIPT_DIR / path
        if not path.exists():
            if "title" in m:
                result.append({"title": m["title"], "value": "--"})
            continue

        if mtype == "uptime":
            title = m.get("title", "Uptime")
            value = "--"
            start_time = None
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if data.get("event") == "start" and "timestamp" in data:
                                start_time = data["timestamp"]
                        except json.JSONDecodeError:
                            continue
            except:
                pass
            if start_time:
                elapsed = time.time() - start_time
                hours, minutes, seconds = (
                    int(elapsed // 3600),
                    int((elapsed % 3600) // 60),
                    int(elapsed % 60),
                )
                if hours > 0:
                    value = f"{hours}h {minutes}m {seconds}s"
                elif minutes > 0:
                    value = f"{minutes}m {seconds}s"
                else:
                    value = f"{seconds}s"
            result.append({"title": title, "value": value})

        elif "event" in m:
            event_name = m.get("event", "")
            skip_fields = {"event"}
            seen_keys = set()
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if data.get("event") == event_name:
                                for key, val in data.items():
                                    if key in skip_fields or key in seen_keys:
                                        continue
                                    seen_keys.add(key)
                                    result.append(
                                        {
                                            "title": key[0].upper() + key[1:],
                                            "value": format_value(val),
                                        }
                                    )
                        except json.JSONDecodeError:
                            continue
            except:
                pass
    return result


def get_inference_samples():
    samples = []
    log_path = SCRIPT_DIR / "../train.log"
    if not log_path.exists():
        return samples
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "idx" not in data or data["idx"] != 0:
                        continue
                    if "input" in data and "output" in data:
                        samples.append(
                            {"input": data["input"], "output": data["output"]}
                        )
                except json.JSONDecodeError:
                    continue
    except:
        pass
    return samples[-10:] if len(samples) > 10 else samples


def get_available_weights():
    weights = []
    root = SCRIPT_DIR.parent
    for name in ["scholar_best.bin", "scholar_last.bin"]:
        path = root / name
        if path.exists():
            weights.append(
                {"name": name, "path": str(path), "size": path.stat().st_size}
            )
    return weights


def parse_inspect_log():
    """Parse inspect.log and return layers data"""
    log_path = SCRIPT_DIR.parent / "inspect.log"
    if not log_path.exists():
        return []

    layers_dict = {}

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    layer = data.get("layer", "")
                    input_token = data.get("inputToken", "")
                    similar = data.get("similar", [])

                    if layer not in layers_dict:
                        layers_dict[layer] = []

                    layers_dict[layer].append(
                        {"inputToken": input_token, "similar": similar}
                    )
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    return organize_layers(layers_dict)


def organize_layers(layers_dict):
    layer_order = [
        "Attn#0",
        "FFN#0",
        "Attn#1",
        "FFN#1",
        "Attn#2",
        "FFN#2",
        "Attn#3",
        "FFN#3",
        "Attn#4",
        "FFN#4",
        "Attn#5",
        "FFN#5",
        "Attn#6",
        "FFN#6",
        "Attn#7",
        "FFN#7",
        "Attn#8",
        "FFN#8",
        "Attn#9",
        "FFN#9",
        "Attn#10",
        "FFN#10",
        "Attn#11",
        "FFN#11",
        "Attn#12",
        "FFN#12",
        "Attn#13",
        "FFN#13",
        "Attn#14",
        "FFN#14",
        "Attn#15",
        "FFN#15",
        "FinalNorm",
    ]

    layers = []
    for layer_name in layer_order:
        if layer_name in layers_dict:
            layers.append({"layer": layer_name, "tokens": layers_dict[layer_name]})

    for layer_name, tokens in layers_dict.items():
        if layer_name not in layer_order:
            layers.append({"layer": layer_name, "tokens": tokens})

    return layers


# ============== TRAINING CONTROL ==============


def get_train_status():
    global train_process
    if train_process is None:
        return "stopped"
    poll = train_process.poll()
    if poll is None:
        return "running"
    return "stopped"


def find_existing_training_processes():
    """Find all scholar training processes running on the system"""
    training_processes = []

    if sys.platform == "win32":
        try:
            # Use PowerShell with simpler query
            ps_command = 'Get-Process python -ErrorAction SilentlyContinue | ForEach-Object { $p = $_; $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($p.Id)").CommandLine; [PSCustomObject]@{ ProcessId=$p.Id; CommandLine=$cmd } } | ConvertTo-Json'
            result = subprocess.run(
                ["powershell", "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                import json as json_module

                processes = json_module.loads(result.stdout)
                if isinstance(processes, dict):
                    processes = [processes]
                for proc in processes:
                    cmdline = proc.get("CommandLine", "") or ""
                    if "scholar.py" in cmdline and "train" in cmdline:
                        training_processes.append(
                            {
                                "pid": proc["ProcessId"],
                                "cmdline": cmdline,
                                "status": "running",
                            }
                        )
        except Exception:
            pass
    else:
        # Linux/Mac - use ps
        try:
            result = subprocess.run(
                ["ps", "aux"], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().split("\n"):
                if "scholar.py" in line and "train" in line and "grep" not in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        pid = int(parts[1])
                        training_processes.append(
                            {"pid": pid, "cmdline": line, "status": "running"}
                        )
        except Exception:
            pass

    return training_processes


def get_train_process_info():
    """Get detailed info about the training process"""
    global train_process
    if train_process is None or train_process.poll() is not None:
        return None

    info = {"pid": train_process.pid, "args": train_process.args}

    # Try to get more info on Windows
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                [
                    "wmic",
                    "process",
                    "where",
                    f"ProcessId={train_process.pid}",
                    "get",
                    "CommandLine",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            lines = [
                l.strip()
                for l in result.stdout.strip().split("\n")
                if l.strip() and l.strip() != "CommandLine"
            ]
            if lines:
                info["cmdline"] = lines[0]
        except:
            pass

    return info


# ============== INFERENCE ==============


def load_model_for_inference(weight_path=None):
    global scholar_instance, current_weight_path

    # Change to project root for proper file access (tokenizer uses "scholar/tokenizer.json")
    original_cwd = os.getcwd()
    os.chdir(str(SCRIPT_DIR.parent))

    try:
        import scholar as scholar_module

        if weight_path is None:
            weight_path = str(SCRIPT_DIR.parent / "scholar_best.bin")

        # Check if we need to reload
        if scholar_instance is not None and current_weight_path == weight_path:
            return scholar_instance

        # Load model
        config = scholar_module.createModelConfig()
        scholar_instance = scholar_module.Scholar(config)
        scholar_instance.model.loadWeights(weight_path)
        scholar_instance.model.eval()
        current_weight_path = weight_path
        return scholar_instance
    finally:
        os.chdir(original_cwd)


# ============== FLASK ROUTES ==============


@app.route("/")
def index():
    return build_html()


@app.route("/api/charts")
def api_charts():
    ratio = request.args.get("ratio", "100", type=int)
    if ratio < 10 or ratio > 100:
        ratio = 100
    charts_data = [draw_single_chart(c, int(ratio)) for c in charts]
    return jsonify(
        {
            "charts": charts_data,
            "metrics": get_metrics_data(),
            "samples": get_inference_samples(),
            "gpu_stats": get_gpu_stats(),
            "hash": get_data_hash() + f"_r{ratio}",
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/save_all")
def api_save_all():
    img_data = draw_all_charts_combined()
    if img_data:
        return Response(
            img_data,
            mimetype="image/png",
            headers={
                "Content-Disposition": 'attachment; filename="training_charts.png"'
            },
        )
    return "", 404


@app.route("/api/train/status")
def api_train_status():
    return jsonify({"status": get_train_status()})


@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    global train_process

    # First check if there are already training processes running
    existing_processes = find_existing_training_processes()
    if existing_processes:
        # There's already a training process, but not tracked by us
        return (
            jsonify(
                {
                    "error": "Training process already running",
                    "existing_processes": existing_processes,
                }
            ),
            409,
        )

    resume = ""
    if request.is_json and request.json:
        resume = request.json.get("resume", "")
    scholar_path = SCRIPT_DIR / "scholar.py"
    log_path = SCRIPT_DIR.parent / "stdout.log"

    try:
        log_file = open(log_path, "a", encoding="utf-8")
        log_file.write(f"\n{'='*50}\nTraining started at {datetime.now()}\n{'='*50}\n")
        log_file.flush()

        if resume:
            train_process = subprocess.Popen(
                [sys.executable, str(scholar_path), "train", resume],
                cwd=str(SCRIPT_DIR.parent),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=(
                    (
                        subprocess.DETACHED_PROCESS
                        | subprocess.CREATE_NO_WINDOW
                        | subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                    if sys.platform == "win32"
                    else 0
                ),
            )
        else:
            train_process = subprocess.Popen(
                [sys.executable, str(scholar_path), "train"],
                cwd=str(SCRIPT_DIR.parent),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=(
                    (
                        subprocess.DETACHED_PROCESS
                        | subprocess.CREATE_NO_WINDOW
                        | subprocess.CREATE_NEW_PROCESS_GROUP
                    )
                    if sys.platform == "win32"
                    else 0
                ),
            )
        return jsonify({"status": "started", "pid": train_process.pid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train/info")
def api_train_info():
    info = get_train_process_info()
    if info is None:
        return jsonify({"error": "No training process running"}), 400
    return jsonify(info)


@app.route("/api/train/processes", methods=["GET"])
def api_train_processes():
    """Get list of all training processes on the system"""
    processes = find_existing_training_processes()
    return jsonify(processes)


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    global train_process
    if get_train_status() != "running":
        return jsonify({"error": "Training not running"}), 400

    try:
        if sys.platform == "win32":
            # On Windows, use taskkill to forcefully terminate the process tree
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(train_process.pid)],
                capture_output=True,
            )
        else:
            train_process.terminate()
            train_process.wait(timeout=5)
    except Exception:
        # Force kill if terminate fails
        train_process.kill()

    train_process = None
    return jsonify({"status": "stopped"})


@app.route("/api/train/kill/<int:pid>", methods=["POST"])
def api_train_kill(pid):
    """Kill a specific training process by PID"""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=5
            )
        else:
            os.kill(pid, signal.SIGTERM)
        return jsonify({"status": "killed", "pid": pid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/weights")
def api_weights():
    return jsonify({"weights": get_available_weights()})


@app.route("/api/infer", methods=["POST"])
def api_infer():
    """Run inference with Inspector tracing"""
    global scholar_instance
    data = request.json
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No input text"}), 400
    debug = data.get("debug", False)

    weight_path = data.get("weight")
    if weight_path:
        weight_path = str(SCRIPT_DIR.parent / weight_path)

    try:
        import scholar as scholar_module

        # Load model
        config = scholar_module.createModelConfig()
        instance = scholar_module.Scholar(config)
        instance.model.loadWeights(
            weight_path or str(SCRIPT_DIR.parent / "scholar_best.bin")
        )
        instance.model.eval()

        # Enable debug mode
        scholar_module.IsDebug = debug

        # Apply parameters
        instance.model.config["temperature"] = float(data.get("temperature", 0.7))
        instance.model.config["topP"] = float(data.get("topP", 0.85))
        max_tokens = int(data.get("maxTokens", 50))

        # Run inference with debugging
        output = instance.model.nextToken(text, numNextToken=max_tokens)

        # Get Inspector data if available
        inspector_output = None
        if hasattr(scholar_module.Inspector, "get_trace"):
            inspector_output = scholar_module.Inspector.get_trace()

        return jsonify({"input": text, "output": output, "inspector": inspector_output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/inspect")
def api_inspect():
    """Get parsed inspect.log data"""
    return jsonify({"layers": parse_inspect_log()})


@app.route("/api/logs")
def api_logs():
    """List all .log files in the project root directory"""
    root = SCRIPT_DIR.parent
    log_files = []
    for f in root.glob("*.log"):
        log_files.append({"name": f.name, "size": f.stat().st_size})
    log_files.sort(key=lambda x: x["name"].lower())
    return jsonify({"logs": log_files})


@app.route("/api/logs/read")
def api_logs_read():
    """Read log file content, limited to last 1000 lines if too long"""
    log_name = request.args.get("name", "")
    if not log_name:
        return jsonify({"error": "No log file specified"}), 400

    # Build path from filename only (security: no directory traversal)
    if ".." in log_name or "/" in log_name or "\\" in log_name:
        return jsonify({"error": "Invalid filename"}), 400

    path = SCRIPT_DIR.parent / log_name

    if not path.exists():
        return jsonify({"error": "File not found", "checked_path": str(path)}), 404

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        if total_lines > 1000:
            lines = lines[-1000:]

        return jsonify(
            {
                "content": "".join(lines),
                "total_lines": total_lines,
                "displayed_lines": len(lines),
                "truncated": total_lines > 1000,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/logs/clear", methods=["POST"])
def api_logs_clear():
    """Clear the content of a log file"""
    log_name = request.args.get("name", "")
    if not log_name:
        return jsonify({"error": "No log file specified"}), 400
    
    # Build path from filename only (security: no directory traversal)
    if ".." in log_name or "/" in log_name or "\\" in log_name:
        return jsonify({"error": "Invalid filename"}), 400
    
    path = SCRIPT_DIR.parent / log_name
    
    if not path.exists():
        return jsonify({"error": "File not found"}), 404
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return jsonify({"success": True, "message": f"Cleared {log_name}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============== HTML BUILDER ==============


def build_html():
    logo_base64 = get_logo_base64()
    logo_data_uri = f"data:image/png;base64,{logo_base64}" if logo_base64 else ""
    logo_pure_base64 = get_logo_pure_base64()
    logo_pure_data_uri = (
        f"data:image/png;base64,{logo_pure_base64}"
        if logo_pure_base64
        else logo_data_uri
    )

    # Build chart cards
    cards = ""
    for i, chart in enumerate(charts):
        title = chart.get("title", f"Chart {i+1}")
        cards += f"""
        <div class="card">
            <div class="card-header">
                <span class="card-title">{title}</span>
                <button class="copy-btn" onclick="copyChart({i})">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2"/>
                        <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
                    </svg>
                    Copy
                </button>
            </div>
            <div class="chart-container" id="chart-{i}">
                <div class="loading"><div class="spinner"></div>Loading...</div>
            </div>
        </div>"""

    favicon_html = (
        f'<link rel="icon" type="image/png" href="{logo_data_uri}">'
        if logo_data_uri
        else ""
    )
    logo_img_html = (
        f'<img src="{logo_data_uri}" alt="Scholar Logo" class="header-logo">'
        if logo_data_uri
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scholar Dashboard</title>
    {favicon_html}
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; min-height: 100vh; }}
        .header {{ background: #fff; border-bottom: 1px solid #e0e0e0; padding: 12px 20px; display: flex; align-items: center; justify-content: space-between; }}
        .header-left {{ display: flex; align-items: center; gap: 12px; }}
        .header-logo {{ width: 32px; height: 32px; object-fit: contain; }}
        .header h1 {{ font-size: 1.2rem; font-weight: 600; color: #333; }}
        .train-status {{ display: flex; align-items: center; gap: 8px; padding: 8px 16px; border-radius: 4px; font-size: 0.85rem; }}
        .train-status .dot {{ width: 8px; height: 8px; border-radius: 50%; }}
        .train-status.running {{ background: #f6ffed; color: #52c41a; }}
        .train-status.running .dot {{ background: #52c41a; animation: pulse 2s infinite; }}
        .train-status.stopped {{ background: #fff2f0; color: #ff4d4f; }}
        .train-status.stopped .dot {{ background: #ff4d4f; }}
        @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
        /* Main Layout with Sidebar */
        .app-layout {{ display: flex; min-height: calc(100vh - 57px); }}
        .side-nav {{ width: 140px; background: #fff; border-right: 1px solid #e0e0e0; padding: 16px 8px; flex-shrink: 0; }}
        .side-nav-title {{ font-size: 0.75rem; font-weight: 600; color: #999; text-transform: uppercase; margin: 16px 12px 8px; }}
        .side-nav-btn {{ display: block; width: 100%; padding: 10px 12px; margin-bottom: 4px; border: none; background: transparent; border-radius: 6px; cursor: pointer; font-size: 0.85rem; color: #666; text-align: left; transition: all 0.2s; }}
        .side-nav-btn:hover {{ background: #f5f5f5; color: #333; }}
        .side-nav-btn.active {{ background: #e6f7ff; color: #1890ff; font-weight: 500; }}
        .main-content {{ flex: 1; padding: 20px; overflow: auto; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .training-tab-content {{ display: none; }}
        .training-tab-content.active {{ display: block; }}
        .inference-sub-content {{ display: none; }}
        .inference-sub-content.active {{ display: block; }}
        /* Log viewer styles */
        .log-layout {{ display: flex; gap: 16px; height: calc(100vh - 220px); }}
        .log-sidebar {{ width: 240px; flex-shrink: 0; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; overflow-y: auto; }}
        .log-sidebar h3 {{ font-size: 0.9rem; color: #333; margin-bottom: 12px; font-weight: 600; }}
        .log-file-item {{ padding: 10px 12px; margin-bottom: 6px; background: #fafafa; border: 1px solid #e8e8e8; border-radius: 6px; cursor: pointer; transition: all 0.2s; }}
        .log-file-item:hover {{ background: #e6f7ff; border-color: #1890ff; }}
        .log-file-item.active {{ background: #1890ff; border-color: #1890ff; color: #fff; }}
        .log-file-name {{ font-size: 0.85rem; font-weight: 500; margin-bottom: 4px; }}
        .log-file-size {{ font-size: 0.7rem; color: #999; }}
        .log-file-item.active .log-file-size {{ color: #bae7ff; }}
        .log-main {{ flex: 1; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; }}
        .log-header {{ padding: 12px 16px; border-bottom: 1px solid #e8e8e8; display: flex; justify-content: space-between; align-items: center; background: #fafafa; }}
        .log-title {{ font-size: 0.9rem; font-weight: 600; color: #333; }}
        .log-info {{ font-size: 0.75rem; color: #666; }}
        .log-content {{ flex: 1; padding: 16px; overflow: auto; font-family: 'Consolas', 'Monaco', 'Courier New', monospace; font-size: 0.8rem; line-height: 1.5; color: #333; white-space: pre-wrap; word-break: break-all; margin: 0; background: #fafafa; }}
        .log-clear-btn {{ padding: 4px 10px; font-size: 0.75rem; background: #fff; color: #ff4d4f; border: 1px solid #ff4d4f; border-radius: 4px; cursor: pointer; }}
        .log-clear-btn:hover {{ background: #ff4d4f; color: #fff; }}
        .no-logs {{ color: #999; text-align: center; padding: 40px; font-size: 0.9rem; }}
        /* Debug page styles */
        .debug-layout {{ display: flex; gap: 20px; height: calc(100vh - 140px); }}
        .debug-sidebar {{ width: 280px; flex-shrink: 0; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; }}
        .debug-sidebar h3 {{ font-size: 0.9rem; color: #333; margin-bottom: 16px; font-weight: 600; }}
        .debug-main {{ flex: 1; display: flex; flex-direction: column; gap: 16px; overflow: hidden; min-width: 0; }}
        .debug-content {{ flex: 1; display: flex; gap: 16px; overflow: hidden; }}
        .debug-result {{ flex: 1; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; overflow: auto; }}
        .debug-result h3 {{ font-size: 0.9rem; color: #333; margin-bottom: 12px; font-weight: 600; }}
        .minimap {{ width: 80px; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 8px; overflow-y: auto; flex-shrink: 0; }}
        .minimap-item {{ padding: 6px 8px; font-size: 0.7rem; color: #666; cursor: pointer; border-radius: 4px; margin-bottom: 2px; text-align: center; transition: all 0.2s; }}
        .minimap-item:hover {{ background: #e6f7ff; color: #1890ff; }}
        .minimap-item.active {{ background: #1890ff; color: #fff; }}
        
        
        .debug-input-card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; }}
        .debug-input {{ width: 100%; padding: 12px; border: 1px solid #d9d9d9; border-radius: 8px; font-size: 0.9rem; resize: vertical; min-height: 60px; }}
        .debug-input:focus {{ border-color: #1890ff; outline: none; }}
        .debug-btn {{ padding: 12px 24px; background: #1890ff; color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 0.9rem; }}
        .debug-btn:hover {{ background: #40a9ff; }}
        .debug-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        .debug-filter-btn {{ flex: 1; padding: 8px 12px; background: #fff; color: #666; border: 1px solid #d9d9d9; border-radius: 6px; cursor: pointer; font-size: 0.8rem; transition: all 0.2s; }}
        .debug-filter-btn:hover {{ border-color: #1890ff; color: #1890ff; }}
        .debug-filter-btn.active {{ background: #1890ff; color: #fff; border-color: #1890ff; }}
        .output-inline {{ background: #f6ffed; border: 1px solid #b7eb8f; border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }}
        .output-inline .label {{ color: #52c41a; font-size: 0.75rem; font-weight: 500; }}
        .output-inline .text {{ font-size: 0.85rem; color: #333; }}
        .output-inline .generated {{ background: #fffbe6; padding: 1px 4px; border-radius: 2px; }}
        .layer-card {{ background: #fafafa; border: 1px solid #e8e8e8; border-radius: 8px; margin-bottom: 10px; overflow: hidden; }}
        .layer-header {{ background: #fff; color: #333; padding: 8px 16px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e8e8e8; }}
        .layer-name {{ font-weight: 600; font-size: 0.8rem; }}
        .layer-body {{ padding: 10px 14px; }}
        .token-row {{ display: flex; align-items: center; padding: 6px 0; border-bottom: 1px solid #f0f0f0; }}
        .token-row:last-child {{ border-bottom: none; }}
        .input-token {{ background: #1890ff; color: #fff; padding: 6px 16px; border-radius: 4px; font-size: 0.8rem; font-weight: 500; min-width: 60px; text-align: center; }}
        .arrow {{ color: #999; margin: 0 10px; font-size: 1.2rem; }}
        .similar-tokens {{ display: flex; flex-wrap: wrap; gap: 5px; }}
        .similar-token {{ background: #f0f0f0; padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; display: flex; align-items: center; gap: 4px; }}
        .similar-token.rank-1 {{ background: linear-gradient(135deg, #1890ff 0%, #40a9ff 100%); border: 1px solid #1890ff; color: #fff; }}
        .similar-token.rank-1 .score {{ color: #e6f7ff; }}
        .similar-token.rank-2 {{ background: linear-gradient(135deg, #91d5ff 0%, #bae7ff 100%); border: 1px solid #69c0ff; color: #0050b3; }}
        .similar-token.rank-2 .score {{ color: #0050b3; }}
        .similar-token.rank-3 {{ background: linear-gradient(135deg, #d6e4ff 0%, #e6f7ff 100%); border: 1px solid #91d5ff; color: #1890ff; }}
        .similar-token.rank-3 .score {{ color: #1890ff; }}
        .similar-token .score {{ color: #1890ff; font-size: 1.2rem; }}
        .no-logs {{ color: #999; text-align: center; padding: 40px; font-size: 0.9rem; }}
        .train-status {{ display: flex; align-items: center; gap: 8px; padding: 8px 16px; border-radius: 4px; font-size: 0.85rem; }}
        .train-status .dot {{ width: 8px; height: 8px; border-radius: 50%; }}
        .train-status.running {{ background: #f6ffed; color: #52c41a; }}
        .train-status.running .dot {{ background: #52c41a; animation: pulse 2s infinite; }}
        .train-status.stopped {{ background: #fff2f0; color: #ff4d4f; }}
        .train-status.stopped .dot {{ background: #ff4d4f; }}
        @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
        .main {{ padding: 20px; max-width: 1800px; margin: 0 auto; }}
        .page {{ display: none; }}
        .page.active {{ display: block; }}
        /* Dashboard styles */
        .dashboard-layout {{ display: flex; gap: 20px; }}
        .dashboard-content {{ flex: 1; min-width: 0; }}
        .dashboard-sidebar {{ width: 280px; flex-shrink: 0; }}
        .charts-area {{ flex: 1; min-width: 0; }}
        .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 16px; }}
        .sidebar {{ width: 280px; flex-shrink: 0; }}
        .metrics-card, .gpu-card, .samples-card, .control-card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 16px; }}
        .metrics-header, .gpu-header, .samples-header {{ display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; cursor: pointer; user-select: none; }}
        .metrics-header:hover, .gpu-header:hover, .samples-header:hover {{ background: #fafafa; border-radius: 8px 8px 0 0; }}
        .metrics-header h3, .gpu-header h3, .samples-header h3 {{ font-size: 0.85rem; color: #666; font-weight: 500; margin: 0; }}
        .metrics-toggle, .gpu-toggle, .samples-toggle {{ font-size: 12px; color: #999; transition: transform 0.3s; }}
        .metrics-toggle.collapsed, .gpu-toggle.collapsed, .samples-toggle.collapsed {{ transform: rotate(-90deg); }}
        .metrics-content, .gpu-content, .samples-content {{ border-top: 1px solid #f0f0f0; padding: 8px 16px; }}
        .metrics-content.collapsed, .gpu-content.collapsed, .samples-content.collapsed {{ display: none; }}
        .metric {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #f0f0f0; }}
        .metric:last-child {{ border-bottom: none; }}
        .metric-title {{ font-size: 0.8rem; color: #666; }}
        .metric-value {{ font-size: 0.9rem; font-weight: 600; color: #333; font-variant-numeric: tabular-nums; }}
        .card {{ background: #fff; border-radius: 8px; padding: 8px; border: 1px solid #e0e0e0; }}
        .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; padding: 0 4px; }}
        .card-title {{ color: #333; font-size: 0.9rem; font-weight: 500; }}
        .copy-btn {{ background: #fff; border: 1px solid #d9d9d9; color: #666; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 0.75rem; display: flex; align-items: center; gap: 4px; }}
        .copy-btn:hover {{ border-color: #1890ff; color: #1890ff; }}
        .copy-btn.copied {{ background: #f6ffed; border-color: #52c41a; color: #52c41a; }}
        .copy-btn svg {{ width: 12px; height: 12px; }}
        .chart-container {{ background: #fff; border-radius: 4px; overflow: hidden; }}
        .chart-container img {{ width: 100%; height: auto; display: block; }}
        .loading {{ display: flex; align-items: center; justify-content: center; height: 200px; color: #999; }}
        .spinner {{ width: 24px; height: 24px; border: 2px solid #e0e0e0; border-top-color: #1890ff; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 8px; }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        .toast {{ position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%) translateY(100px); background: #52c41a; color: #fff; padding: 10px 20px; border-radius: 4px; font-size: 0.85rem; opacity: 0; transition: all 0.3s ease; z-index: 1000; }}
        .toast.show {{ transform: translateX(-50%) translateY(0); opacity: 1; }}
        .gpu-item {{ padding: 8px 0; border-bottom: 1px solid #f0f0f0; }}
        .gpu-item:last-child {{ border-bottom: none; }}
        .gpu-name {{ font-size: 0.8rem; color: #1890ff; font-weight: 500; margin-bottom: 6px; }}
        .gpu-stats {{ display: flex; flex-wrap: wrap; gap: 8px; }}
        .gpu-stat {{ flex: 1; min-width: 80px; }}
        .gpu-stat-label {{ font-size: 0.7rem; color: #999; }}
        .gpu-stat-value {{ font-size: 0.85rem; font-weight: 600; color: #333; }}
        .gpu-bar {{ height: 4px; background: #f0f0f0; border-radius: 2px; margin-top: 2px; }}
        .gpu-bar-fill {{ height: 100%; border-radius: 2px; transition: width 0.3s; }}
        .gpu-bar-fill.util {{ background: linear-gradient(90deg, #52c41a, #faad14, #f5222d); }}
        .gpu-bar-fill.mem {{ background: linear-gradient(90deg, #1890ff, #722ed1); }}
        .no-gpu {{ padding: 16px; text-align: center; color: #999; font-size: 0.8rem; }}
        .samples-content {{ max-height: 280px; overflow-y: auto; }}
        .sample-item {{ padding: 10px 16px; border-bottom: 1px solid #f0f0f0; }}
        .sample-item:last-child {{ border-bottom: none; }}
        .sample-input {{ font-size: 0.8rem; color: #1890ff; font-weight: 500; margin-bottom: 4px; }}
        .sample-output {{ font-size: 0.8rem; color: #333; line-height: 1.4; word-break: break-all; }}
        .no-samples {{ padding: 20px; text-align: center; color: #999; font-size: 0.8rem; }}
        .control-card {{ padding: 16px; }}
        .control-card h3 {{ font-size: 0.85rem; color: #666; font-weight: 500; margin: 0 0 12px 0; }}
        .control-btns {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .ctrl-btn {{ flex: 1; min-width: 80px; padding: 10px 12px; border-radius: 4px; cursor: pointer; font-size: 0.8rem; display: flex; align-items: center; justify-content: center; gap: 4px; border: none; }}
        .ratio-slider {{ width: 100%; height: 6px; border-radius: 3px; background: #f0f0f0; outline: none; -webkit-appearance: none; }}
        .ratio-slider::-webkit-slider-thumb {{ -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%; background: #1890ff; cursor: pointer; transition: background 0.2s; }}
        .ratio-slider::-webkit-slider-thumb:hover {{ background: #40a9ff; }}
        .ratio-slider::-moz-range-thumb {{ width: 16px; height: 16px; border-radius: 50%; background: #1890ff; cursor: pointer; border: none; transition: background 0.2s; }}
        .ratio-slider::-moz-range-thumb:hover {{ background: #40a9ff; }}
        .ctrl-btn.start {{ background: #52c41a; color: #fff; }}
        .ctrl-btn.start:hover {{ background: #73d13d; }}
        .ctrl-btn.stop {{ background: #ff4d4f; color: #fff; }}
        .ctrl-btn.stop:hover {{ background: #ff7875; }}
        .ctrl-btn.save {{ background: #1890ff; color: #fff; }}
        .ctrl-btn.save:hover {{ background: #40a9ff; }}
        .ctrl-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        .ctrl-btn svg {{ width: 16px; height: 16px; }}
        /* Inference page styles */
        .infer-layout {{ display: flex; gap: 20px; height: calc(100vh - 140px); }}
        .infer-sidebar {{ width: 280px; flex-shrink: 0; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; }}
        .infer-sidebar h3 {{ font-size: 0.9rem; color: #333; margin-bottom: 16px; font-weight: 600; }}
        .param-group {{ margin-bottom: 16px; }}
        .param-label {{ font-size: 0.8rem; color: #666; margin-bottom: 6px; display: flex; justify-content: space-between; }}
        .param-value {{ color: #1890ff; font-weight: 500; }}
        .param-input {{ width: 100%; padding: 8px; border: 1px solid #d9d9d9; border-radius: 4px; font-size: 0.85rem; }}
        .param-input:focus {{ border-color: #1890ff; outline: none; }}
        .param-slider {{ width: 100%; }}
        .weight-select {{ width: 100%; padding: 8px; border: 1px solid #d9d9d9; border-radius: 4px; font-size: 0.85rem; }}
        .infer-main {{ flex: 1; display: flex; flex-direction: column; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; }}
        .chat-area {{ flex: 1; padding: 16px; overflow-y: auto; }}
        .chat-msg {{ margin-bottom: 16px; display: flex; align-items: flex-start; gap: 10px; }}
        .chat-msg.user {{ flex-direction: row-reverse; }}
        .chat-avatar {{ width: 36px; height: 36px; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; }}
        .chat-msg.user .chat-avatar {{ background: #e6f7ff; }}
        .chat-msg.assistant .chat-avatar {{ background: #fff; border: 1px solid #e0e0e0; }}
        .chat-msg.assistant .chat-avatar img {{ width: 24px; height: 24px; object-fit: contain; }}
        .chat-msg .bubble {{ max-width: 80%; padding: 10px 14px; border-radius: 12px; font-size: 0.9rem; line-height: 1.5; white-space: pre-wrap; word-break: break-all; }}
        .chat-msg.user .bubble {{ background: #1890ff; color: #fff; border-bottom-right-radius: 4px; }}
        .chat-msg.assistant .bubble {{ background: #f0f0f0; color: #333; border-bottom-left-radius: 4px; }}
        .chat-input-area {{ border-top: 1px solid #e0e0e0; padding: 16px; display: flex; gap: 12px; }}
        .chat-input {{ flex: 1; padding: 12px; border: 1px solid #d9d9d9; border-radius: 8px; font-size: 0.9rem; resize: none; }}
        .chat-input:focus {{ border-color: #1890ff; outline: none; }}
        .send-btn {{ padding: 12px 24px; background: #1890ff; color: #fff; border: none; border-radius: 8px; cursor: pointer; font-size: 0.9rem; }}
        .send-btn:hover {{ background: #40a9ff; }}
        .send-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        .empty-chat {{ display: flex; align-items: center; justify-content: center; height: 100%; color: #999; font-size: 0.9rem; }}
        .bubble.thinking {{ color: #666; }}
        .bubble.thinking .dots::after {{ content: ''; animation: dots 1.5s steps(4, end) infinite; }}
        @keyframes dots {{ 0% {{ content: ''; }} 25% {{ content: '.'; }} 50% {{ content: '..'; }} 75% {{ content: '...'; }} }}
        @media (max-width: 800px) {{ .layout, .infer-layout {{ flex-direction: column; }} .sidebar, .infer-sidebar {{ width: 100%; }} }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-left">
            {logo_img_html}
            <h1>Scholar Dashboard</h1>
        </div>
        <div class="train-status stopped" id="train-status">
            <span class="dot"></span>
            <span id="train-status-text">Stopped</span>
        </div>
    </div>
    <div class="app-layout">
        <!-- Side Navigation -->
        <div class="side-nav">
            <div class="side-nav-title">Training</div>
            <button class="side-nav-btn active" onclick="switchSubTab('training', 'training-tab-training', this)">Training</button>
            <button class="side-nav-btn" onclick="switchSubTab('training', 'training-tab-metrics', this)">Metrics</button>
            <button class="side-nav-btn" onclick="switchSubTab('training', 'training-tab-log', this)">Log</button>
            
            <div class="side-nav-title" style="margin-top: 24px;">Inference</div>
            <button class="side-nav-btn" onclick="switchSubTab('inference', 'inference-sub-chat', this)">Chat</button>
            <button class="side-nav-btn" onclick="switchSubTab('inference', 'inference-sub-debug', this)">Debug</button>
        </div>
        <!-- Main Content Area -->
        <div class="main-content">
            <!-- Training Tab Content -->
            <div class="tab-content active" id="tab-training">
                <div class="training-tab-content active" id="training-tab-training">
                    <div class="dashboard-layout">
                        <div class="dashboard-content"><div class="charts-grid">{cards}</div></div>
                        <div class="dashboard-sidebar">
                            <div class="control-card">
                                <h3>Training Control</h3>
                                <div class="control-btns">
                                    <button class="ctrl-btn start" id="btn-start" onclick="startTrain()">
                                        <svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>
                                        Start
                                    </button>
                                    <button class="ctrl-btn resume" id="btn-resume" onclick="resumeTrain()" style="background: #faad14; border-color: #faad14;">
                                        <svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>
                                        Resume
                                    </button>
                                    <button class="ctrl-btn stop" id="btn-stop" onclick="stopTrain()" disabled>
                                        <svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12"/></svg>
                                        Stop
                                    </button>
                                </div>
                                <div class="ratio-control" style="margin-top: 16px; padding-top: 12px; border-top: 1px solid #f0f0f0;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                        <h3 style="font-size: 0.8rem; color: #666; margin: 0;">Data Ratio</h3>
                                        <span class="ratio-value" id="ratio-value" style="font-size: 0.85rem; font-weight: 600; color: #1890ff;">100%</span>
                                    </div>
                                    <input type="range" class="ratio-slider" id="ratio-slider" min="10" max="100" step="10" value="100" oninput="handleRatioChange(this.value)">
                                    <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #999; margin-top: 4px;">
                                        <span>10%</span>
                                        <span>100%</span>
                                    </div>
                                </div>
                                <div style="margin-top: 12px;">
                                    <button class="ctrl-btn save" onclick="saveAllCharts()" style="width: 100%;">
                                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                                            <polyline points="7 10 12 15 17 10"/>
                                            <line x1="12" y1="15" x2="12" y2="3"/>
                                        </svg>
                                        Save All Charts
                                    </button>
                                </div>
                            </div>
                            <div class="gpu-card">
                                <div class="gpu-header" onclick="toggleSection('gpu')">
                                    <h3>GPU Monitor</h3>
                                    <span class="gpu-toggle" id="gpu-toggle">▼</span>
                                </div>
                                <div class="gpu-content" id="gpu-content"></div>
                            </div>
                            <div class="samples-card">
                                <div class="samples-header" onclick="toggleSection('processes')">
                                    <h3>Training Processes</h3>
                                    <span class="samples-toggle" id="processes-toggle">▼</span>
                                </div>
                                <div class="samples-content" id="processes-content"></div>
                            </div>
                            <div class="samples-card">
                                <div class="samples-header" onclick="toggleSection('samples')">
                                    <h3>Inference Samples</h3>
                                    <span class="samples-toggle" id="samples-toggle">▼</span>
                                </div>
                                <div class="samples-content" id="samples-content"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="training-tab-content" id="training-tab-metrics">
                    <div class="dashboard-layout">
                        <div class="dashboard-content">
                            <div class="metrics-card" style="margin-bottom: 0;">
                                <div class="metrics-header">
                                    <h3>All Metrics</h3>
                                </div>
                                <div class="metrics-content" id="metrics-container" style="border-top: none; padding: 16px;"></div>
                            </div>
                        </div>
                        <div class="dashboard-sidebar">
                            <div class="control-card">
                                <h3>Training Control</h3>
                                <div class="control-btns">
                                    <button class="ctrl-btn start" id="btn-start-metrics" onclick="startTrain()">
                                        <svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>
                                        Start
                                    </button>
                                    <button class="ctrl-btn resume" id="btn-resume-metrics" onclick="resumeTrain()" style="background: #faad14; border-color: #faad14;">
                                        <svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>
                                        Resume
                                    </button>
                                    <button class="ctrl-btn stop" id="btn-stop-metrics" onclick="stopTrain()" disabled>
                                        <svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12"/></svg>
                                        Stop
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="training-tab-content" id="training-tab-log">
                    <div class="log-layout">
                        <div class="log-sidebar">
                            <h3>Log Files</h3>
                            <div class="log-file-list" id="log-file-list"></div>
                        </div>
                        <div class="log-main">
                            <div class="log-header">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span class="log-title" id="log-title">Select a log file</span>
                                    <button class="log-clear-btn" id="log-clear-btn" onclick="clearLogFile()" style="display: none;">Clear</button>
                                </div>
                                <span class="log-info" id="log-info"></span>
                            </div>
                            <pre class="log-content" id="log-content">Click on a log file to view its content</pre>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Inference Tab Content -->
            <div class="tab-content" id="tab-inference">
                <div class="inference-sub-content" id="inference-sub-chat">
                    <div class="infer-layout">
                        <div class="infer-sidebar">
                            <h3>Parameters</h3>
                            <div class="param-group">
                                <div class="param-label">
                                    <span>Temperature</span>
                                    <span class="param-value" id="temp-value">0.7</span>
                                </div>
                                <input type="range" class="param-slider" id="temperature" min="0.1" max="2.0" step="0.1" value="0.7" oninput="updateParamDisplay('temp', this.value)">
                            </div>
                            <div class="param-group">
                                <div class="param-label">
                                    <span>Top P</span>
                                    <span class="param-value" id="topp-value">0.85</span>
                                </div>
                                <input type="range" class="param-slider" id="topP" min="0.1" max="1.0" step="0.05" value="0.85" oninput="updateParamDisplay('topp', this.value)">
                            </div>
                            <div class="param-group">
                                <div class="param-label">
                                    <span>Max Tokens</span>
                                    <span class="param-value" id="tokens-value">50</span>
                                </div>
                                <input type="range" class="param-slider" id="maxTokens" min="10" max="200" step="10" value="50" oninput="updateParamDisplay('tokens', this.value)">
                            </div>
                            <div class="param-group">
                                <div class="param-label"><span>Weight File</span></div>
                                <select class="weight-select" id="weight-select">
                                    <option value="scholar_best.bin">scholar_best.bin</option>
                                    <option value="scholar_last.bin">scholar_last.bin</option>
                                </select>
                            </div>
                        </div>
                        <div class="infer-main">
                            <div class="chat-area" id="chat-area">
                                <div class="empty-chat">Enter a sentence to start inference</div>
                            </div>
                            <div class="chat-input-area">
                                <textarea class="chat-input" id="chat-input" placeholder="Enter a sentence to continue..." rows="2" onkeydown="handleChatKey(event)"></textarea>
                                <button class="send-btn" id="send-btn" onclick="sendInfer()">Send</button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="inference-sub-content" id="inference-sub-debug">
                    <div class="debug-layout">
                        <div class="debug-sidebar">
                            <h3>Debug Parameters</h3>
                            <div class="param-group">
                                <div class="param-label">
                                    <span>Temperature</span>
                                    <span class="param-value" id="debug-temp-value">0.7</span>
                                </div>
                                <input type="range" class="param-slider" id="debug-temperature" min="0.1" max="2.0" step="0.1" value="0.7" oninput="updateParamDisplay('debug-temp', this.value)">
                            </div>
                            <div class="param-group">
                                <div class="param-label">
                                    <span>Top P</span>
                                    <span class="param-value" id="debug-topp-value">0.85</span>
                                </div>
                                <input type="range" class="param-slider" id="debug-topP" min="0.1" max="1.0" step="0.05" value="0.85" oninput="updateParamDisplay('debug-topp', this.value)">
                            </div>
                            <div class="param-group">
                                <div class="param-label">
                                    <span>Max Tokens</span>
                                    <span class="param-value" id="debug-tokens-value">1</span>
                                </div>
                                <input type="range" class="param-slider" id="debug-maxTokens" min="1" max="50" step="1" value="1" oninput="updateParamDisplay('debug-tokens', this.value)">
                            </div>
                            <div class="param-group">
                                <div class="param-label"><span>Weight File</span></div>
                                <select class="weight-select" id="debug-weight-select">
                                    <option value="scholar_best.bin">scholar_best.bin</option>
                                    <option value="scholar_last.bin">scholar_last.bin</option>
                                </select>
                            </div>
                            <div class="param-group" style="margin-top: 16px;">
                                <div class="param-label"><span>Layer Filter</span></div>
                                <div style="display: flex; gap: 8px; margin-top: 8px;">
                                    <button class="debug-filter-btn active" id="filter-all" onclick="setLayerFilter('all')">All</button>
                                    <button class="debug-filter-btn" id="filter-attn" onclick="setLayerFilter('attn')">Attn</button>
                                    <button class="debug-filter-btn" id="filter-ffn" onclick="setLayerFilter('ffn')">FFN</button>
                                </div>
                            </div>
                            <div class="param-group">
                                <div class="param-label">
                                    <span>Mask Tokens</span>
                                    <span class="param-value" id="debug-mask-value">0</span>
                                </div>
                                <input type="range" class="param-slider" id="debug-mask" min="0" max="0" step="1" value="0" oninput="updateMask(this.value)">
                            </div>
                            <div class="debug-input-card" style="margin-top: 16px; padding: 12px;">
                                <h3 style="font-size: 0.85rem; margin-bottom: 8px;">Input Sentence</h3>
                                <textarea class="debug-input" id="debug-input" placeholder="Enter a sentence to debug..." rows="2" style="min-height: 50px;"></textarea>
                                <button class="debug-btn" id="debug-run-btn" onclick="runDebug()" style="width: 100%; margin-top: 12px;">
                                    Run Debug Inference
                                </button>
                                <button class="debug-btn" onclick="copyAsText()" style="width: 100%; margin-top: 8px; background: #52c41a;">
                                    Copy as Text
                                </button>
                            </div>
                        </div>
                        <div class="debug-main">
                            <div class="debug-content">
                                <div class="debug-result">
                                    <div class="output-inline" id="debug-output-inline" style="display: none;">
                                        <span class="label">Output:</span>
                                        <span class="text" id="debug-output-text"></span>
                                    </div>
                                    <h3>Scholar Model Internals</h3>
                                    <div id="debug-layer-container">
                                        <div class="no-logs">Run debug inference to see how tokens evolve through each layer</div>
                                    </div>
                                </div>
                                <div class="minimap" id="layer-minimap"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="toast" id="toast"></div>
    <script>
        let lastHash = '';
        const refreshInterval = {refresh_interval} * 1000;
        let chatHistory = [];
        let isInferring = false;
        let isDebugRunning = false;
        let currentLayerFilter = 'all';
        let currentMask = 0;
        let cachedLayerData = null;
        let currentLogPath = null;
        let currentRatio = 100;
        let ratioFetchTimeout = null;
        
        function handleRatioChange(value) {{
            currentRatio = parseInt(value);
            document.getElementById('ratio-value').textContent = value + '%';
            
            // Debounce: wait 300ms after last change before fetching
            if (ratioFetchTimeout) {{
                clearTimeout(ratioFetchTimeout);
            }}
            ratioFetchTimeout = setTimeout(() => {{
                fetchCharts();
            }}, 300);
        }}
        
        function switchSubTab(mainTab, contentId, btnElement) {{
            // Update side nav buttons
            document.querySelectorAll('.side-nav-btn').forEach(btn => btn.classList.remove('active'));
            btnElement.classList.add('active');
            
            // Update main tab content area
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            document.getElementById('tab-' + mainTab).classList.add('active');
            
            // Update sub content visibility within the main tab
            const mainTabContent = document.getElementById('tab-' + mainTab);
            mainTabContent.querySelectorAll('.training-tab-content, .inference-sub-content').forEach(content => {{
                content.classList.remove('active');
            }});
            document.getElementById(contentId).classList.add('active');
            
            // Load logs if switching to log tab
            if (contentId === 'training-tab-log') {{
                loadLogFiles();
            }}
        }}
        
        async function loadLogFiles() {{
            try {{
                const response = await fetch('/api/logs');
                const data = await response.json();
                const container = document.getElementById('log-file-list');
                
                if (!data.logs || data.logs.length === 0) {{
                    container.innerHTML = '<div class="no-logs">No log files found</div>';
                    return;
                }}
                
                container.innerHTML = data.logs.map(log => `
                    <div class="log-file-item" onclick="loadLog('${{log.name}}', this)">
                        <div class="log-file-name">${{log.name}}</div>
                        <div class="log-file-size">${{(log.size / 1024).toFixed(1)}} KB</div>
                    </div>
                `).join('');
            }} catch (error) {{
                console.error('Load logs error:', error);
            }}
        }}
        
        async function loadLog(name, element) {{
            // Update active state
            document.querySelectorAll('.log-file-item').forEach(item => item.classList.remove('active'));
            if (element) element.classList.add('active');
            
            currentLogPath = name;
            document.getElementById('log-title').textContent = name;
            document.getElementById('log-clear-btn').style.display = 'inline-block';
            document.getElementById('log-content').innerHTML = '<div class="loading"><div class="spinner"></div>Loading...</div>';
            
            try {{
                const response = await fetch('/api/logs/read?name=' + encodeURIComponent(name));
                const data = await response.json();
                
                if (data.error) {{
                    document.getElementById('log-content').textContent = 'Error: ' + data.error;
                    document.getElementById('log-info').textContent = '';
                    return;
                }}
                
                const content = document.getElementById('log-content');
                content.textContent = data.content;
                
                let infoText = `${{data.displayed_lines}} lines`;
                if (data.truncated) {{
                    infoText += ` (showing last ${{data.displayed_lines}} of ${{data.total_lines}})`;
                }}
                document.getElementById('log-info').textContent = infoText;
                
                // Scroll to bottom
                setTimeout(() => {{
                    content.scrollTop = content.scrollHeight;
                }}, 100);
            }} catch (error) {{
                document.getElementById('log-content').textContent = 'Failed to load log: ' + error;
                document.getElementById('log-info').textContent = '';
            }}
        }}
        
        async function clearLogFile() {{
            if (!currentLogPath) {{
                showToast('No log file selected');
                return;
            }}
            
            if (!confirm('Are you sure you want to clear ' + currentLogPath + '?')) {{
                return;
            }}
            
            try {{
                const response = await fetch('/api/logs/clear?name=' + encodeURIComponent(currentLogPath), {{
                    method: 'POST'
                }});
                const data = await response.json();
                
                if (data.error) {{
                    showToast('Error: ' + data.error);
                }} else {{
                    showToast('Log cleared');
                    // Reload the log content
                    loadLog(currentLogPath, null);
                }}
            }} catch (error) {{
                showToast('Failed to clear log: ' + error);
            }}
        }}
        
        function updateMask(value) {{
            currentMask = parseInt(value);
            document.getElementById('debug-mask-value').textContent = value;
            fetchAndRenderLayers();
        }}
        
        function setLayerFilter(filter) {{
            currentLayerFilter = filter;
            document.querySelectorAll('.debug-filter-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById('filter-' + filter).classList.add('active');
            fetchAndRenderLayers();
        }}
        
        function copyAsText() {{
            if (!cachedLayerData) {{
                showToast('No data to copy');
                return;
            }}
            
            let text = '';
            cachedLayerData.forEach((layerData) => {{
                const layerName = layerData.layer;
                const isFinalNorm = layerName.toLowerCase().includes('finalnorm');
                const isAttn = layerName.toLowerCase().includes('attn');
                const isFfn = layerName.toLowerCase().includes('ffn');
                
                if (currentLayerFilter !== 'all' && !isFinalNorm) {{
                    if (currentLayerFilter === 'attn' && !isAttn) return;
                    if (currentLayerFilter === 'ffn' && !isFfn) return;
                }}
                
                text += layerName + ':\\n';
                const tokens = layerData.tokens;
                const showCount = Math.max(1, tokens.length - currentMask);
                tokens.slice(0, showCount).forEach((tokenData) => {{
                    const similar = tokenData.similar.map(s => {{
                        const match = s.match(/(.+)\\((.+)\\)/);
                        return match ? match[1] : s;
                    }}).join(',');
                    text += tokenData.inputToken + ' -> ' + similar + '\\n';
                }});
                text += '\\n';
            }});
            
            navigator.clipboard.writeText(text.trim()).then(() => {{
                showToast('Copied to clipboard');
            }}).catch(() => {{
                showToast('Failed to copy');
            }});
        }}
        
        function showTrainingTab(tab) {{
            // Update tab group buttons - find buttons within training tab
            const trainingTab = document.getElementById('tab-training');
            trainingTab.querySelectorAll('.tab-group-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            // Update content visibility
            trainingTab.querySelectorAll('.training-tab-content').forEach(content => content.classList.remove('active'));
            document.getElementById('training-tab-' + tab).classList.add('active');
            
            // Load logs if switching to log tab
            if (tab === 'log') {{
                loadLogFiles();
            }}
        }}
        
        function toggleSection(name) {{
            const content = document.getElementById(name + '-content') || document.getElementById(name + '-container');
            const toggle = document.getElementById(name + '-toggle');
            content.classList.toggle('collapsed');
            toggle.classList.toggle('collapsed');
        }}
        
        function showToast(msg) {{
            const toast = document.getElementById('toast');
            toast.textContent = msg;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 2000);
        }}
        
        function updateParamDisplay(name, value) {{
            document.getElementById(name + '-value').textContent = value;
        }}
        
        function renderGpuStats(gpus) {{
            const container = document.getElementById('gpu-content');
            if (!gpus || gpus.length === 0) {{
                container.innerHTML = '<div class="no-gpu">No GPU detected</div>';
                return;
            }}
            container.innerHTML = gpus.map(gpu => `
                <div class="gpu-item">
                    <div class="gpu-name">GPU ${{gpu.index}}: ${{gpu.name}}</div>
                    <div class="gpu-stats">
                        <div class="gpu-stat">
                            <div class="gpu-stat-label">Utilization</div>
                            <div class="gpu-stat-value">${{gpu.utilization}}%</div>
                            <div class="gpu-bar"><div class="gpu-bar-fill util" style="width: ${{gpu.utilization}}%"></div></div>
                        </div>
                        <div class="gpu-stat">
                            <div class="gpu-stat-label">Memory</div>
                            <div class="gpu-stat-value">${{gpu.memory_used}}/${{gpu.memory_total}} MB</div>
                            <div class="gpu-bar"><div class="gpu-bar-fill mem" style="width: ${{(gpu.memory_used/gpu.memory_total*100).toFixed(1)}}%"></div></div>
                        </div>
                        <div class="gpu-stat">
                            <div class="gpu-stat-label">Temp</div>
                            <div class="gpu-stat-value">${{gpu.temperature}}°C</div>
                        </div>
                    </div>
                </div>
            `).join('');
        }}
        
        async function fetchTrainingProcesses() {{
            try {{
                const response = await fetch('/api/train/processes');
                const processes = await response.json();
                const container = document.getElementById('processes-content');
                
                if (!processes || processes.length === 0) {{
                    container.innerHTML = '<div class="no-samples">No training processes running</div>';
                    return;
                }}
                
                container.innerHTML = processes.map(proc => `
                    <div class="sample-item">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div class="sample-input" style="font-family: monospace; font-size: 0.75rem; flex: 1;">
                                PID: ${{proc.pid}}
                            </div>
                            <button onclick="killProcess(${{proc.pid}})" style="color: white; border: none; padding: 4px 8px; cursor: pointer; border-radius: 4px; font-size: 0.7rem;">
                                X
                            </button>
                        </div>
                        <div class="sample-output" style="font-family: monospace; font-size: 0.7rem; word-break: break-all; margin-top: 4px;">
                            ${{proc.cmdline}}
                        </div>
                    </div>
                `).join('');
            }} catch (error) {{ console.error('Process fetch error:', error); }}
        }}
        
        async function fetchCharts() {{
            try {{
                const response = await fetch('/api/charts?ratio=' + currentRatio);
                const data = await response.json();
                if (data.metrics) {{
                    document.getElementById('metrics-container').innerHTML = data.metrics.map(m => 
                        `<div class="metric"><span class="metric-title">${{m.title}}</span><span class="metric-value">${{m.value}}</span></div>`
                    ).join('');
                }}
                renderGpuStats(data.gpu_stats);
                fetchTrainingProcesses();
                if (data.samples) {{
                    const container = document.getElementById('samples-content');
                    container.innerHTML = data.samples.length > 0 
                        ? data.samples.map(s => `<div class="sample-item"><div class="sample-input">${{s.input}}</div><div class="sample-output">${{s.output}}</div></div>`).join('')
                        : '<div class="no-samples">No samples yet</div>';
                }}
                if (data.hash !== lastHash) {{
                    lastHash = data.hash;
                    data.charts.forEach((imgData, index) => {{
                        const container = document.getElementById(`chart-${{index}}`);
                        if (container) container.innerHTML = `<img src="data:image/png;base64,${{imgData}}" />`;
                    }});
                }}
            }} catch (error) {{ console.error('Fetch error:', error); }}
        }}
        
        async function fetchTrainStatus() {{
            try {{
                // First check for existing training processes
                const procRes = await fetch('/api/train/processes');
                const processes = await procRes.json();
                
                const el = document.getElementById('train-status');
                const text = document.getElementById('train-status-text');
                
                // Get all button pairs
                const btnStart = document.getElementById('btn-start');
                const btnStop = document.getElementById('btn-stop');
                const btnResume = document.getElementById('btn-resume');
                const btnStartMetrics = document.getElementById('btn-start-metrics');
                const btnStopMetrics = document.getElementById('btn-stop-metrics');
                const btnResumeMetrics = document.getElementById('btn-resume-metrics');
                
                if (processes.length > 1) {{
                    // Multiple processes - error state
                    el.className = 'train-status running';
                    text.textContent = 'Error: Multiple Processes';
                    [btnStart, btnStartMetrics].forEach(b => b && (b.disabled = true));
                    [btnStop, btnStopMetrics].forEach(b => b && (b.disabled = false));
                    [btnResume, btnResumeMetrics].forEach(b => b && (b.disabled = true));
                }} else if (processes.length === 1) {{
                    // One process - running
                    const pid = processes[0].pid;
                    el.className = 'train-status running';
                    text.textContent = 'Running(PID:' + pid + ')';
                    [btnStart, btnStartMetrics].forEach(b => b && (b.disabled = true));
                    [btnStop, btnStopMetrics].forEach(b => b && (b.disabled = false));
                    [btnResume, btnResumeMetrics].forEach(b => b && (b.disabled = true));
                }} else {{
                    // No processes - stopped
                    el.className = 'train-status stopped';
                    text.textContent = 'Stopped';
                    [btnStart, btnStartMetrics].forEach(b => b && (b.disabled = false));
                    [btnStop, btnStopMetrics].forEach(b => b && (b.disabled = true));
                    [btnResume, btnResumeMetrics].forEach(b => b && (b.disabled = false));
                }}
            }} catch (error) {{ console.error('Status error:', error); }}
        }}
        
        async function startTrain() {{
            try {{
                const response = await fetch('/api/train/start', {{ method: 'POST' }});
                const data = await response.json();
                if (data.error) {{ showToast('Error: ' + data.error); return; }}
                showToast('Training started');
                fetchTrainStatus();
            }} catch (error) {{ showToast('Failed to start'); }}
        }}
        
        async function resumeTrain() {{
            try {{
                const response = await fetch('/api/train/start', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ resume: 'scholar_best.bin' }})
                }});
                const data = await response.json();
                if (data.error) {{ showToast('Error: ' + data.error); return; }}
                showToast('Training resumed from scholar_best.bin');
                fetchTrainStatus();
            }} catch (error) {{ showToast('Failed to resume'); }}
        }}
        
        async function killProcess(pid) {{
            if (!confirm('Are you sure you want to stop ' + pid + '?')) return;
            try {{
                const response = await fetch('/api/train/kill/' + pid, {{ method: 'POST' }});
                const data = await response.json();
                if (data.error) {{ showToast('Error: ' + data.error); return; }}
                showToast('Process killed');
                fetchTrainingProcesses();
                fetchTrainStatus();
            }} catch (error) {{ showToast('Failed to kill process'); }}
        }}
        
        async function stopTrain() {{
            try {{
                // Get all training processes
                const processesRes = await fetch('/api/train/processes');
                const processes = await processesRes.json();
                
                if (!processes || processes.length === 0) {{
                    // No external processes, try to stop internal one
                    const response = await fetch('/api/train/stop', {{ method: 'POST' }});
                    const data = await response.json();
                    if (data.error) {{ showToast('No training process found'); return; }}
                    showToast('Training stopped');
                    fetchTrainStatus();
                    return;
                }}
                
                // Show confirmation dialog
                const process = processes[0];
                const confirmed = confirm(
                    `Are you sure you want to stop the following training process?\n\n` +
                    `PID: ${{process.pid}}\n` +
                    `Command: ${{process.cmdline}}`
                );
                if (!confirmed) return;
                
                // Kill the process
                const response = await fetch('/api/train/kill/' + process.pid, {{ method: 'POST' }});
                const data = await response.json();
                if (data.error) {{ showToast('Error: ' + data.error); return; }}
                showToast('Training stopped');
                fetchTrainStatus();
            }} catch (error) {{ showToast('Failed to stop'); }}
        }}
        
        async function saveAllCharts() {{
            try {{
                const response = await fetch('/api/save_all');
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'training_charts_' + new Date().toISOString().slice(0,10) + '.png';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                showToast('Charts saved!');
            }} catch (error) {{ showToast('Failed to save'); }}
        }}
        
        async function copyChart(index) {{
            const img = document.getElementById(`chart-${{index}}`).querySelector('img');
            if (!img) return;
            try {{
                const response = await fetch(img.src);
                const blob = await response.blob();
                await navigator.clipboard.write([new ClipboardItem({{ 'image/png': blob }})]);
                showToast('Image copied!');
            }} catch (error) {{ showToast('Copy failed'); }}
        }}
        
        const logoDataUri = '{logo_pure_data_uri}';
        
        function renderChat(showThinking = false) {{
            const area = document.getElementById('chat-area');
            if (chatHistory.length === 0 && !showThinking) {{
                area.innerHTML = '<div class="empty-chat">Enter a sentence to start inference</div>';
                return;
            }}
            let html = chatHistory.map(msg => {{
                const avatar = msg.role === 'user' 
                    ? '<div class="chat-avatar">🤔</div>'
                    : `<div class="chat-avatar"><img src="${{logoDataUri}}" alt="Scholar"></div>`;
                return `
                <div class="chat-msg ${{msg.role}}">
                    ${{avatar}}
                    <div class="bubble">${{msg.content}}</div>
                </div>
            `}}).join('');
            if (showThinking) {{
                html += `
                <div class="chat-msg assistant">
                    <div class="chat-avatar"><img src="${{logoDataUri}}" alt="Scholar"></div>
                    <div class="bubble thinking">Thinking...<span class="dots"></span></div>
                </div>
            `;
            }}
            area.innerHTML = html;
            area.scrollTop = area.scrollHeight;
        }}
        
        function handleChatKey(e) {{
            if (e.key === 'Enter' && !e.shiftKey) {{
                e.preventDefault();
                sendInfer();
            }}
        }}
        
        async function runDebug() {{
            if (isDebugRunning) return;
            const input = document.getElementById('debug-input');
            const text = input.value.trim();
            if (!text) {{ showToast('Please enter a sentence'); return; }}
            
            isDebugRunning = true;
            document.getElementById('debug-run-btn').disabled = true;
            
            document.getElementById('debug-layer-container').innerHTML = '<div class="loading"><div class="spinner"></div>Running debug inference...</div>';
            document.getElementById('debug-output-inline').style.display = 'none';
            
            try {{
                const response = await fetch('/api/infer', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        text: text,
                        temperature: parseFloat(document.getElementById('debug-temperature').value),
                        topP: parseFloat(document.getElementById('debug-topP').value),
                        maxTokens: parseInt(document.getElementById('debug-maxTokens').value),
                        weight: document.getElementById('debug-weight-select').value,
                        debug: true
                    }})
                }});
                const data = await response.json();
                if (data.error) {{
                    document.getElementById('debug-layer-container').innerHTML = `<div class="result-content" style="color: red; padding: 20px;">Error: ${{data.error}}</div>`;
                }} else {{
                    document.getElementById('debug-output-inline').style.display = 'flex';
                    document.getElementById('debug-output-text').innerHTML = `<span>${{text}}</span><span class="generated">${{data.output}}</span>`;
                    
                    await fetchAndRenderLayers();
                }}
            }} catch (error) {{
                document.getElementById('debug-layer-container').innerHTML = `<div class="result-content" style="color: red; padding: 20px;">Request failed: ${{error}}</div>`;
            }}
            
            isDebugRunning = false;
            document.getElementById('debug-run-btn').disabled = false;
        }}
        
        async function fetchAndRenderLayers() {{
            try {{
                const response = await fetch('/api/inspect');
                const data = await response.json();
                const container = document.getElementById('debug-layer-container');
                const minimap = document.getElementById('layer-minimap');
                
                if (!data.layers || data.layers.length === 0) {{
                    container.innerHTML = '<div class="no-logs">No layer data available. Run debug inference first.</div>';
                    minimap.innerHTML = '';
                    cachedLayerData = null;
                    return;
                }}
                
                cachedLayerData = data.layers;
                
                let html = '';
                let minimapHtml = '';
                let layerIndex = 0;
                let maxTokens = 0;
                
                data.layers.forEach((layerData) => {{
                    const layerName = layerData.layer;
                    const isFinalNorm = layerName.toLowerCase().includes('finalnorm');
                    const isAttn = layerName.toLowerCase().includes('attn');
                    const isFfn = layerName.toLowerCase().includes('ffn');
                    
                    if (currentLayerFilter !== 'all' && !isFinalNorm) {{
                        if (currentLayerFilter === 'attn' && !isAttn) return;
                        if (currentLayerFilter === 'ffn' && !isFfn) return;
                    }}
                    
                    if (layerData.tokens && layerData.tokens.length > maxTokens) {{
                        maxTokens = layerData.tokens.length;
                    }}
                }});
                
                const maskSlider = document.getElementById('debug-mask');
                maskSlider.max = Math.max(0, maxTokens - 1);
                if (currentMask > maxTokens - 1) {{
                    currentMask = Math.max(0, maxTokens - 1);
                    maskSlider.value = currentMask;
                    document.getElementById('debug-mask-value').textContent = currentMask;
                }}
                
                data.layers.forEach((layerData) => {{
                    const layerName = layerData.layer;
                    const isFinalNorm = layerName.toLowerCase().includes('finalnorm');
                    const isAttn = layerName.toLowerCase().includes('attn');
                    const isFfn = layerName.toLowerCase().includes('ffn');
                    
                    if (currentLayerFilter !== 'all' && !isFinalNorm) {{
                        if (currentLayerFilter === 'attn' && !isAttn) return;
                        if (currentLayerFilter === 'ffn' && !isFfn) return;
                    }}
                    
const layerId = 'layer-' + layerIndex;
                        const shortName = layerName.replace('#', '\\n#');
                        
                        minimapHtml += `<div class="minimap-item" onclick="scrollToLayer('${{layerId}}')">${{shortName}}</div>`;
                        
                        html += `
                            <div class="layer-card" id="${{layerId}}">
                                <div class="layer-header">
                                    <span class="layer-name">${{layerName}}</span>
                                </div>
                                <div class="layer-body">
                        `;
                        
                        const tokens = layerData.tokens;
                        const showCount = Math.max(1, tokens.length - currentMask);
                        tokens.slice(0, showCount).forEach((tokenData) => {{
                            html += `
                                <div class="token-row">
                                    <span class="input-token">${{tokenData.inputToken}}</span>
                                    <span class="arrow">→</span>
                                    <div class="similar-tokens">
                            `;
                            
                            tokenData.similar.forEach((s, rank) => {{
                                const match = s.match(/(.+)\\((.+)\\)/);
                                if (match) {{
                                    const token = match[1];
                                    const score = match[2];
                                    const rankClass = rank === 0 ? 'rank-1' : (rank === 1 ? 'rank-2' : (rank === 2 ? 'rank-3' : ''));
                                    html += `<span class="similar-token ${{rankClass}}">${{token}}<span class="score">${{score}}</span></span>`;
                                }}
                            }});
                            
                            html += `</div></div>`;
                        }});
                        
                        html += `</div></div>`;
                        layerIndex++;
                    }});
                
                container.innerHTML = html;
                minimap.innerHTML = minimapHtml;
            }} catch (error) {{
                console.error('Layer fetch error:', error);
            }}
        }}
        
        function scrollToLayer(layerId) {{
            const element = document.getElementById(layerId);
            if (element) {{
                element.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                document.querySelectorAll('.minimap-item').forEach(item => item.classList.remove('active'));
                event.target.classList.add('active');
            }}
        }}
        
        async function sendInfer() {{
            if (isInferring) return;
            const input = document.getElementById('chat-input');
            const text = input.value.trim();
            if (!text) return;
            
            chatHistory.push({{ role: 'user', content: text }});
            input.value = '';
            renderChat(true);  // Show thinking indicator
            
            isInferring = true;
            document.getElementById('send-btn').disabled = true;
            
            try {{
                const response = await fetch('/api/infer', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        text: text,
                        temperature: parseFloat(document.getElementById('temperature').value),
                        topP: parseFloat(document.getElementById('topP').value),
                        maxTokens: parseInt(document.getElementById('maxTokens').value),
                        weight: document.getElementById('weight-select').value
                    }})
                }});
                const data = await response.json();
                if (data.error) {{
                    chatHistory.push({{ role: 'assistant', content: 'Error: ' + data.error }});
                }} else {{
                    chatHistory.push({{ role: 'assistant', content: text + data.output }});
                }}
            }} catch (error) {{
                chatHistory.push({{ role: 'assistant', content: 'Request failed' }});
            }}
            
            isInferring = false;
            document.getElementById('send-btn').disabled = false;
            renderChat();  // Remove thinking indicator
        }}
        
        // Initial load
        fetchCharts();
        fetchTrainStatus();
        fetchAndRenderLayers();
        setInterval(fetchCharts, refreshInterval);
        setInterval(fetchTrainStatus, 10000);
    </script>
</body>
</html>"""


# ============== MAIN ==============

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"Scholar Dashboard")
    print(f"{'='*50}")
    print(f"  URL: http://localhost:{port}")
    print(f"  Charts: {len(charts)}")
    print(f"  Refresh: {refresh_interval}s")
    print(f"{'='*50}")
    print(f"Press Ctrl+C to stop\n")

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
