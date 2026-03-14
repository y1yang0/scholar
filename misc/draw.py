#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Dashboard Server - Web interface for real-time training metrics visualization.
Run: python draw.py
Open: http://localhost:5001
"""

import io
import json
import base64
import hashlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import subprocess

# ============== CONFIG ==============
charts = [
    {
        "files": ["../debug.log"],
        "fields": ["trainLoss"],
        "xlabel": "Iteration",
        "ylabel": "Loss",
        "title": "Train Loss",
    },
    {
        "files": ["../debug.log"],
        "fields": ["valLoss"],
        "xlabel": "Iteration",
        "ylabel": "Loss",
        "title": "Val Loss",
    },
    {
        "files": ["../debug.log"],
        "fields": ["valPPL", "trainPPL"],
        "xlabel": "Iteration",
        "ylabel": "PPL",
        "title": "Perplexity",
    },
    {
        "files": ["../debug.log"],
        "fields": ["elapsed"],
        "xlabel": "Epoch",
        "ylabel": "seconds",
        "title": "Elapsed",
    },
    {
        "files": ["../debug.log"],
        "fields": ["gradNorm"],
        "xlabel": "Iteration",
        "ylabel": "Gradient Norm",
        "title": "Gradient Norm",
    },
    {
        "files": ["../debug.log"],
        "fields": ["currentLR"],
        "xlabel": "Iteration",
        "ylabel": "Learning Rate",
        "title": "Learning Rate",
    },
]
metrics = [
    {"file": "../debug.log", "type": "uptime", "title": "Uptime"},
    {"file": "../debug.log", "event": "config"},
]
port = 5002
refresh_interval = 3  # seconds
# ====================================

SCRIPT_DIR = Path(__file__).parent
LOGO_PATH = SCRIPT_DIR / "logo.png"


def get_logo_base64():
    """Read logo.png and return as base64 data URI"""
    if not LOGO_PATH.exists():
        return None
    try:
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except:
        return None


def get_gpu_stats():
    """Get GPU utilization and memory usage via nvidia-smi"""
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
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization": int(parts[2]),
                    "memory_used": int(parts[3]),
                    "memory_total": int(parts[4]),
                    "temperature": int(parts[5]),
                })
        return gpus
    except Exception:
        return []


def parse_log_file(log_path: str, fields: list):
    """Parse log file and extract values for all specified fields"""
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


def draw_single_chart(chart_config: dict):
    """Draw a single chart and return as base64 PNG"""
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
    stats_lines = []  # collect stats for footer

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

            # Calculate stats
            min_val = min(values)
            max_val = max(values)
            cur_val = values[-1]
            min_idx = values.index(min_val)
            max_idx = values.index(max_val)

            # Mark min/max points on chart
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

            # Collect stats for footer
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

    # Add stats footer
    if stats_lines:
        footer_text = "  |  ".join(stats_lines)
        fig.text(0.5, 0.02, footer_text, ha="center", fontsize=10, color="#333")

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # leave space for footer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


def draw_all_charts_combined():
    """Draw all charts combined into a single image"""
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

    # Hide unused subplots
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
    """Get MD5 hash of file content"""
    if not file_path.exists():
        return None
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None


def get_data_hash():
    """Get combined hash of all watched files"""
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
    """Format value for display"""
    if isinstance(val, float):
        if val < 0.01:
            return f"{val:.2e}"
        return f"{val:.4g}"
    if isinstance(val, int):
        if val >= 1_000_000:
            return f"{val/1_000_000:.1f}M"
        if val >= 1_000:
            return f"{val/1_000:.1f}K"
    return str(val)


def get_metrics_data():
    """Get text metrics data"""
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
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)
                if hours > 0:
                    value = f"{hours}h {minutes}m {seconds}s"
                elif minutes > 0:
                    value = f"{minutes}m {seconds}s"
                else:
                    value = f"{seconds}s"
            result.append({"title": title, "value": value})

        elif "event" in m:
            # Read all fields from all matching events
            event_name = m.get("event", "")
            skip_fields = {"event"}  # fields to skip
            seen_keys = set()  # avoid duplicates
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
                                    title = (
                                        key[0].upper() + key[1:]
                                    )  # capitalize first letter
                                    result.append(
                                        {"title": title, "value": format_value(val)}
                                    )
                        except json.JSONDecodeError:
                            continue
            except:
                pass

    return result


def get_inference_samples():
    """Get latest inference samples from log file (only idx=0)"""
    samples = []
    log_path = SCRIPT_DIR / "../debug.log"

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
                    # Only read records with idx=0, skip if idx field doesn't exist
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

    # Return only the last 10 samples
    return samples[-10:] if len(samples) > 10 else samples


def build_html():
    """Build HTML page"""
    logo_base64 = get_logo_base64()
    logo_data_uri = f"data:image/png;base64,{logo_base64}" if logo_base64 else ""
    
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

    favicon_html = f'<link rel="icon" type="image/png" href="{logo_data_uri}">' if logo_data_uri else ""
    logo_img_html = f'<img src="{logo_data_uri}" alt="Scholar Logo" class="header-logo">' if logo_data_uri else ""
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scholar Training Dashboard</title>
    {favicon_html}
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; min-height: 100vh; padding: 20px; }}
        .header {{ text-align: center; margin-bottom: 20px; color: #333; }}
        .header-title {{ display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 8px; }}
        .header-logo {{ width: 40px; height: 40px; object-fit: contain; }}
        .header h1 {{ font-size: 1.5rem; font-weight: 600; margin: 0; }}
        .header .status {{ font-size: 0.85rem; color: #666; }}
        .header .status .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #52c41a; margin-right: 6px; animation: pulse 2s infinite; }}
        @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
        .layout {{ display: flex; gap: 20px; max-width: 1800px; margin: 0 auto; }}
        .charts-area {{ flex: 1; min-width: 0; }}
        .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 16px; }}
        .sidebar {{ width: 280px; flex-shrink: 0; }}
        .metrics-card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; }}
        .metrics-header {{ display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; cursor: pointer; user-select: none; }}
        .metrics-header:hover {{ background: #fafafa; border-radius: 8px 8px 0 0; }}
        .metrics-header h3 {{ font-size: 0.85rem; color: #666; font-weight: 500; margin: 0; }}
        .metrics-toggle {{ font-size: 12px; color: #999; transition: transform 0.3s; }}
        .metrics-toggle.collapsed {{ transform: rotate(-90deg); }}
        .metrics-content {{ border-top: 1px solid #f0f0f0; padding: 8px 16px; }}
        .metrics-content.collapsed {{ display: none; }}
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
        .samples-card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; margin-top: 16px; }}
        .samples-header {{ display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; cursor: pointer; user-select: none; }}
        .samples-header:hover {{ background: #fafafa; border-radius: 8px 8px 0 0; }}
        .samples-header h3 {{ font-size: 0.85rem; color: #666; font-weight: 500; margin: 0; }}
        .samples-toggle {{ font-size: 12px; color: #999; transition: transform 0.3s; }}
        .samples-toggle.collapsed {{ transform: rotate(-90deg); }}
        .samples-content {{ max-height: 280px; overflow-y: auto; border-top: 1px solid #f0f0f0; }}
        .samples-content.collapsed {{ display: none; }}
        .gpu-card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; margin-top: 16px; }}
        .gpu-header {{ display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; cursor: pointer; user-select: none; }}
        .gpu-header:hover {{ background: #fafafa; border-radius: 8px 8px 0 0; }}
        .gpu-header h3 {{ font-size: 0.85rem; color: #666; font-weight: 500; margin: 0; }}
        .gpu-toggle {{ font-size: 12px; color: #999; transition: transform 0.3s; }}
        .gpu-toggle.collapsed {{ transform: rotate(-90deg); }}
        .gpu-content {{ border-top: 1px solid #f0f0f0; padding: 8px 16px; }}
        .gpu-content.collapsed {{ display: none; }}
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
        .sample-item {{ padding: 10px 16px; border-bottom: 1px solid #f0f0f0; }}
        .sample-item:last-child {{ border-bottom: none; }}
        .sample-input {{ font-size: 0.8rem; color: #1890ff; font-weight: 500; margin-bottom: 4px; }}
        .sample-output {{ font-size: 0.8rem; color: #333; line-height: 1.4; word-break: break-all; }}
        .no-samples {{ padding: 20px; text-align: center; color: #999; font-size: 0.8rem; }}
        .control-card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; margin-top: 16px; padding: 16px; }}
        .control-card h3 {{ font-size: 0.85rem; color: #666; font-weight: 500; margin: 0 0 12px 0; }}
        .save-btn {{ width: 100%; background: #1890ff; border: none; color: #fff; padding: 10px 16px; border-radius: 4px; cursor: pointer; font-size: 0.85rem; display: flex; align-items: center; justify-content: center; gap: 6px; }}
        .save-btn:hover {{ background: #40a9ff; }}
        .save-btn:active {{ background: #096dd9; }}
        .save-btn svg {{ width: 16px; height: 16px; }}
        @media (max-width: 800px) {{ .layout {{ flex-direction: column-reverse; }} .sidebar {{ width: 100%; }} }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">
            {logo_img_html}
            <h1>Scholar Training Dashboard</h1>
        </div>
        <div class="status"><span class="dot"></span><span id="update-time">Loading...</span></div>
    </div>
    <div class="layout">
        <div class="charts-area"><div class="charts-grid">{cards}</div></div>
        <div class="sidebar">
            <div class="metrics-card">
                <div class="metrics-header" onclick="toggleMetrics()">
                    <h3>Metrics</h3>
                    <span class="metrics-toggle" id="metrics-toggle">▼</span>
                </div>
                <div class="metrics-content" id="metrics-container"></div>
            </div>
            <div class="gpu-card">
                <div class="gpu-header" onclick="toggleGpu()">
                    <h3>GPU Monitor</h3>
                    <span class="gpu-toggle" id="gpu-toggle">▼</span>
                </div>
                <div class="gpu-content" id="gpu-content"></div>
            </div>
            <div class="samples-card">
                <div class="samples-header" onclick="toggleSamples()">
                    <h3>Inference Samples</h3>
                    <span class="samples-toggle" id="samples-toggle">▼</span>
                </div>
                <div class="samples-content" id="samples-content"></div>
            </div>
            <div class="control-card">
                <h3>Control Panel</h3>
                <button class="save-btn" onclick="saveAllCharts()">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                        <polyline points="7 10 12 15 17 10"/>
                        <line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                    Save All Charts
                </button>
            </div>
        </div>
    </div>
    <div class="toast" id="toast">Image copied to clipboard!</div>
    <script>
        let lastHash = '';
        const refreshInterval = {refresh_interval} * 1000;
        
        function toggleSamples() {{
            const content = document.getElementById('samples-content');
            const toggle = document.getElementById('samples-toggle');
            content.classList.toggle('collapsed');
            toggle.classList.toggle('collapsed');
        }}
        
        function toggleGpu() {{
            const content = document.getElementById('gpu-content');
            const toggle = document.getElementById('gpu-toggle');
            content.classList.toggle('collapsed');
            toggle.classList.toggle('collapsed');
        }}
        
        function renderGpuStats(gpus) {{
            const container = document.getElementById('gpu-content');
            if (!gpus || gpus.length === 0) {{
                container.innerHTML = '<div class="no-gpu">No GPU detected or nvidia-smi unavailable</div>';
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
        
        function toggleMetrics() {{
            const content = document.getElementById('metrics-container');
            const toggle = document.getElementById('metrics-toggle');
            content.classList.toggle('collapsed');
            toggle.classList.toggle('collapsed');
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
                document.getElementById('toast').textContent = 'Charts saved!';
                document.getElementById('toast').classList.add('show');
                setTimeout(() => document.getElementById('toast').classList.remove('show'), 2000);
            }} catch (error) {{
                alert('Failed to save charts');
            }}
        }}
        
        async function fetchCharts() {{
            try {{
                const response = await fetch('/api/charts');
                const data = await response.json();
                // Update metrics dynamically
                if (data.metrics) {{
                    const container = document.getElementById('metrics-container');
                    container.innerHTML = data.metrics.map(m => 
                        `<div class="metric"><span class="metric-title">${{m.title}}</span><span class="metric-value">${{m.value}}</span></div>`
                    ).join('');
                }}
                // Update GPU stats
                renderGpuStats(data.gpu_stats);
                // Update inference samples
                if (data.samples) {{
                    const samplesContainer = document.getElementById('samples-content');
                    if (data.samples.length > 0) {{
                        samplesContainer.innerHTML = data.samples.map(s => 
                            `<div class="sample-item"><div class="sample-input">${{s.input}}</div><div class="sample-output">${{s.output}}</div></div>`
                        ).join('');
                    }} else {{
                        samplesContainer.innerHTML = '<div class="no-samples">No inference samples yet</div>';
                    }}
                }}
                // Update charts
                if (data.hash !== lastHash) {{
                    lastHash = data.hash;
                    data.charts.forEach((imgData, index) => {{
                        const container = document.getElementById(`chart-${{index}}`);
                        if (container) container.innerHTML = `<img src="data:image/png;base64,${{imgData}}" />`;
                    }});
                }}
                const uptimeMetric = data.metrics.find(m => m.title === 'Uptime');
                const uptimeStr = uptimeMetric ? ` | Training elapsed: ${{uptimeMetric.value}}` : '';
                document.getElementById('update-time').textContent = `Last updated: ${{new Date().toLocaleTimeString()}}${{uptimeStr}}`;
            }} catch (error) {{ console.error('Failed to fetch charts:', error); }}
        }}
        
        async function copyChart(index) {{
            const img = document.getElementById(`chart-${{index}}`).querySelector('img');
            if (!img) return;
            const btn = img.closest('.card').querySelector('.copy-btn');
            try {{
                const response = await fetch(img.src);
                const blob = await response.blob();
                await navigator.clipboard.write([new ClipboardItem({{ 'image/png': blob }})]);
                btn.classList.add('copied');
                btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>Copied!`;
                document.getElementById('toast').classList.add('show');
                setTimeout(() => document.getElementById('toast').classList.remove('show'), 2000);
                setTimeout(() => {{
                    btn.classList.remove('copied');
                    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>Copy`;
                }}, 2000);
            }} catch (error) {{ alert('Failed to copy image'); }}
        }}
        
        fetchCharts();
        setInterval(fetchCharts, refreshInterval);
    </script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(build_html().encode("utf-8"))
        elif self.path == "/api/charts":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            charts_data = [draw_single_chart(c) for c in charts]
            response = json.dumps(
                {
                    "charts": charts_data,
                    "metrics": get_metrics_data(),
                    "samples": get_inference_samples(),
                    "gpu_stats": get_gpu_stats(),
                    "hash": get_data_hash(),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            self.wfile.write(response.encode("utf-8"))
        elif self.path == "/api/save_all":
            img_data = draw_all_charts_combined()
            if img_data:
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header(
                    "Content-Disposition", 'attachment; filename="training_charts.png"'
                )
                self.end_headers()
                self.wfile.write(img_data)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def run_server():
    """Start the dashboard web server"""
    print(f"\n{'='*50}")
    print(f"Training Dashboard Server")
    print(f"{'='*50}")
    print(f"  URL: http://localhost:{port}")
    print(f"  Charts: {len(charts)}")
    print(f"  Metrics: {len(metrics)}")
    print(f"  Refresh: {refresh_interval}s")
    print(f"{'='*50}")
    print(f"Press Ctrl+C to stop\n")

    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()


if __name__ == "__main__":
    run_server()
