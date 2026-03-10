"""Leaderboard tab for the Gradio demo.

Loads aggregated benchmark results from the most recent
``experiments/results/*_results.json`` file and renders them as a
sortable pandas DataFrame table and Plotly bar charts comparing
GraphRAG vs GNN-RAG on EM, F1, and latency metrics.

If no results file is found, a placeholder with example values is shown
so the tab is always functional.

Usage::

    from demo.leaderboard import load_summary_df, build_metric_chart, build_latency_chart
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path("experiments/results")

# Placeholder shown before any real benchmark has been run
_PLACEHOLDER_SUMMARY = {
    "GraphRAG": {
        "em": 0.0,
        "f1": 0.0,
        "latency_p50": 0.0,
        "latency_p95": 0.0,
        "n_questions": 0,
    },
    "GNN-RAG": {
        "em": 0.0,
        "f1": 0.0,
        "latency_p50": 0.0,
        "latency_p95": 0.0,
        "n_questions": 0,
    },
}


def load_summary(results_dir: Path | None = None) -> dict:
    """Load the most recent benchmark summary from disk.

    Scans ``results_dir`` for ``*_results.json`` files, picks the latest
    by filename (timestamp-prefixed), and returns the ``summary`` sub-dict.

    Args:
        results_dir: Directory to scan. Defaults to ``experiments/results/``.

    Returns:
        Summary dict keyed by pipeline name, each value a metrics dict.
        Returns the placeholder summary if no results files are found.
    """
    resolved = Path(results_dir or _RESULTS_DIR)
    candidates = sorted(resolved.glob("*_results.json"))
    if not candidates:
        logger.info("No benchmark results found in %s — using placeholder.", resolved)
        return _PLACEHOLDER_SUMMARY

    latest = candidates[-1]
    try:
        with latest.open("r", encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary", data)
        logger.info("Loaded benchmark summary from %s.", latest)
        return summary
    except Exception as exc:
        logger.warning("Failed to load %s: %s — using placeholder.", latest, exc)
        return _PLACEHOLDER_SUMMARY


def load_summary_df(results_dir: Path | None = None):
    """Load benchmark summary as a formatted pandas DataFrame.

    Columns: Pipeline, EM (%), F1 (%), Latency P50 (ms), Latency P95 (ms), N Questions.

    Args:
        results_dir: Directory to scan for result files.

    Returns:
        pandas DataFrame with one row per pipeline, or None if pandas is unavailable.
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError:
        logger.warning("pandas not installed — leaderboard table unavailable.")
        return None

    summary = load_summary(results_dir)
    rows = []
    for pipeline_name, metrics in summary.items():
        rows.append(
            {
                "Pipeline": pipeline_name,
                "EM (%)": round(metrics.get("em", 0.0) * 100, 2),
                "F1 (%)": round(metrics.get("f1", 0.0) * 100, 2),
                "Latency P50 (ms)": round(metrics.get("latency_p50", 0.0), 1),
                "Latency P95 (ms)": round(metrics.get("latency_p95", 0.0), 1),
                "N Questions": metrics.get("n_questions", 0),
            }
        )
    return pd.DataFrame(rows)


def build_metric_chart(results_dir: Path | None = None):
    """Build a Plotly grouped bar chart comparing EM and F1.

    Args:
        results_dir: Directory to scan for result files.

    Returns:
        A ``plotly.graph_objects.Figure``, or None if plotly is unavailable.
    """
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError:
        logger.warning("plotly not installed — metric chart unavailable.")
        return None

    summary = load_summary(results_dir)
    pipelines = list(summary.keys())
    em_vals = [summary[p].get("em", 0.0) * 100 for p in pipelines]
    f1_vals = [summary[p].get("f1", 0.0) * 100 for p in pipelines]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Exact Match (%)",
            x=pipelines,
            y=em_vals,
            marker_color="#4e79a7",
            text=[f"{v:.1f}%" for v in em_vals],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Token F1 (%)",
            x=pipelines,
            y=f1_vals,
            marker_color="#f28e2b",
            text=[f"{v:.1f}%" for v in f1_vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="GraphRAG vs GNN-RAG: Exact Match & F1",
        yaxis_title="Score (%)",
        barmode="group",
        yaxis_range=[0, 105],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    return fig


def build_latency_chart(results_dir: Path | None = None):
    """Build a Plotly grouped bar chart comparing P50 and P95 latency.

    Args:
        results_dir: Directory to scan for result files.

    Returns:
        A ``plotly.graph_objects.Figure``, or None if plotly is unavailable.
    """
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError:
        logger.warning("plotly not installed — latency chart unavailable.")
        return None

    summary = load_summary(results_dir)
    pipelines = list(summary.keys())
    p50_vals = [summary[p].get("latency_p50", 0.0) for p in pipelines]
    p95_vals = [summary[p].get("latency_p95", 0.0) for p in pipelines]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="P50 Latency (ms)",
            x=pipelines,
            y=p50_vals,
            marker_color="#59a14f",
            text=[f"{v:.0f} ms" for v in p50_vals],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="P95 Latency (ms)",
            x=pipelines,
            y=p95_vals,
            marker_color="#e15759",
            text=[f"{v:.0f} ms" for v in p95_vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="GraphRAG vs GNN-RAG: Latency",
        yaxis_title="Latency (ms)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    return fig


def has_real_results(results_dir: Path | None = None) -> bool:
    """Return True if at least one benchmark result file exists on disk."""
    resolved = Path(results_dir or _RESULTS_DIR)
    return bool(list(resolved.glob("*_results.json")))
