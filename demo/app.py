"""Gradio demo application for GraphBench — HuggingFace Spaces entry point.

Three-tab interface:

- **Tab 1 — Live Q&A**: Enter a question, see both pipeline answers side-by-side
  with retrieved context triples and per-pipeline latency.
- **Tab 2 — Graph View**: Interactive pyvis visualisation of the knowledge-graph
  subgraph retrieved for the last question, with community (GraphRAG) or
  GAT-score (GNN-RAG) colouring.
- **Tab 3 — Leaderboard**: Aggregated benchmark results table and Plotly charts
  loaded from ``experiments/results/``.

Environment variables (set as HuggingFace Spaces secrets):
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD — Neo4j AuraDB connection
    HF_TOKEN — HuggingFace Hub token for model loading
    FAISS_INDEX_PATH — Path to FAISS index (default: data/faiss_index)
    CHECKPOINT_DIR — Path to GAT checkpoint directory

If required services are unavailable the app enters **demo mode**: pre-baked
sample answers are shown for 10 questions and live inference is disabled.

Launch locally::

    cd demo && pip install -r requirements.txt
    python app.py
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline loading (lazy, with graceful fallback to demo mode)
# ---------------------------------------------------------------------------

_PIPELINES_LOADED = False
_GRAPHRAG = None
_GNNRAG = None
_LOAD_ERROR: str | None = None

# 10 pre-baked Q&A pairs for demo mode (filled after real benchmark)
_DEMO_QA: list[dict] = [
    {
        "question": "Where was Marie Curie born?",
        "graphrag": "Warsaw, Poland",
        "gnnrag": "Warsaw",
        "context": [
            ("Marie Curie", "place_of_birth", "Warsaw"),
            ("Warsaw", "country", "Poland"),
        ],
    },
    {
        "question": "What did Albert Einstein win the Nobel Prize for?",
        "graphrag": "Physics — photoelectric effect",
        "gnnrag": "Photoelectric effect",
        "context": [
            ("Albert Einstein", "award_received", "Nobel Prize in Physics"),
            ("Albert Einstein", "notable_work", "Photoelectric effect"),
        ],
    },
    {
        "question": "Which country is Warsaw located in?",
        "graphrag": "Poland",
        "gnnrag": "Poland",
        "context": [("Warsaw", "country", "Poland")],
    },
    {
        "question": "Who discovered penicillin?",
        "graphrag": "Alexander Fleming",
        "gnnrag": "Alexander Fleming",
        "context": [("Alexander Fleming", "discovered", "Penicillin")],
    },
    {
        "question": "In what field did Marie Curie work?",
        "graphrag": "Chemistry and Physics",
        "gnnrag": "Physics",
        "context": [
            ("Marie Curie", "field_of_work", "Chemistry"),
            ("Marie Curie", "field_of_work", "Physics"),
        ],
    },
    {
        "question": "Where was Albert Einstein born?",
        "graphrag": "Ulm, Germany",
        "gnnrag": "Ulm",
        "context": [
            ("Albert Einstein", "place_of_birth", "Ulm"),
            ("Ulm", "country", "Germany"),
        ],
    },
    {
        "question": "What theory is Stephen Hawking known for?",
        "graphrag": "Hawking radiation and black hole thermodynamics",
        "gnnrag": "Hawking radiation",
        "context": [
            ("Stephen Hawking", "notable_work", "A Brief History of Time"),
            ("Stephen Hawking", "field_of_work", "Theoretical physics"),
        ],
    },
    {
        "question": "Which organisation did Tim Berners-Lee found?",
        "graphrag": "World Wide Web Consortium (W3C)",
        "gnnrag": "W3C",
        "context": [("Tim Berners-Lee", "founded", "World Wide Web Consortium")],
    },
    {
        "question": "What is the capital of France?",
        "graphrag": "Paris",
        "gnnrag": "Paris",
        "context": [("France", "capital", "Paris")],
    },
    {
        "question": "Who wrote the theory of evolution?",
        "graphrag": "Charles Darwin",
        "gnnrag": "Charles Darwin",
        "context": [("Charles Darwin", "known_for", "Theory of Evolution")],
    },
]

_DEMO_QA_MAP = {item["question"].lower(): item for item in _DEMO_QA}


def _try_load_pipelines() -> str:
    """Attempt to load real pipelines from env-configured services.

    Returns:
        Status message string indicating success or failure reason.
    """
    global _PIPELINES_LOADED, _GRAPHRAG, _GNNRAG, _LOAD_ERROR  # noqa: PLW0603

    required_vars = ["NEO4J_URI", "NEO4J_PASSWORD"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        msg = f"Demo mode — missing env vars: {', '.join(missing)}."
        _LOAD_ERROR = msg
        return msg

    try:
        from graphbench.pipelines.gnnrag_pipeline import GNNRAGPipeline  # noqa: PLC0415
        from graphbench.pipelines.graphrag_pipeline import (  # noqa: PLC0415
            GraphRAGPipeline,
        )
        from graphbench.utils.config import settings  # noqa: PLC0415
        from graphbench.utils.faiss_client import FAISSClient  # noqa: PLC0415
        from graphbench.utils.llm_client import LLMClient  # noqa: PLC0415
        from graphbench.utils.neo4j_client import Neo4jClient  # noqa: PLC0415

        neo4j = Neo4jClient()
        faiss_client = FAISSClient.load()
        llm = LLMClient()

        _GRAPHRAG = GraphRAGPipeline(
            neo4j_client=neo4j,
            faiss_client=faiss_client,
            llm_client=llm,
        )

        # Try to load GAT checkpoint for GNN-RAG
        try:
            from graphbench.gnn.model import GATModel  # noqa: PLC0415
            from graphbench.utils.checkpoint import (  # noqa: PLC0415
                load_best_checkpoint,
            )

            checkpoint, _ = load_best_checkpoint(settings.checkpoint_dir)
            gat = GATModel()
            gat.load_state_dict(checkpoint["model_state_dict"])
            gat.eval()

            _GNNRAG = GNNRAGPipeline(
                neo4j_client=neo4j,
                faiss_client=faiss_client,
                llm_client=llm,
                gat_model=gat,
            )
            msg = "✅ Both pipelines loaded (live mode)."
        except Exception as gnn_exc:
            logger.warning(
                "GNN checkpoint unavailable: %s. GNN-RAG in demo mode.", gnn_exc
            )
            _GNNRAG = None
            msg = "⚠️ GraphRAG live, GNN-RAG in demo mode (no checkpoint)."

        _PIPELINES_LOADED = True
        return msg

    except Exception as exc:
        _LOAD_ERROR = str(exc)
        logger.warning("Pipeline loading failed: %s", exc)
        return f"Demo mode — pipeline error: {exc}"


# ---------------------------------------------------------------------------
# Answer function
# ---------------------------------------------------------------------------


def answer(question: str, state: dict) -> tuple[str, str, str, str, dict]:
    """Handle a question submission from the Gradio interface.

    Args:
        question: User's question string.
        state: Gradio state dict (carries last subgraph triples).

    Returns:
        Tuple of:
        - graphrag_answer: Formatted GraphRAG answer markdown
        - gnnrag_answer:   Formatted GNN-RAG answer markdown
        - graphrag_context: Context triples as text
        - gnnrag_context:  Context triples as text
        - updated_state:   State dict with ``last_triples`` for graph tab
    """
    if not question or not question.strip():
        return "Please enter a question.", "", "", "", state

    question = question.strip()

    if _PIPELINES_LOADED and _GRAPHRAG is not None:
        return _live_answer(question, state)
    return _demo_answer(question, state)


def _live_answer(question: str, state: dict) -> tuple[str, str, str, str, dict]:
    """Call real pipelines and format results."""
    from graphbench.pipelines.base import PipelineResult  # noqa: PLC0415

    def run(pipeline, name: str) -> tuple[PipelineResult, float]:
        t0 = time.perf_counter()
        result = pipeline.answer(question)
        ms = (time.perf_counter() - t0) * 1000
        result.latency_ms = ms
        return result, ms

    ra, lat_a = run(_GRAPHRAG, "GraphRAG")
    rb_answer = "GNN-RAG not available (no checkpoint)."
    rb_context: list = []
    lat_b = 0.0
    if _GNNRAG is not None:
        rb, lat_b = run(_GNNRAG, "GNN-RAG")
        rb_answer = rb.predicted_answer
        rb_context = rb.context_triples

    state["last_triples_a"] = ra.context_triples
    state["last_triples_b"] = rb_context
    state["last_community_map"] = ra.metadata.get("community_map")

    return (
        _fmt_answer(ra.predicted_answer, lat_a),
        _fmt_answer(rb_answer, lat_b),
        _fmt_triples(ra.context_triples),
        _fmt_triples(rb_context),
        state,
    )


def _demo_answer(question: str, state: dict) -> tuple[str, str, str, str, dict]:
    """Return pre-baked demo answers (no live pipelines available)."""
    item = _DEMO_QA_MAP.get(question.lower())
    if item:
        triples = item["context"]
        state["last_triples_a"] = triples
        state["last_triples_b"] = triples
        return (
            _fmt_answer(item["graphrag"], latency_ms=None, demo=True),
            _fmt_answer(item["gnnrag"], latency_ms=None, demo=True),
            _fmt_triples(triples),
            _fmt_triples(triples),
            state,
        )
    # Not in demo set
    state["last_triples_a"] = []
    state["last_triples_b"] = []
    return (
        "*(Demo mode — try one of the sample questions below.)*",
        "*(Demo mode — try one of the sample questions below.)*",
        "",
        "",
        state,
    )


# ---------------------------------------------------------------------------
# Graph visualisation
# ---------------------------------------------------------------------------


def render_graph(pipeline_choice: str, state: dict) -> str:
    """Render the last retrieved subgraph as pyvis HTML.

    Args:
        pipeline_choice: ``"GraphRAG"`` or ``"GNN-RAG"``.
        state: Gradio state dict containing ``last_triples_a/b``.

    Returns:
        HTML string for ``gr.HTML`` component.
    """
    from demo.graph_viz import build_legend_html, triples_to_html  # noqa: PLC0415

    key = "last_triples_a" if pipeline_choice == "GraphRAG" else "last_triples_b"
    triples = state.get(key, [])

    if not triples:
        return "<p>Submit a question first to see the retrieved subgraph.</p>"

    community_map = (
        state.get("last_community_map") if pipeline_choice == "GraphRAG" else None
    )
    graph_html = triples_to_html(triples, community_map=community_map)
    legend_html = build_legend_html(community_map=community_map)
    return graph_html + legend_html


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def refresh_leaderboard() -> tuple:
    """Load latest benchmark results and return (df, metric_chart, latency_chart)."""
    from demo.leaderboard import (  # noqa: PLC0415
        build_latency_chart,
        build_metric_chart,
        has_real_results,
        load_summary_df,
    )

    df = load_summary_df()
    metric_fig = build_metric_chart()
    latency_fig = build_latency_chart()
    status = (
        "✅ Benchmark results loaded."
        if has_real_results()
        else "⚠️ No benchmark results yet — run `Evaluator.run()` first."
    )
    return df, metric_fig, latency_fig, status


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_answer(answer: str, latency_ms: float | None, demo: bool = False) -> str:
    suffix = " *(demo)*" if demo else (f" `{latency_ms:.0f} ms`" if latency_ms else "")
    return f"**{answer}**{suffix}"


def _fmt_triples(triples: list[tuple[str, str, str]]) -> str:
    if not triples:
        return "*No context retrieved.*"
    lines = [f"- **{s}** —[{r}]→ **{o}**" for s, r, o in triples[:20]]
    suffix = f"\n*…and {len(triples) - 20} more*" if len(triples) > 20 else ""
    return "\n".join(lines) + suffix


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_app():
    """Build and return the Gradio Blocks app."""
    import gradio as gr  # noqa: PLC0415

    pipeline_status = _try_load_pipelines()

    sample_questions = [item["question"] for item in _DEMO_QA]

    with gr.Blocks(
        title="GraphBench — Graph-Based RAG Demo",
        theme=gr.themes.Soft(),
        css=".gr-button { font-weight: 600; }",
    ) as demo_app:

        state = gr.State({})

        gr.Markdown(
            """
# 🔬 GraphBench
**Benchmarking Graph-Based RAG Pipelines on Multi-Hop QA**

Compare two retrieval strategies on HotpotQA questions:
| Pipeline | Strategy |
|----------|----------|
| **GraphRAG** | Louvain community detection on Neo4j subgraph |
| **GNN-RAG**  | 3-layer GAT edge scoring on Neo4j subgraph |
"""
        )

        gr.Markdown(f"**Status:** {pipeline_status}")

        # ── Tab 1: Live Q&A ──────────────────────────────────────────────
        with gr.Tab("🔍 Live Q&A"):
            with gr.Row():
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="e.g. Where was Marie Curie born?",
                    lines=2,
                    scale=4,
                )
                submit_btn = gr.Button("Ask", variant="primary", scale=1)

            gr.Examples(
                examples=sample_questions,
                inputs=question_input,
                label="Sample questions",
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### GraphRAG")
                    graphrag_answer = gr.Markdown(
                        "*Submit a question to see the answer.*"
                    )
                    gr.Markdown("**Retrieved context:**")
                    graphrag_context = gr.Markdown("")

                with gr.Column():
                    gr.Markdown("### GNN-RAG")
                    gnnrag_answer = gr.Markdown(
                        "*Submit a question to see the answer.*"
                    )
                    gr.Markdown("**Retrieved context:**")
                    gnnrag_context = gr.Markdown("")

            submit_btn.click(
                fn=answer,
                inputs=[question_input, state],
                outputs=[
                    graphrag_answer,
                    gnnrag_answer,
                    graphrag_context,
                    gnnrag_context,
                    state,
                ],
            )
            question_input.submit(
                fn=answer,
                inputs=[question_input, state],
                outputs=[
                    graphrag_answer,
                    gnnrag_answer,
                    graphrag_context,
                    gnnrag_context,
                    state,
                ],
            )

        # ── Tab 2: Graph View ────────────────────────────────────────────
        with gr.Tab("🕸️ Graph View"):
            gr.Markdown(
                "Interactive visualisation of the knowledge-graph subgraph retrieved "
                "for the last question. Submit a question on the **Live Q&A** tab first."
            )
            pipeline_radio = gr.Radio(
                choices=["GraphRAG", "GNN-RAG"],
                value="GraphRAG",
                label="Pipeline",
            )
            render_btn = gr.Button("Render Graph", variant="secondary")
            graph_html = gr.HTML("<p>Submit a question first.</p>")

            render_btn.click(
                fn=render_graph,
                inputs=[pipeline_radio, state],
                outputs=[graph_html],
            )

        # ── Tab 3: Leaderboard ───────────────────────────────────────────
        with gr.Tab("🏆 Leaderboard"):
            gr.Markdown(
                "Benchmark results from running `Evaluator.run()` on 500 HotpotQA "
                "distractor questions. Refresh to load latest results."
            )
            refresh_btn = gr.Button("Refresh Results", variant="secondary")
            leaderboard_status = gr.Markdown("")

            leaderboard_table = gr.Dataframe(
                headers=[
                    "Pipeline",
                    "EM (%)",
                    "F1 (%)",
                    "Latency P50 (ms)",
                    "Latency P95 (ms)",
                    "N Questions",
                ],
                label="Benchmark Summary",
                interactive=False,
            )

            with gr.Row():
                metric_chart = gr.Plot(label="EM & F1 Comparison")
                latency_chart = gr.Plot(label="Latency Comparison")

            refresh_btn.click(
                fn=refresh_leaderboard,
                inputs=[],
                outputs=[
                    leaderboard_table,
                    metric_chart,
                    latency_chart,
                    leaderboard_status,
                ],
            )

            # Auto-load on tab open
            demo_app.load(
                fn=refresh_leaderboard,
                inputs=[],
                outputs=[
                    leaderboard_table,
                    metric_chart,
                    latency_chart,
                    leaderboard_status,
                ],
            )

    return demo_app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
