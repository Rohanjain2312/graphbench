"""GraphBench — HuggingFace Spaces demo entry point.

Three-tab Gradio interface:
  Tab 1 — Live Q&A:    Ask a question, see both pipeline answers side-by-side.
  Tab 2 — Graph View:  Interactive pyvis KG subgraph for the last question.
  Tab 3 — Leaderboard: Real benchmark results table and Plotly charts.

Running in demo mode on HF Spaces (no GPU / Neo4j).
Live inference requires a GPU instance with Neo4j AuraDB + FAISS index.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline loading — graceful fallback to demo mode
# ---------------------------------------------------------------------------

_PIPELINES_LOADED = False
_GRAPHRAG = None
_GNNRAG = None
_LOAD_ERROR: str | None = None

_DEMO_QA: list[dict] = [
    {
        "question": "Where was Marie Curie born?",
        "graphrag": "Warsaw, Poland",
        "gnnrag": "Warsaw",
        "context": [
            ("marie curie", "place_of_birth", "warsaw"),
            ("warsaw", "country", "poland"),
        ],
    },
    {
        "question": "What did Albert Einstein win the Nobel Prize for?",
        "graphrag": "Physics — discovery of the law of the photoelectric effect",
        "gnnrag": "Photoelectric effect",
        "context": [
            ("albert einstein", "award_received", "nobel prize in physics"),
            ("albert einstein", "notable_work", "photoelectric effect"),
        ],
    },
    {
        "question": "Which country is Warsaw located in?",
        "graphrag": "Poland",
        "gnnrag": "Poland",
        "context": [("warsaw", "country", "poland")],
    },
    {
        "question": "Who discovered penicillin?",
        "graphrag": "Alexander Fleming",
        "gnnrag": "Alexander Fleming",
        "context": [
            ("alexander fleming", "notable_work", "penicillin"),
            ("alexander fleming", "field_of_work", "bacteriology"),
        ],
    },
    {
        "question": "In what field did Marie Curie work?",
        "graphrag": "Chemistry and Physics",
        "gnnrag": "Physics",
        "context": [
            ("marie curie", "field_of_work", "chemistry"),
            ("marie curie", "field_of_work", "physics"),
        ],
    },
    {
        "question": "Where was Albert Einstein born?",
        "graphrag": "Ulm, Kingdom of Württemberg, Germany",
        "gnnrag": "Ulm, Germany",
        "context": [
            ("albert einstein", "place_of_birth", "ulm"),
            ("ulm", "country", "germany"),
        ],
    },
    {
        "question": "What theory is Stephen Hawking known for?",
        "graphrag": "Hawking radiation and black hole thermodynamics",
        "gnnrag": "Hawking radiation",
        "context": [
            ("stephen hawking", "notable_work", "a brief history of time"),
            ("stephen hawking", "field_of_work", "theoretical physics"),
        ],
    },
    {
        "question": "Which organisation did Tim Berners-Lee found?",
        "graphrag": "World Wide Web Consortium (W3C)",
        "gnnrag": "W3C",
        "context": [
            ("tim berners-lee", "employer", "world wide web consortium"),
            ("tim berners-lee", "notable_work", "world wide web"),
        ],
    },
    {
        "question": "What is the capital of France?",
        "graphrag": "Paris",
        "gnnrag": "Paris",
        "context": [("france", "capital", "paris")],
    },
    {
        "question": "Who wrote the theory of evolution by natural selection?",
        "graphrag": "Charles Darwin",
        "gnnrag": "Charles Darwin",
        "context": [
            ("charles darwin", "notable_work", "on the origin of species"),
            ("charles darwin", "field_of_work", "natural history"),
        ],
    },
]

_DEMO_QA_MAP = {item["question"].lower(): item for item in _DEMO_QA}


def _try_load_pipelines() -> str:
    global _PIPELINES_LOADED, _GRAPHRAG, _GNNRAG, _LOAD_ERROR  # noqa: PLW0603

    required_vars = ["NEO4J_URI", "NEO4J_PASSWORD"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        msg = f"🟡 Demo mode — missing env vars: {', '.join(missing)}. Showing pre-baked answers."
        _LOAD_ERROR = msg
        return msg

    try:
        from graphbench.pipelines.gnnrag_pipeline import GNNRAGPipeline
        from graphbench.pipelines.graphrag_pipeline import GraphRAGPipeline
        from graphbench.utils.config import settings
        from graphbench.utils.faiss_client import FAISSClient
        from graphbench.utils.llm_client import LLMClient
        from graphbench.utils.neo4j_client import Neo4jClient

        neo4j = Neo4jClient()
        faiss_client = FAISSClient.load()
        llm = LLMClient()

        _GRAPHRAG = GraphRAGPipeline(
            neo4j_client=neo4j, faiss_client=faiss_client, llm_client=llm
        )

        try:
            from graphbench.gnn.model import GATModel
            from graphbench.utils.checkpoint import load_best_checkpoint

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
            msg = "🟢 Both pipelines loaded (live mode)."
        except Exception as gnn_exc:
            logger.warning("GNN checkpoint unavailable: %s", gnn_exc)
            _GNNRAG = None
            msg = "🟡 GraphRAG live, GNN-RAG in demo mode (no checkpoint)."

        _PIPELINES_LOADED = True
        return msg

    except Exception as exc:
        _LOAD_ERROR = str(exc)
        return f"🟡 Demo mode — pipeline error: {exc}"


# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------

def answer(question: str, state: dict) -> tuple[str, str, str, str, dict]:
    if not question or not question.strip():
        return "Please enter a question.", "", "", "", state
    question = question.strip()
    if _PIPELINES_LOADED and _GRAPHRAG is not None:
        return _live_answer(question, state)
    return _demo_answer(question, state)


def _live_answer(question: str, state: dict) -> tuple[str, str, str, str, dict]:
    def run(pipeline):
        t0 = time.perf_counter()
        result = pipeline.answer(question)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    ra = run(_GRAPHRAG)
    rb_answer, rb_context, lat_b = "GNN-RAG not available (no checkpoint).", [], 0.0
    if _GNNRAG is not None:
        rb = run(_GNNRAG)
        rb_answer, rb_context, lat_b = rb.predicted_answer, rb.context_triples, rb.latency_ms

    state.update({
        "last_triples_a": ra.context_triples,
        "last_triples_b": rb_context,
        "last_community_map": ra.metadata.get("community_map"),
    })
    return (
        _fmt_answer(ra.predicted_answer, ra.latency_ms),
        _fmt_answer(rb_answer, lat_b),
        _fmt_triples(ra.context_triples),
        _fmt_triples(rb_context),
        state,
    )


def _demo_answer(question: str, state: dict) -> tuple[str, str, str, str, dict]:
    item = _DEMO_QA_MAP.get(question.lower())
    if item:
        triples = item["context"]
        state.update({"last_triples_a": triples, "last_triples_b": triples})
        return (
            _fmt_answer(item["graphrag"], latency_ms=None, demo=True),
            _fmt_answer(item["gnnrag"], latency_ms=None, demo=True),
            _fmt_triples(triples),
            _fmt_triples(triples),
            state,
        )
    state.update({"last_triples_a": [], "last_triples_b": []})
    return (
        "_Demo mode — try one of the sample questions below._",
        "_Demo mode — try one of the sample questions below._",
        "", "", state,
    )


# ---------------------------------------------------------------------------
# Graph view
# ---------------------------------------------------------------------------

def render_graph(pipeline_choice: str, state: dict) -> str:
    from graph_viz import build_legend_html, triples_to_html

    key = "last_triples_a" if pipeline_choice == "GraphRAG" else "last_triples_b"
    triples = state.get(key, [])
    if not triples:
        return "<p style='color:#888;padding:16px'>Submit a question on the <b>Live Q&A</b> tab first.</p>"

    community_map = state.get("last_community_map") if pipeline_choice == "GraphRAG" else None
    return triples_to_html(triples, community_map=community_map) + build_legend_html(community_map=community_map)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def refresh_leaderboard() -> tuple:
    from leaderboard import build_latency_chart, build_metric_chart, has_real_results, load_summary_df

    df = load_summary_df()
    metric_fig = build_metric_chart()
    latency_fig = build_latency_chart()
    status = "✅ Benchmark results loaded." if has_real_results() else "⚠️ No results found."
    return df, metric_fig, latency_fig, status


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_answer(ans: str, latency_ms: float | None, demo: bool = False) -> str:
    tag = " _(demo)_" if demo else (f"  `{latency_ms:.0f} ms`" if latency_ms else "")
    return f"**{ans}**{tag}"


def _fmt_triples(triples: list) -> str:
    if not triples:
        return "_No context retrieved._"
    lines = [f"- **{s}** —[{r}]→ **{o}**" for s, r, o in triples[:20]]
    if len(triples) > 20:
        lines.append(f"_…and {len(triples) - 20} more_")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_HEADER = """
<div style="text-align:center;padding:24px 0 8px">
  <h1 style="margin:0;font-size:2.2em">🔬 GraphBench</h1>
  <p style="margin:6px 0 0;font-size:1.1em;color:#555">
    Benchmarking Graph-Based RAG Pipelines on Multi-Hop Question Answering
  </p>
</div>
"""

_ABOUT = """
<div style="max-width:860px;margin:0 auto;padding:0 8px 16px">

**GraphBench** compares two knowledge-graph retrieval strategies on 500 HotpotQA distractor questions:

| Pipeline | Retrieval strategy | EM | F1 |
|----------|-------------------|:--:|:--:|
| **GraphRAG** | Louvain community detection on 2-hop Neo4j subgraph | 3.2% | 10.5% |
| **GNN-RAG** | 3-layer GAT edge scoring on 2-hop Neo4j subgraph | **5.0%** | **12.8%** |

**Knowledge graph:** ~60k REBEL triples (Babelscape/rebel-dataset) · Neo4j AuraDB
**LLM:** Mistral-7B-Instruct-v0.2 (fp16) · **Embeddings:** all-MiniLM-L6-v2
**GNN:** 3-layer GAT · 4 heads · hidden=256 · test AUC-ROC = 0.7697

<div style="margin-top:12px;display:flex;gap:18px;flex-wrap:wrap;align-items:center">
  <a href="https://github.com/Rohanjain2312/graphbench" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Rohanjain2312%2Fgraphbench-181717?logo=github" />
  </a>
  <a href="https://pypi.org/project/graphbench-kg/" target="_blank">
    <img src="https://img.shields.io/pypi/v/graphbench-kg?label=PyPI" />
  </a>
  <a href="https://huggingface.co/rohanjain2312" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-rohanjain2312-orange" />
  </a>
  <a href="https://www.linkedin.com/in/jaroh23/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-jaroh23-0077B5?logo=linkedin" />
  </a>
  <a href="mailto:jaroh23@umd.edu">
    <img src="https://img.shields.io/badge/Email-jaroh23%40umd.edu-D14836?logo=gmail&logoColor=white" />
  </a>
</div>

</div>
"""

_DEMO_NOTE = """
> **Running in demo mode** — live inference requires GPU + Neo4j AuraDB.
> Tab 1 shows pre-baked answers for the 10 sample questions.
> Tab 3 shows real benchmark results from the full run.
> [Run your own instance →](https://github.com/Rohanjain2312/graphbench)
"""


def build_app():
    import gradio as gr

    pipeline_status = _try_load_pipelines()
    sample_questions = [item["question"] for item in _DEMO_QA]

    with gr.Blocks(
        title="GraphBench — Graph-Based RAG Benchmark",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
        ),
        css="""
            .gr-button-primary { font-weight: 600; }
            .pipeline-col { border-radius: 8px; padding: 12px; }
            footer { display: none !important; }
        """,
    ) as demo_app:

        state = gr.State({})

        gr.HTML(_HEADER)
        gr.Markdown(_ABOUT)

        # ── Tab 1: Live Q&A ──────────────────────────────────────────────
        with gr.Tab("🔍 Live Q&A"):
            gr.Markdown(_DEMO_NOTE if not _PIPELINES_LOADED else f"**Status:** {pipeline_status}")

            with gr.Row():
                question_input = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. Where was Marie Curie born?",
                    lines=2,
                    scale=5,
                )
                submit_btn = gr.Button("Ask ▶", variant="primary", scale=1, min_width=80)

            gr.Examples(
                examples=sample_questions,
                inputs=question_input,
                label="Sample questions — click to load",
            )

            with gr.Row(equal_height=True):
                with gr.Column(elem_classes="pipeline-col"):
                    gr.Markdown("### 🟦 GraphRAG\n_Louvain community detection_")
                    graphrag_answer = gr.Markdown("_Submit a question to see the answer._")
                    with gr.Accordion("Retrieved context triples", open=False):
                        graphrag_context = gr.Markdown("_No context yet._")

                with gr.Column(elem_classes="pipeline-col"):
                    gr.Markdown("### 🟧 GNN-RAG\n_3-layer GAT edge scoring_")
                    gnnrag_answer = gr.Markdown("_Submit a question to see the answer._")
                    with gr.Accordion("Retrieved context triples", open=False):
                        gnnrag_context = gr.Markdown("_No context yet._")

            for trigger in [submit_btn.click, question_input.submit]:
                trigger(
                    fn=answer,
                    inputs=[question_input, state],
                    outputs=[graphrag_answer, gnnrag_answer, graphrag_context, gnnrag_context, state],
                )

        # ── Tab 2: Graph View ────────────────────────────────────────────
        with gr.Tab("🕸️ Graph View"):
            gr.Markdown(
                "Interactive force-directed visualisation of the knowledge-graph subgraph "
                "retrieved for your last question. **Submit a question on the Live Q&A tab first.**\n\n"
                "- 🟦 GraphRAG: nodes coloured by Louvain community\n"
                "- 🟧 GNN-RAG: edge opacity proportional to GAT attention score"
            )
            with gr.Row():
                pipeline_radio = gr.Radio(
                    choices=["GraphRAG", "GNN-RAG"],
                    value="GraphRAG",
                    label="Pipeline to visualise",
                )
                render_btn = gr.Button("Render Graph", variant="secondary")
            graph_html = gr.HTML(
                "<p style='color:#888;padding:16px'>Submit a question on the Live Q&A tab first.</p>"
            )
            render_btn.click(fn=render_graph, inputs=[pipeline_radio, state], outputs=[graph_html])

        # ── Tab 3: Leaderboard ───────────────────────────────────────────
        with gr.Tab("🏆 Leaderboard"):
            gr.Markdown(
                "### Benchmark Results\n"
                "Full run: 500 HotpotQA distractor questions · seed=42 · "
                "~60k REBEL triples · Mistral-7B-Instruct-v0.2 (fp16)"
            )

            leaderboard_status = gr.Markdown("")
            leaderboard_table = gr.Dataframe(
                headers=["Pipeline", "EM (%)", "F1 (%)", "Latency P50 (ms)", "Latency P95 (ms)", "N Questions"],
                label="Summary",
                interactive=False,
            )
            with gr.Row():
                metric_chart = gr.Plot(label="Exact Match & F1")
                latency_chart = gr.Plot(label="Latency")

            refresh_btn = gr.Button("Refresh", variant="secondary", size="sm")
            refresh_btn.click(
                fn=refresh_leaderboard,
                inputs=[],
                outputs=[leaderboard_table, metric_chart, latency_chart, leaderboard_status],
            )
            demo_app.load(
                fn=refresh_leaderboard,
                inputs=[],
                outputs=[leaderboard_table, metric_chart, latency_chart, leaderboard_status],
            )

        # ── Footer ───────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center;padding:20px 0 8px;color:#888;font-size:0.85em;border-top:1px solid #e5e5e5;margin-top:16px">
          Built by <a href="https://huggingface.co/rohanjain2312" target="_blank">Rohan Jain</a> ·
          MS Applied ML @ University of Maryland ·
          <a href="https://www.linkedin.com/in/jaroh23/" target="_blank">LinkedIn</a> ·
          <a href="mailto:jaroh23@umd.edu">jaroh23@umd.edu</a> ·
          <a href="https://github.com/Rohanjain2312/graphbench" target="_blank">GitHub</a> ·
          <a href="https://pypi.org/project/graphbench-kg/" target="_blank">PyPI</a>
        </div>
        """)

    return demo_app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
