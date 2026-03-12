"""Interactive knowledge graph visualisation for the Gradio demo.

Uses pyvis to render a NetworkX subgraph as an interactive HTML widget
embedded in the Gradio interface. Nodes are coloured by community
(GraphRAG) or by GAT attention score (GNN-RAG).

Usage::

    from demo.graph_viz import triples_to_html

    html = triples_to_html(triples, community_map={"Alice": 0, "Bob": 1})
    html = triples_to_html(triples, score_map={"Alice->Bob": 0.9})
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Colour palettes for up to 10 communities
_COMMUNITY_COLOURS = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
]
_DEFAULT_NODE_COLOUR = "#97c2fc"
_HIGH_SCORE_COLOUR = "#e15759"  # red  — high GAT score
_LOW_SCORE_COLOUR = "#d3e6f5"  # pale blue — low GAT score


def triples_to_html(
    triples: list[tuple[str, str, str]],
    *,
    community_map: dict[str, int] | None = None,
    score_map: dict[str, float] | None = None,
    height: str = "500px",
    width: str = "100%",
) -> str:
    """Render a list of knowledge graph triples as an interactive pyvis HTML graph.

    Colour strategy (mutually exclusive, community_map takes priority):
    - ``community_map``: nodes coloured by Louvain community ID (GraphRAG mode).
    - ``score_map``: edge colours interpolated from pale blue (low) to red (high)
      based on normalised GAT attention score (GNN-RAG mode).
    - Neither: all nodes use the default blue colour.

    Args:
        triples: List of ``(subject, relation, object)`` tuples to display.
        community_map: Optional ``{entity: community_id}`` from CommunityDetector.
        score_map: Optional ``{"{subject}→{object}": score}`` from GATModel.
        height: CSS height of the rendered canvas (default ``"500px"``).
        width: CSS width of the rendered canvas (default ``"100%"``).

    Returns:
        HTML string suitable for embedding in ``gr.HTML``.
        Returns an error message string if pyvis is not installed.
    """
    try:
        from pyvis.network import Network  # noqa: PLC0415
    except ImportError:
        return "<p>pyvis not installed. Run: pip install pyvis</p>"

    if not triples:
        return "<p>No triples to display.</p>"

    net = Network(height=height, width=width, directed=True, notebook=False)
    net.set_options(_PYVIS_OPTIONS)

    # Collect unique entities
    entities: set[str] = set()
    for subj, _, obj in triples:
        entities.add(subj)
        entities.add(obj)

    # Assign node colours
    for entity in entities:
        colour = _node_colour(entity, community_map)
        net.add_node(entity, label=entity, color=colour, title=entity)

    # Add edges
    for subj, rel, obj in triples:
        edge_key = f"{subj}→{obj}"
        score = score_map.get(edge_key) if score_map else None
        edge_colour = _edge_colour(score, score_map)
        title = f"{rel}" + (f" (score: {score:.3f})" if score is not None else "")
        net.add_edge(subj, obj, label=rel, title=title, color=edge_colour, arrows="to")

    try:
        html = net.generate_html(notebook=False)
        return html
    except Exception as exc:
        logger.warning("pyvis HTML generation failed: %s", exc)
        return f"<p>Graph rendering failed: {exc}</p>"


def build_legend_html(
    community_map: dict[str, int] | None = None,
    score_map: dict[str, float] | None = None,
) -> str:
    """Build a simple HTML colour legend for the graph.

    Args:
        community_map: If provided, generate community colour legend.
        score_map: If provided, generate score gradient legend.

    Returns:
        HTML string for a legend panel, or empty string if no legend needed.
    """
    if community_map:
        n_communities = len(set(community_map.values()))
        items = []
        for cid in range(min(n_communities, len(_COMMUNITY_COLOURS))):
            colour = _COMMUNITY_COLOURS[cid % len(_COMMUNITY_COLOURS)]
            items.append(
                f'<span style="display:inline-block;width:14px;height:14px;'
                f'background:{colour};border-radius:50%;margin-right:4px;"></span>'
                f"Community {cid}"
            )
        joined = "&nbsp;&nbsp;".join(items)
        return f'<div style="margin-top:8px;font-size:0.85em">{joined}</div>'

    if score_map:
        return (
            '<div style="margin-top:8px;font-size:0.85em">'
            '<span style="display:inline-block;width:60px;height:10px;'
            "background:linear-gradient(to right,#d3e6f5,#e15759);"
            'vertical-align:middle;margin-right:6px;"></span>'
            "Edge score: low → high"
            "</div>"
        )
    return ""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _node_colour(entity: str, community_map: dict[str, int] | None) -> str:
    if community_map is None:
        return _DEFAULT_NODE_COLOUR
    cid = community_map.get(entity, -1)
    if cid < 0:
        return _DEFAULT_NODE_COLOUR
    return _COMMUNITY_COLOURS[cid % len(_COMMUNITY_COLOURS)]


def _edge_colour(score: float | None, score_map: dict[str, float] | None) -> str:
    """Interpolate between pale blue and red based on normalised score."""
    if score is None or score_map is None:
        return "#848484"
    all_scores = list(score_map.values())
    if not all_scores:
        return "#848484"
    min_s, max_s = min(all_scores), max(all_scores)
    if max_s == min_s:
        t = 0.5
    else:
        t = (score - min_s) / (max_s - min_s)
    # Interpolate RGB: pale blue (211,230,245) → red (225,87,89)
    r = int(211 + t * (225 - 211))
    g = int(230 + t * (87 - 230))
    b = int(245 + t * (89 - 245))
    return f"#{r:02x}{g:02x}{b:02x}"


# Physics + display options for a clean force-directed layout
_PYVIS_OPTIONS = """
{
  "physics": {
    "enabled": true,
    "forceAtlas2Based": {
      "gravitationalConstant": -50,
      "centralGravity": 0.01,
      "springLength": 100,
      "springConstant": 0.08
    },
    "solver": "forceAtlas2Based",
    "stabilization": {"iterations": 150}
  },
  "edges": {
    "font": {"size": 10, "align": "middle"},
    "smooth": {"type": "continuous"}
  },
  "nodes": {
    "font": {"size": 12},
    "shape": "dot",
    "size": 16
  },
  "interaction": {
    "navigationButtons": true,
    "zoomView": true,
    "tooltipDelay": 100
  }
}
"""
