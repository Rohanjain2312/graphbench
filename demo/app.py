"""Gradio demo application for GraphBench — HuggingFace Spaces entry point.

Three-tab interface:
- Tab 1: Live Q&A — enter a question, see both pipeline answers side-by-side
          with context triples and latency comparison.
- Tab 2: Graph Visualisation — interactive pyvis graph of the retrieved subgraph
          for the last question, rendered via graph_viz.py.
- Tab 3: Leaderboard — aggregated benchmark results table and charts
          from experiments/results/, rendered via leaderboard.py.

Implementation: Phase 7 (demo).
"""
