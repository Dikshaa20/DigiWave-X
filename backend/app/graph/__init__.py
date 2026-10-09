"""app/graph/__init__.py — LangGraph graph for the DigiWave X tutor.

Exports the entry points used by the API and tests:

    make_graph() -> StateGraph

The graph is a strict sequential pipeline with a single conditional edge at
the verify step. The ten nodes are workflow nodes, not ten separate agents,
so a single request never runs all ten: the router decides the path, and the
verify_gate loops only on failure.
"""

from app.graph.graph import make_graph

__all__ = ["make_graph"]
