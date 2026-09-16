"""TruthCheck v2 LangGraph StateMachine Workflow Compilation."""

from __future__ import annotations

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import TruthCheckState
from graph.nodes import (
    ingest_node,
    route_and_decompose_node,
    hybrid_retrieve_and_rerank_node,
    evidence_and_calibration_node,
    assess_risk_node,
    human_review_node,
    generate_response_node
)


def route_decision_edge(state: TruthCheckState) -> Literal["generate_response", "hybrid_retrieve_and_rerank"]:
    if state.route == "medical_advice":
        return "generate_response"
    return "hybrid_retrieve_and_rerank"


def review_decision_edge(state: TruthCheckState) -> Literal["human_review", "generate_response"]:
    if state.needs_human_review:
        return "human_review"
    return "generate_response"


def create_truthcheck_graph(checkpointer=None):
    """Compiles the LangGraph verification workflow."""
    builder = StateGraph(TruthCheckState)

    # Add Nodes
    builder.add_node("ingest", ingest_node)
    builder.add_node("route_and_decompose", route_and_decompose_node)
    builder.add_node("hybrid_retrieve_and_rerank", hybrid_retrieve_and_rerank_node)
    builder.add_node("evidence_and_calibration", evidence_and_calibration_node)
    builder.add_node("assess_risk", assess_risk_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("generate_response", generate_response_node)

    # Add Edges
    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "route_and_decompose")

    # Medical routing gate conditional edge
    builder.add_conditional_edges(
        "route_and_decompose",
        route_decision_edge,
        {
            "generate_response": "generate_response",
            "hybrid_retrieve_and_rerank": "hybrid_retrieve_and_rerank"
        }
    )

    builder.add_edge("hybrid_retrieve_and_rerank", "evidence_and_calibration")
    builder.add_edge("evidence_and_calibration", "assess_risk")

    # Human review conditional edge
    builder.add_conditional_edges(
        "assess_risk",
        review_decision_edge,
        {
            "human_review": "human_review",
            "generate_response": "generate_response"
        }
    )

    builder.add_edge("human_review", "generate_response")
    builder.add_edge("generate_response", END)

    saver = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=saver)


# Compiled default graph app instance
graph_app = create_truthcheck_graph()
