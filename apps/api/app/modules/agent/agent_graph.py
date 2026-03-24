"""
LangGraph 1.0 Agent Graph for the Piezo.AI Research Assistant.

Uses StateGraph with MessagesState, ToolNode, and conditional edges.
Follows LangGraph 1.0 conventions (Oct 2025 stable release).
Model-agnostic: works with any LLM provider via init_chat_model.
"""
import logging
from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

logger = logging.getLogger("piezo.agent.graph")

# ── System Prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are PiezoBot, an expert AI research assistant for the Piezo.AI platform.

Your mission is to help researchers discover lead-free piezoelectric materials. You have deep expertise in:
- Piezoelectric ceramics, especially KNN (Potassium Sodium Niobate) systems
- Materials science: crystal structure, phase boundaries, doping effects
- Machine learning for materials property prediction
- Interpreting SHAP values and feature importance for scientific insights

GUIDELINES:
1. Always use your tools when asked about specific materials or predictions — don't guess properties.
2. Cite your confidence level: "The model predicts with R²=0.84..." or "Based on the indexed literature..."
3. When suggesting compositions, explain the materials science rationale (why specific dopants help).
4. If asked about something outside your knowledge, be honest. Suggest using the literature search tool.
5. Format responses with markdown: use **bold** for key values, bullet lists for comparisons, and LaTeX for equations.
6. When comparing materials, use a structured table format.
7. Always mention units: pC/N for d33, °C for Tc, HV for Vickers hardness.
8. If a tool call fails, explain the error clearly and suggest alternatives.

PERSONALITY:
- Scientific and precise, but approachable
- Enthusiastic about lead-free piezoelectrics (it's an environmental mission!)
- Proactive: suggest follow-up analyses the researcher might find useful
"""

# ── State Definition ──────────────────────────────────────────────────

class AgentState(MessagesState):
    """Agent state with message history and thinking steps."""
    thinking_steps: list[str]


# ── Graph Builder ─────────────────────────────────────────────────────

def build_agent_graph(llm, tools):
    """
    Build the LangGraph StateGraph for the Piezo.AI agent.
    
    Graph structure:
        START → agent → tools_condition → tools → agent (loop)
                     → END (if no tool call)
    
    Max iterations capped at 10 to prevent infinite loops.
    """
    logger.info("agent_graph.building")

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        """Main agent reasoning node. Calls the LLM with message history."""
        messages = state["messages"]

        # Prepend system message if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        response = llm_with_tools.invoke(messages)

        # Track thinking step
        thinking = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            thinking.append(f"Deciding to use tools: {', '.join(tool_names)}")

        return {
            "messages": [response],
            "thinking_steps": state.get("thinking_steps", []) + thinking,
        }

    # Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    # Add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    # Compile with recursion limit (max iterations)
    compiled = graph.compile()
    
    logger.info("agent_graph.ready")
    return compiled


# ── Singleton Graph Instance ──────────────────────────────────────────

_graph_instance = None


def get_agent_graph():
    """Get or create the singleton agent graph instance."""
    global _graph_instance
    if _graph_instance is None:
        from apps.api.app.modules.agent.llm_provider import get_chat_model
        from apps.api.app.modules.agent.tools import ALL_TOOLS

        llm = get_chat_model()
        _graph_instance = build_agent_graph(llm, ALL_TOOLS)
    return _graph_instance


def reset_agent_graph():
    """Reset the graph instance (e.g., after config change)."""
    global _graph_instance
    _graph_instance = None
