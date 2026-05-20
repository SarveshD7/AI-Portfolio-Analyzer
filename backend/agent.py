import json
import os
from typing import TypedDict, Annotated

import numpy as np
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from tools.returns import calculate_portfolio_returns

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))


@tool
def portfolio_returns_tool(tickers: list, weights: list, period: str = "1y") -> dict:
    """Calculate historical returns for a portfolio of stocks.

    Args:
        tickers: List of stock ticker symbols (e.g. ['RELIANCE.NS', 'TCS.NS'])
        weights: List of portfolio weights corresponding to each ticker (must sum to 1.0)
        period: Time period for calculation — one of '1mo', '3mo', '6mo', '1y', '3y', '5y'

    Returns:
        Dict with total_return_pct (float), daily_returns, and period used
    """
    return calculate_portfolio_returns(tickers, weights, period)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    portfolio: dict           # {"tickers": [...], "weights": [...], "period": "1y"}
    question: str
    period: str               # resolved period (extracted or default)
    messages: list
    tool_results: dict
    needs_tool: bool
    visualization_type: str   # "line_chart" | "metrics" | "pie_chart" | ""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def analyze_question(state: AgentState) -> AgentState:
    """Determine tool need, period, and visualization type from the question."""
    prompt = (
        f'Analyze this portfolio question and respond with JSON only — no other text.\n\n'
        f'Question: "{state["question"]}"\n\n'
        f'Respond with exactly this structure:\n'
        f'{{"needs_tool": true, "period": "1y", "visualization_type": "line_chart"}}\n\n'
        f'Rules for needs_tool:\n'
        f'- true: question is about returns, performance, gains, losses, risk, sharpe, volatility, or allocation.\n'
        f'- false: unrelated question (weather, cooking, etc.).\n\n'
        f'Rules for period (extract from question, else use default "{state["period"]}"):\n'
        f'  "1 month" / "30 days"                          → "1mo"\n'
        f'  "3 months" / "quarter"                         → "3mo"\n'
        f'  "6 months" / "half year"                       → "6mo"\n'
        f'  "1 year" / "last year" / "past year" / "yearly"→ "1y"\n'
        f'  "3 years"                                      → "3y"\n'
        f'  "5 years"                                      → "5y"\n\n'
        f'Rules for visualization_type:\n'
        f'- "line_chart": returns / performance / gains / losses / how did / trend\n'
        f'- "metrics"   : sharpe / risk / volatility / standard deviation / metrics\n'
        f'- "pie_chart" : allocation / sector / breakdown / diversification / weights\n'
        f'- "line_chart": default for any other portfolio question\n'
        f'- ""          : not a portfolio question (needs_tool is false)'
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content.strip())
        needs_tool = bool(parsed.get("needs_tool", True))
        period = parsed.get("period", state["period"])
        visualization_type = parsed.get("visualization_type", "line_chart")
    except (json.JSONDecodeError, KeyError, IndexError):
        needs_tool = True
        period = state["period"]
        visualization_type = "line_chart"

    return {**state, "needs_tool": needs_tool, "period": period, "visualization_type": visualization_type}


def call_tool(state: AgentState) -> AgentState:
    """Call calculate_portfolio_returns with the resolved period."""
    tickers = state["portfolio"]["tickers"]
    weights = state["portfolio"]["weights"]
    # pie_chart questions don't need market data
    if state.get("visualization_type") == "pie_chart":
        return state
    result = calculate_portfolio_returns(tickers, weights, state["period"])
    return {**state, "tool_results": result}


def generate_response(state: AgentState) -> AgentState:
    """Use the LLM to produce a conversational answer."""
    viz_type = state.get("visualization_type", "")
    tool_results = state.get("tool_results", {})

    if not state.get("needs_tool"):
        prompt = (
            f'The user asked: "{state["question"]}"\n\n'
            f"This is not a portfolio question. Politely decline in one sentence."
        )
    elif viz_type == "pie_chart":
        holdings = "\n".join(
            f"  - {t}: {w * 100:.1f}%"
            for t, w in zip(state["portfolio"]["tickers"], state["portfolio"]["weights"])
        )
        prompt = (
            f"You are a portfolio analysis assistant.\n\n"
            f"Portfolio allocation:\n{holdings}\n\n"
            f'Question: "{state["question"]}"\n\n'
            f"Write 2 conversational sentences describing this allocation."
        )
    elif viz_type == "metrics" and tool_results:
        data = tool_results
        holdings = "\n".join(
            f"  - {t}: {w * 100:.1f}%"
            for t, w in zip(state["portfolio"]["tickers"], state["portfolio"]["weights"])
        )
        prompt = (
            f"You are a portfolio analysis assistant.\n\n"
            f"Portfolio:\n{holdings}\n\n"
            f'Question: "{state["question"]}"\n\n'
            f"Risk/return metrics for {data['period']}:\n"
            f"  Total Return     : {data['total_return_pct']}%\n\n"
            f"Write 2–3 sentences interpreting the risk and return profile."
        )
    elif tool_results:
        data = tool_results
        holdings = "\n".join(
            f"  - {t}: {w * 100:.1f}%"
            for t, w in zip(state["portfolio"]["tickers"], state["portfolio"]["weights"])
        )
        prompt = (
            f"You are a portfolio analysis assistant.\n\n"
            f"Portfolio:\n{holdings}\n\n"
            f'Question: "{state["question"]}"\n\n'
            f"Result — Period: {data['period']}, Total Return: {data['total_return_pct']}%\n\n"
            f"Write 2–3 conversational sentences directly answering the question with these numbers."
        )
    else:
        prompt = (
            f'The user asked: "{state["question"]}"\n\n'
            f"No data is available. Politely explain you can only help with portfolio analysis."
        )

    response = llm.invoke([HumanMessage(content=prompt)])
    updated_messages = state.get("messages", []) + [AIMessage(content=response.content)]
    return {**state, "messages": updated_messages}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_analysis(state: AgentState) -> str:
    return "call_tool" if state.get("needs_tool") else "generate_response"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def _build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("analyze_question", analyze_question)
    graph.add_node("call_tool", call_tool)
    graph.add_node("generate_response", generate_response)

    graph.set_entry_point("analyze_question")

    graph.add_conditional_edges(
        "analyze_question",
        route_after_analysis,
        {"call_tool": "call_tool", "generate_response": "generate_response"},
    )
    graph.add_edge("call_tool", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()


_graph = _build_graph()


# ---------------------------------------------------------------------------
# Visualization builder
# ---------------------------------------------------------------------------

def _build_visualization(viz_type: str, tool_results: dict, portfolio: dict) -> dict:
    if viz_type == "line_chart" and tool_results:
        return {
            "type": "line_chart",
            "data": {
                "daily_returns": tool_results.get("daily_returns", []),
                "period": tool_results.get("period", ""),
                "total_return_pct": tool_results.get("total_return_pct", 0),
            },
        }

    if viz_type == "metrics" and tool_results:
        daily = tool_results.get("daily_returns", [])
        raw = [d["return"] / 100 for d in daily]

        vol_annual = float(np.std(raw)) * (252 ** 0.5) * 100 if raw else 0
        total_return = tool_results.get("total_return_pct", 0)
        period = tool_results.get("period", "1y")
        years = {"1mo": 1 / 12, "3mo": 0.25, "6mo": 0.5, "1y": 1, "3y": 3, "5y": 5}.get(period, 1)
        ann_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else total_return
        risk_free = 6.5  # Indian T-bill approximation
        sharpe = (ann_return - risk_free) / vol_annual if vol_annual > 0 else 0

        return {
            "type": "metrics",
            "data": {
                "total_return_pct": round(total_return, 2),
                "annualized_return_pct": round(ann_return, 2),
                "annualized_volatility_pct": round(vol_annual, 2),
                "sharpe_ratio": round(sharpe, 2),
                "period": period,
            },
        }

    if viz_type == "pie_chart":
        return {
            "type": "pie_chart",
            "data": {
                "tickers": portfolio.get("tickers", []),
                "weights": portfolio.get("weights", []),
            },
        }

    return {"type": None, "data": {}}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_agent(portfolio: dict, question: str) -> dict:
    """Run the LangGraph agent.

    Args:
        portfolio: {"tickers": [...], "weights": [...], "period": "1y"}
        question: Natural language question about the portfolio

    Returns:
        {"response": str, "visualization": {"type": str, "data": dict}}
    """
    initial_state = {
        "portfolio": portfolio,
        "question": question,
        "period": portfolio.get("period", "1y"),
        "messages": [],
        "tool_results": {},
        "needs_tool": True,
        "visualization_type": "line_chart",
    }

    final_state = _graph.invoke(initial_state)

    visualization = _build_visualization(
        final_state.get("visualization_type", ""),
        final_state.get("tool_results", {}),
        portfolio,
    )

    return {
        "response": final_state["messages"][-1].content if final_state["messages"] else "",
        "visualization": visualization,
    }
