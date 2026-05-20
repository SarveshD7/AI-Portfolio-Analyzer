import json
import os
from typing import TypedDict, Annotated

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
        Dict with total_return_pct (float), daily_returns (last 30 days), and period used
    """
    return calculate_portfolio_returns(tickers, weights, period)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    portfolio: dict       # {"tickers": [...], "weights": [...], "period": "1y"}
    question: str
    period: str           # resolved period (extracted or default)
    messages: list
    tool_results: dict
    needs_tool: bool


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def analyze_question(state: AgentState) -> AgentState:
    """Determine if the question needs the returns tool and extract the period."""
    prompt = (
        f'Analyze this portfolio question and respond with JSON only — no other text.\n\n'
        f'Question: "{state["question"]}"\n\n'
        f'Respond with exactly this structure:\n'
        f'{{"needs_tool": true, "period": "1y"}}\n\n'
        f'Rules:\n'
        f'- needs_tool: true when the question is about returns, performance, gains, losses, or portfolio value.\n'
        f'  false for anything unrelated (weather, cooking, general chat, etc.).\n'
        f'- period: extract from the question, otherwise use the default "{state["period"]}".\n\n'
        f'Period mapping:\n'
        f'  "1 month" / "30 days"           → "1mo"\n'
        f'  "3 months" / "quarter"           → "3mo"\n'
        f'  "6 months" / "half year"         → "6mo"\n'
        f'  "1 year" / "last year" / "past year" / "yearly" → "1y"\n'
        f'  "3 years"                        → "3y"\n'
        f'  "5 years"                        → "5y"'
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        content = response.content.strip()
        # Strip markdown fences if the model wrapped the JSON
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content.strip())
        needs_tool = bool(parsed.get("needs_tool", True))
        period = parsed.get("period", state["period"])
    except (json.JSONDecodeError, KeyError, IndexError):
        needs_tool = True
        period = state["period"]

    return {**state, "needs_tool": needs_tool, "period": period}


def call_tool(state: AgentState) -> AgentState:
    """Call calculate_portfolio_returns with the resolved period."""
    tickers = state["portfolio"]["tickers"]
    weights = state["portfolio"]["weights"]
    result = calculate_portfolio_returns(tickers, weights, state["period"])
    return {**state, "tool_results": result}


def generate_response(state: AgentState) -> AgentState:
    """Use the LLM to produce a conversational answer."""
    if state.get("tool_results"):
        data = state["tool_results"]
        holdings = "\n".join(
            f"  - {t}: {w * 100:.1f}%"
            for t, w in zip(state["portfolio"]["tickers"], state["portfolio"]["weights"])
        )
        prompt = (
            f"You are a portfolio analysis assistant.\n\n"
            f"Portfolio:\n{holdings}\n\n"
            f'Question: "{state["question"]}"\n\n'
            f"Analysis result:\n"
            f"  Period : {data['period']}\n"
            f"  Total Return: {data['total_return_pct']}%\n\n"
            f"Write 2–3 conversational sentences that directly answer the question using these numbers."
        )
    else:
        prompt = (
            f'The user asked: "{state["question"]}"\n\n'
            f"This is not a portfolio question. Politely decline in one sentence and explain "
            f"you can only help with portfolio returns, performance, and related financial analysis."
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
# Public interface
# ---------------------------------------------------------------------------

def run_agent(portfolio: dict, question: str) -> dict:
    """Run the LangGraph agent.

    Args:
        portfolio: {"tickers": [...], "weights": [...], "period": "1y"}
        question: Natural language question about the portfolio

    Returns:
        {"response": str, "data": dict, "visualization": str | None}
    """
    initial_state = {
        "portfolio": portfolio,
        "question": question,
        "period": portfolio.get("period", "1y"),
        "messages": [],
        "tool_results": {},
        "needs_tool": True,
    }

    final_state = _graph.invoke(initial_state)

    return {
        "response": final_state["messages"][-1].content if final_state["messages"] else "",
        "data": final_state.get("tool_results", {}),
        "visualization": "line_chart" if final_state.get("tool_results") else None,
    }
