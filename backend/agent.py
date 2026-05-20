import json
import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from tools.portfolio_info import get_portfolio_composition
from tools.returns import calculate_portfolio_returns
from tools.risk import calculate_sharpe_ratio, calculate_max_drawdown

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))


@tool
def portfolio_returns_tool(tickers: list, weights: list, period: str = "1y") -> dict:
    """Calculate historical returns for a portfolio of stocks.
    Use when the user asks about returns, performance, gains, losses, or how the portfolio did.

    Args:
        tickers: List of stock ticker symbols (e.g. ['RELIANCE.NS', 'TCS.NS'])
        weights: List of portfolio weights corresponding to each ticker (must sum to 1.0)
        period: Time period — one of '1mo', '3mo', '6mo', '1y', '3y', '5y'
    """
    return calculate_portfolio_returns(tickers, weights, period)


@tool
def sharpe_ratio_tool(tickers: list, weights: list, period: str = "1y") -> dict:
    """Calculates Sharpe ratio (risk-adjusted return metric), annualized return, and volatility.
    Use when the user asks about risk, Sharpe ratio, volatility, or risk-adjusted performance.

    Args:
        tickers: List of stock ticker symbols (e.g. ['RELIANCE.NS', 'TCS.NS'])
        weights: List of portfolio weights corresponding to each ticker (must sum to 1.0)
        period: Time period — one of '1mo', '3mo', '6mo', '1y', '3y', '5y'
    """
    return calculate_sharpe_ratio(tickers, weights, period)


@tool
def max_drawdown_tool(tickers: list, weights: list, period: str = "1y") -> dict:
    """Calculates maximum drawdown — the largest peak-to-trough decline in portfolio value.
    Use when the user asks about worst loss, biggest drop, drawdown, or how bad things got.

    Args:
        tickers: List of stock ticker symbols (e.g. ['RELIANCE.NS', 'TCS.NS'])
        weights: List of portfolio weights corresponding to each ticker (must sum to 1.0)
        period: Time period — one of '1mo', '3mo', '6mo', '1y', '3y', '5y'
    """
    return calculate_max_drawdown(tickers, weights, period)


@tool
def portfolio_composition_tool(tickers: list, weights: list) -> dict:
    """Shows portfolio composition and individual stock holdings with weights.
    Use when user asks 'what's in my portfolio', 'show my holdings', 'how much X do I have', or 'what do I own'.

    Args:
        tickers: List of stock ticker symbols (e.g. ['RELIANCE.NS', 'TCS.NS'])
        weights: List of portfolio weights corresponding to each ticker (must sum to 1.0)
    """
    return get_portfolio_composition(tickers, weights)


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
    visualization_type: str   # "line_chart" | "metrics" | "pie_chart" | "drawdown_chart" | "portfolio_pie" | ""
    tool_name: str            # "returns" | "sharpe" | "drawdown" | "portfolio_info" | ""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def analyze_question(state: AgentState) -> AgentState:
    """Determine tool, period, and visualization type from the question."""
    prompt = (
        f'Analyze this portfolio question and respond with JSON only — no other text.\n\n'
        f'Question: "{state["question"]}"\n\n'
        f'Respond with exactly this structure:\n'
        f'{{"needs_tool": true, "period": "1y", "visualization_type": "line_chart", "tool_name": "returns"}}\n\n'
        f'Rules for needs_tool:\n'
        f'- true: question is about returns, performance, gains, losses, risk, sharpe, volatility, allocation, or holdings.\n'
        f'- false: unrelated question (weather, cooking, etc.).\n\n'
        f'Rules for period (extract from question, else use default "{state["period"]}"):\n'
        f'  "1 month" / "30 days"                           → "1mo"\n'
        f'  "3 months" / "quarter"                          → "3mo"\n'
        f'  "6 months" / "half year"                        → "6mo"\n'
        f'  "1 year" / "last year" / "past year" / "yearly" → "1y"\n'
        f'  "3 years"                                       → "3y"\n'
        f'  "5 years"                                       → "5y"\n\n'
        f'Rules for tool_name + visualization_type:\n'
        f'  "return"/"performance"/"gain"/"loss"/"how did"/"trend"  → tool_name:"returns",  visualization_type:"line_chart"\n'
        f'  "sharpe"/"risk"/"volatility"/"risky"/"risk-adjusted"    → tool_name:"sharpe",   visualization_type:"metrics"\n'
        f'  "drawdown"/"worst loss"/"biggest drop"/"max drop"      → tool_name:"drawdown", visualization_type:"drawdown_chart"\n'
        f'  "allocation"/"sector"/"breakdown"/"diversification"       → tool_name:"",              visualization_type:"pie_chart"\n'
        f'  "what\'s in"/"holdings"/"what do I own"/"show stocks"   → tool_name:"portfolio_info", visualization_type:"portfolio_pie"\n'
        f'  unrelated question (needs_tool:false)                   → tool_name:"",              visualization_type:""\n'
        f'  default for any other portfolio question                 → tool_name:"returns",  visualization_type:"line_chart"'
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
        tool_name = parsed.get("tool_name", "returns")
    except (json.JSONDecodeError, KeyError, IndexError):
        needs_tool = True
        period = state["period"]
        visualization_type = "line_chart"
        tool_name = "returns"

    return {**state, "needs_tool": needs_tool, "period": period,
            "visualization_type": visualization_type, "tool_name": tool_name}


def call_tool(state: AgentState) -> AgentState:
    """Dispatch to the correct tool based on tool_name."""
    if state.get("visualization_type") == "pie_chart":
        return state

    tickers = state["portfolio"]["tickers"]
    weights = state["portfolio"]["weights"]
    period = state["period"]
    tool_name = state.get("tool_name", "returns")

    if tool_name == "sharpe":
        result = calculate_sharpe_ratio(tickers, weights, period)
    elif tool_name == "drawdown":
        result = calculate_max_drawdown(tickers, weights, period)
    elif tool_name == "portfolio_info":
        result = get_portfolio_composition(tickers, weights)
    else:
        result = calculate_portfolio_returns(tickers, weights, period)

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
        holdings_text = "\n".join(
            f"  - {t}: {w * 100:.1f}%"
            for t, w in zip(state["portfolio"]["tickers"], state["portfolio"]["weights"])
        )
        prompt = (
            f"You are a portfolio analysis assistant.\n\n"
            f"Portfolio allocation:\n{holdings_text}\n\n"
            f'Question: "{state["question"]}"\n\n'
            f"Write 2 conversational sentences describing this allocation."
        )
    elif viz_type == "portfolio_pie" and tool_results:
        lines = "\n".join(
            f"  - {h['name']} ({h['ticker']}): {h['weight_pct']}%"
            for h in tool_results.get("holdings", [])
        )
        prompt = (
            f"You are a portfolio analysis assistant.\n\n"
            f'Question: "{state["question"]}"\n\n'
            f"Portfolio holdings:\n{lines}\n\n"
            f"Write 2–3 conversational sentences summarising what the user owns, "
            f"mentioning the largest position."
        )
    elif viz_type == "drawdown_chart" and tool_results:
        d = tool_results
        status = "not yet recovered" if d.get("currently_in_drawdown") else f"recovered on {d.get('recovery_date')}"
        prompt = (
            f"You are a portfolio analysis assistant.\n\n"
            f'Question: "{state["question"]}"\n\n'
            f"Max drawdown analysis for {d['period']}:\n"
            f"  Max Drawdown : {d['max_drawdown_pct']}%\n"
            f"  Peak Date    : {d['peak_date']}\n"
            f"  Trough Date  : {d['trough_date']}\n"
            f"  Recovery     : {status}\n\n"
            f"Write 2–3 sentences interpreting this drawdown for the user."
        )
    elif viz_type == "metrics" and tool_results:
        d = tool_results
        holdings = "\n".join(
            f"  - {t}: {w * 100:.1f}%"
            for t, w in zip(state["portfolio"]["tickers"], state["portfolio"]["weights"])
        )
        prompt = (
            f"You are a portfolio analysis assistant.\n\n"
            f"Portfolio:\n{holdings}\n\n"
            f'Question: "{state["question"]}"\n\n'
            f"Risk metrics for {d['period']}:\n"
            f"  Sharpe Ratio         : {d['sharpe_ratio']}\n"
            f"  Annualized Return    : {d['annualized_return_pct']}%\n"
            f"  Annualized Volatility: {d['annualized_volatility_pct']}%\n"
            f"  Risk-Free Rate       : {d['risk_free_rate_pct']}%\n\n"
            f"Write 2–3 sentences interpreting these numbers for the user."
        )
    elif tool_results:
        d = tool_results
        holdings = "\n".join(
            f"  - {t}: {w * 100:.1f}%"
            for t, w in zip(state["portfolio"]["tickers"], state["portfolio"]["weights"])
        )
        prompt = (
            f"You are a portfolio analysis assistant.\n\n"
            f"Portfolio:\n{holdings}\n\n"
            f'Question: "{state["question"]}"\n\n'
            f"Result — Period: {d['period']}, Total Return: {d['total_return_pct']}%\n\n"
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
        return {"type": "metrics", "data": tool_results}

    if viz_type == "drawdown_chart" and tool_results:
        return {"type": "drawdown_chart", "data": tool_results}

    if viz_type == "portfolio_pie" and tool_results:
        return {"type": "portfolio_pie", "data": tool_results}

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
        "tool_name": "returns",
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
