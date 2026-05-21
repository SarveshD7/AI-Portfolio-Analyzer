import json
import os
from typing import Optional

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from tools.benchmark import benchmark_portfolio
from tools.concentration import analyze_concentration
from tools.correlation import calculate_correlation
from tools.portfolio_info import get_portfolio_composition
from tools.portfolio_modification import modify_portfolio
from tools.returns import calculate_portfolio_returns
from tools.risk import calculate_sharpe_ratio, calculate_max_drawdown, calculate_var
from tools.suggestions import generate_smart_suggestions

load_dotenv()

# ---------------------------------------------------------------------------
# Tools  (docstrings drive LLM tool selection — keep them descriptive)
# ---------------------------------------------------------------------------

@tool
def portfolio_returns_tool(tickers: list, weights: list, period: str = "1y") -> dict:
    """Calculate historical portfolio returns, performance, and per-stock contribution breakdown.

    Use when the user asks about:
    - How the portfolio performed / did
    - Returns, gains, losses, or profit
    - Performance over a specific period
    - "How much did I make / lose?"
    - "Which stocks performed well or poorly?"
    - "What is dragging down my portfolio?"
    - "Which holdings are underperforming?"
    - Individual stock contributions to portfolio performance
    - Performance breakdown by holding

    Args:
        tickers: List of stock tickers (e.g. ["RELIANCE.NS", "TCS.NS"])
        weights: Corresponding weights as decimals summing to 1.0 (e.g. [0.6, 0.4])
        period: "1mo", "3mo", "6mo", "1y", "3y", or "5y"
    """
    return calculate_portfolio_returns(tickers, weights, period)


@tool
def sharpe_ratio_tool(tickers: list, weights: list, period: str = "1y") -> dict:
    """Calculate Sharpe ratio and risk-adjusted return metrics.

    Use when the user asks about:
    - Sharpe ratio or risk-adjusted returns
    - Volatility or standard deviation
    - Whether returns justify the risk taken

    Args:
        tickers: List of stock tickers (e.g. ["RELIANCE.NS", "TCS.NS"])
        weights: Corresponding weights as decimals summing to 1.0 (e.g. [0.6, 0.4])
        period: "1mo", "3mo", "6mo", "1y", "3y", or "5y"
    """
    return calculate_sharpe_ratio(tickers, weights, period)


@tool
def max_drawdown_tool(tickers: list, weights: list, period: str = "1y") -> dict:
    """Calculate maximum drawdown — the worst peak-to-trough decline.

    Use when the user asks about:
    - Drawdown or maximum drawdown
    - Worst loss or biggest historical drop
    - How bad things got / peak and trough dates

    Args:
        tickers: List of stock tickers (e.g. ["RELIANCE.NS", "TCS.NS"])
        weights: Corresponding weights as decimals summing to 1.0 (e.g. [0.6, 0.4])
        period: "1mo", "3mo", "6mo", "1y", "3y", or "5y"
    """
    return calculate_max_drawdown(tickers, weights, period)


@tool
def var_tool(tickers: list, weights: list, period: str = "1y") -> dict:
    """Calculate Value at Risk (VaR) and tail risk metrics.

    Use when the user asks about:
    - Value at Risk (VaR)
    - How much they could lose on a bad day
    - Tail risk, downside risk, or expected shortfall (CVaR)
    - Worst-case daily loss scenarios

    Args:
        tickers: List of stock tickers (e.g. ["RELIANCE.NS", "TCS.NS"])
        weights: Corresponding weights as decimals summing to 1.0 (e.g. [0.6, 0.4])
        period: "1mo", "3mo", "6mo", "1y", "3y", or "5y"
    """
    return calculate_var(tickers, weights, period)


@tool
def benchmark_tool(
    tickers: list,
    weights: list,
    period: str = "1y",
    benchmark: str = "^NSEI",
) -> dict:
    """Compare portfolio performance against a market index benchmark.

    Use when the user asks about:
    - How the portfolio compares to an index (Nifty, Sensex, S&P 500, etc.)
    - Whether the portfolio beat or underperformed the market
    - Alpha or beta relative to a benchmark
    - "Did I beat the Nifty / S&P 500?"
    - Relative performance vs an index

    Args:
        tickers: List of stock tickers (e.g. ["RELIANCE.NS", "TCS.NS"])
        weights: Corresponding weights as decimals summing to 1.0 (e.g. [0.6, 0.4])
        period: "1mo", "3mo", "6mo", "1y", "3y", or "5y"
        benchmark: Index ticker or alias. Common values:
            "^NSEI"    / "nifty50"   → Nifty 50  (default)
            "^BSESN"   / "sensex"    → BSE Sensex
            "^GSPC"    / "sp500"     → S&P 500
            "^IXIC"    / "nasdaq"    → NASDAQ Composite
            "^DJI"     / "dow"       → Dow Jones
            "^NSEBANK" / "niftybank" → Nifty Bank
    """
    return benchmark_portfolio(tickers, weights, period, benchmark)


@tool
def correlation_tool(tickers: list, weights: list, period: str = "1y") -> dict:
    """Compute pairwise correlation between all assets in the portfolio.

    Use when the user asks about:
    - Correlation between stocks / assets / holdings
    - How much their assets move together
    - Whether the portfolio is well-diversified or all moving in sync
    - Which stocks are most / least correlated
    - Concentration risk due to correlated assets
    - Correlation matrix or heatmap

    Args:
        tickers: List of stock tickers (e.g. ["RELIANCE.NS", "TCS.NS"])
        weights: Corresponding weights as decimals summing to 1.0 (e.g. [0.6, 0.4])
        period: "1mo", "3mo", "6mo", "1y", "3y", or "5y"
    """
    return calculate_correlation(tickers, weights, period)


@tool
def concentration_tool(tickers: list, weights: list, breakdown_type: str = "sector") -> dict:
    """Analyze portfolio concentration by sector, asset class, or market-cap factor.

    Use when the user asks about:
    - Sector concentration or sector exposure (e.g., "Am I too heavy in Tech?")
    - Whether they are secretly over-indexed to a single sector
    - Asset class breakdown (equity vs ETF vs crypto vs mutual fund)
    - Factor exposure or market-cap tiers: Large Cap vs Mid Cap vs Small Cap
    - Momentum, Value, or Growth factor concentration (use breakdown_type="factor")
    - Hidden concentration risk or diversification across sectors/classes

    Args:
        tickers: List of stock tickers (e.g. ["RELIANCE.NS", "TCS.NS"])
        weights: Corresponding weights as decimals summing to 1.0 (e.g. [0.6, 0.4])
        breakdown_type: "sector" for sector breakdown, "asset_class" for equity/ETF/crypto,
                        "factor" for Large Cap / Mid Cap / Small Cap market-cap factor tiers
    """
    return analyze_concentration(tickers, weights, breakdown_type)


@tool
def portfolio_composition_tool(tickers: list, weights: list) -> dict:
    """Show portfolio composition with company names, allocation weights, and largest positions.

    Use when the user asks about:
    - What stocks are in the portfolio / what they own
    - Portfolio holdings or composition
    - Allocation or sector breakdown
    - "Which stocks are my largest positions?"
    - "What is dragging my portfolio down?" (position-level view)
    - "How much of stock X do I hold?"
    - Stock-level details, names, or weightings

    Args:
        tickers: List of stock tickers (e.g. ["RELIANCE.NS", "TCS.NS"])
        weights: Corresponding weights as decimals summing to 1.0 (e.g. [0.6, 0.4])
    """
    return get_portfolio_composition(tickers, weights)


@tool
def simulate_portfolio_change_tool(
    tickers: list,
    weights: list,
    remove: Optional[list] = None,
    add: Optional[dict] = None,
    change_weight: Optional[dict] = None,
) -> dict:
    """Simulate what-if portfolio changes and return the rebalanced portfolio.

    Use when the user asks "what if" questions or wants to modify the portfolio:
    - "What if I exit / remove / sell [stock]?"
    - "What if I didn't have [stock] or [sector]?"
    - "What if I added / bought [stock]?"
    - "Replace X with Y"
    - "Increase / decrease [stock] to X%"
    - "Remove all my energy / tech / banking stocks"
    - "What happens if I sell half my TCS?"
    - "Should I swap Reliance for Gold?"

    After calling this tool, tell the user what changed and what the new portfolio
    looks like. All subsequent analysis will automatically use the modified portfolio.

    Args:
        tickers: Current portfolio tickers exactly as shown in the portfolio context
        weights: Current portfolio weights (decimals summing to 1.0)
        remove: Tickers to remove (e.g., ["RELIANCE.NS"])
        add: Tickers to add with target weights (e.g., {"GOLD": 0.15})
        change_weight: Tickers whose weights should change (e.g., {"TCS.NS": 0.40})
    """
    return modify_portfolio(tickers, weights, remove=remove, add=add, change_weight=change_weight)


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

_tools = [
    portfolio_returns_tool,
    sharpe_ratio_tool,
    max_drawdown_tool,
    var_tool,
    portfolio_composition_tool,
    concentration_tool,
    correlation_tool,
    benchmark_tool,
    simulate_portfolio_change_tool,
]

_tool_map = {t.name: t for t in _tools}

_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
_llm_with_tools = _llm.bind_tools(_tools)

_SYSTEM_PROMPT = """You are a portfolio analysis assistant.

Use the available tools to answer questions about the user's portfolio.
The portfolio's tickers and weights are in the human message — always pass
those exact values when calling tools.

Period mapping (extract from the question; default to "1y" if unspecified):
  "1 month" / "30 days"                          → "1mo"
  "3 months" / "quarter"                          → "3mo"
  "6 months" / "half year"                        → "6mo"
  "1 year" / "last year" / "past year" / "yearly" → "1y"
  "3 years"                                        → "3y"
  "5 years"                                        → "5y"

After calling a tool, respond conversationally in 2–3 sentences:
- Explain the results in plain English
- Embed the specific numbers from the tool result
- Provide brief context or interpretation"""

# Maps called tool name → canvas visualization type
_TOOL_VIZ_MAP = {
    "portfolio_returns_tool":     "line_chart",
    "sharpe_ratio_tool":          "metrics",
    "var_tool":                   "metrics",
    "max_drawdown_tool":          "drawdown_chart",
    "portfolio_composition_tool": "portfolio_pie",
    "concentration_tool":         "concentration_pie",
    "correlation_tool":           "correlation_heatmap",
    "benchmark_tool":             "benchmark_chart",
}

# Large array fields that are only needed for visualization, not for the LLM's text response.
# Stripping these from ToolMessages cuts token usage by ~2–5k tokens per call.
_VIZ_ONLY_KEYS = {
    "portfolio_returns_tool": ["daily_returns"],
    "max_drawdown_tool":      ["drawdown_series"],
    "benchmark_tool":         ["portfolio_cumulative", "benchmark_cumulative"],
}

# Maps called tool name → accumulated_analysis key for suggestions
_TOOL_DATA_KEY_MAP = {
    "portfolio_returns_tool":     "returns_data",
    "sharpe_ratio_tool":          "sharpe_data",
    "max_drawdown_tool":          "drawdown_data",
    "var_tool":                   "var_data",
    "portfolio_composition_tool": "portfolio_composition",
    # concentration_tool → "sector_breakdown" only when breakdown_type == "sector"
}


def _get_analysis_key(tool_name: str, tool_results: dict) -> str | None:
    if tool_name == "concentration_tool":
        return "sector_breakdown" if tool_results.get("breakdown_type") == "sector" else None
    return _TOOL_DATA_KEY_MAP.get(tool_name)


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
                "stock_contributions": tool_results.get("stock_contributions", []),
                "best_performer": tool_results.get("best_performer", {}),
                "worst_performer": tool_results.get("worst_performer", {}),
            },
        }
    if viz_type == "metrics" and tool_results:
        return {"type": "metrics", "data": tool_results}
    if viz_type == "drawdown_chart" and tool_results:
        return {"type": "drawdown_chart", "data": tool_results}
    if viz_type == "portfolio_pie" and tool_results:
        return {"type": "portfolio_pie", "data": tool_results}
    if viz_type == "concentration_pie" and tool_results:
        return {"type": "concentration_pie", "data": tool_results}
    if viz_type == "correlation_heatmap" and tool_results:
        return {"type": "correlation_heatmap", "data": tool_results}
    if viz_type == "benchmark_chart" and tool_results:
        return {"type": "benchmark_chart", "data": tool_results}
    return {"type": None, "data": {}}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_agent(
    portfolio: dict,
    question: str,
    accumulated_analysis: dict = None,
    is_initial: bool = False,
) -> dict:
    """Run the tool-calling agent using bind_tools.

    Args:
        portfolio:           {"tickers": [...], "weights": [...], "period": "1y"}
        question:            Natural language question about the portfolio
        accumulated_analysis: Prior tool results keyed by analysis type (for suggestions)
        is_initial:          When True, runs composition + sector directly (no LLM) to seed
                             the canvas and generate first-pass suggestions on upload.

    Returns:
        {"response": str, "visualization": dict, "suggestions": list, "analysis_updates": dict}
    """
    if is_initial:
        comp = get_portfolio_composition(portfolio["tickers"], portfolio["weights"])
        sector = analyze_concentration(portfolio["tickers"], portfolio["weights"], "sector")
        suggestions = generate_smart_suggestions(
            portfolio_composition=comp,
            sector_breakdown=sector,
        )
        return {
            "response": "",
            "visualization": _build_visualization("portfolio_pie", comp, portfolio),
            "suggestions": suggestions,
            "analysis_updates": {"portfolio_composition": comp, "sector_breakdown": sector},
        }

    _FALLBACK_RESPONSE = (
        "I can help with portfolio analysis — including historical returns, risk metrics "
        "(Sharpe ratio, VaR, max drawdown), sector concentration, asset correlation, and "
        "benchmark comparisons. Could you rephrase your question with one of those in mind?"
    )

    portfolio_context = "\n".join(
        f"  - {t}: {w * 100:.1f}%"
        for t, w in zip(portfolio["tickers"], portfolio["weights"])
    )
    formatted_input = (
        f"Portfolio holdings:\n{portfolio_context}\n\n"
        f"Tickers: {portfolio['tickers']}\n"
        f"Weights: {portfolio['weights']}\n\n"
        f"Question: {question}"
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=formatted_input),
    ]

    called_tool = None
    tool_results = {}

    try:
        response = _llm_with_tools.invoke(messages)
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

    if getattr(response, "tool_calls", None):
        messages.append(response)

        for tc in response.tool_calls:
            name = tc["name"]
            args = tc["args"]

            tool_fn = _tool_map.get(name)
            if tool_fn is None:
                # Unknown tool — stub a ToolMessage so the message sequence stays valid
                messages.append(ToolMessage(
                    content=f"Tool '{name}' is not available.",
                    tool_call_id=tc["id"],
                ))
                continue
            called_tool = name
            raw = tool_fn.invoke(args)
            if isinstance(raw, dict):
                tool_results = raw
            # Strip visualization-only arrays before sending to LLM — saves 2–5k tokens
            if isinstance(raw, dict) and name in _VIZ_ONLY_KEYS:
                llm_raw = {k: v for k, v in raw.items() if k not in _VIZ_ONLY_KEYS[name]}
            else:
                llm_raw = raw
            messages.append(ToolMessage(
                content=json.dumps(llm_raw) if isinstance(llm_raw, dict) else str(llm_raw),
                tool_call_id=tc["id"],
            ))

        try:
            final = _llm_with_tools.invoke(messages)
            response_text = final.content or _FALLBACK_RESPONSE
        except Exception as e:
            raise RuntimeError(f"LLM follow-up call failed: {e}")
    else:
        response_text = response.content or _FALLBACK_RESPONSE

    # Handle portfolio simulation — build a fresh pie chart and return the new portfolio
    portfolio_update = None
    if called_tool == "simulate_portfolio_change_tool" and tool_results:
        new_tickers = tool_results["tickers"]
        new_weights  = tool_results["weights"]
        try:
            comp = get_portfolio_composition(new_tickers, new_weights)
            visualization = _build_visualization("portfolio_pie", comp, portfolio)
        except Exception:
            visualization = {"type": None, "data": {}}
        portfolio_update = {
            "tickers":         new_tickers,
            "weights":         new_weights,
            "period":          portfolio["period"],
            "changes_summary": tool_results.get("changes_summary", ""),
        }
    else:
        viz_type = _TOOL_VIZ_MAP.get(called_tool, "")
        visualization = _build_visualization(viz_type, tool_results, portfolio)

    # Build accumulated analysis: merge prior data with this call's result
    all_analysis = dict(accumulated_analysis or {})
    analysis_updates: dict = {}
    if called_tool and called_tool != "simulate_portfolio_change_tool" and tool_results:
        key = _get_analysis_key(called_tool, tool_results)
        if key:
            analysis_updates[key] = tool_results
            all_analysis[key] = tool_results

    suggestions = generate_smart_suggestions(
        portfolio_composition=all_analysis.get("portfolio_composition"),
        sector_breakdown=all_analysis.get("sector_breakdown"),
        sharpe_data=all_analysis.get("sharpe_data"),
        returns_data=all_analysis.get("returns_data"),
        drawdown_data=all_analysis.get("drawdown_data"),
        var_data=all_analysis.get("var_data"),
    )

    return {
        "response":         response_text,
        "visualization":    visualization,
        "suggestions":      suggestions,
        "analysis_updates": analysis_updates,
        "portfolio_update": portfolio_update,
    }
