import json
import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from tools.benchmark import benchmark_portfolio
from tools.concentration import analyze_concentration
from tools.correlation import calculate_correlation
from tools.portfolio_info import get_portfolio_composition
from tools.returns import calculate_portfolio_returns
from tools.risk import calculate_sharpe_ratio, calculate_max_drawdown, calculate_var

load_dotenv()

# ---------------------------------------------------------------------------
# Tools  (docstrings drive LLM tool selection — keep them descriptive)
# ---------------------------------------------------------------------------

@tool
def portfolio_returns_tool(tickers: list, weights: list, period: str = "1y") -> dict:
    """Calculate historical portfolio returns over a time period.

    Use when the user asks about:
    - How the portfolio performed / did
    - Returns, gains, losses, or profit
    - Performance over a specific period
    - "How much did I make / lose?"

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
    """Show portfolio composition with company names and allocation weights.

    Use when the user asks about:
    - What stocks are in the portfolio / what they own
    - Portfolio holdings or composition
    - Allocation or sector breakdown

    Args:
        tickers: List of stock tickers (e.g. ["RELIANCE.NS", "TCS.NS"])
        weights: Corresponding weights as decimals summing to 1.0 (e.g. [0.6, 0.4])
    """
    return get_portfolio_composition(tickers, weights)


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

def run_agent(portfolio: dict, question: str) -> dict:
    """Run the tool-calling agent using bind_tools.

    Args:
        portfolio: {"tickers": [...], "weights": [...], "period": "1y"}
        question:  Natural language question about the portfolio

    Returns:
        {"response": str, "visualization": {"type": str, "data": dict}}
    """
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

    try:
        response = _llm_with_tools.invoke(messages)
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

    called_tool = None
    tool_results = {}

    if getattr(response, "tool_calls", None):
        messages.append(response)

        for tc in response.tool_calls:
            name = tc["name"]
            args = tc["args"]
            called_tool = name

            tool_fn = _tool_map.get(name)
            if tool_fn:
                raw = tool_fn.invoke(args)
                if isinstance(raw, dict):
                    tool_results = raw
                messages.append(ToolMessage(
                    content=json.dumps(raw) if isinstance(raw, dict) else str(raw),
                    tool_call_id=tc["id"],
                ))

        try:
            final = _llm_with_tools.invoke(messages)
            response_text = final.content
        except Exception as e:
            raise RuntimeError(f"LLM follow-up call failed: {e}")
    else:
        response_text = response.content

    viz_type = _TOOL_VIZ_MAP.get(called_tool, "")
    visualization = _build_visualization(viz_type, tool_results, portfolio)

    return {
        "response": response_text,
        "visualization": visualization,
    }
