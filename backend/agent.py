import json
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from tools.benchmark import benchmark_portfolio
from tools.concentration import analyze_concentration
from tools.correlation import calculate_correlation
from tools.portfolio_comparison import compare_portfolios
from tools.portfolio_info import get_portfolio_composition
from tools.portfolio_modification import modify_portfolio
from tools.returns import calculate_portfolio_returns
from tools.risk import calculate_sharpe_ratio, calculate_max_drawdown, calculate_var, calculate_beta, calculate_rolling_metrics
from tools.suggestions import generate_smart_suggestions

load_dotenv()

# ---------------------------------------------------------------------------
# Tools  (docstrings drive LLM tool selection — keep them descriptive)
# ---------------------------------------------------------------------------

@tool
def portfolio_returns_tool(
    tickers: List[str],
    weights: List[float],
    period: Optional[str] = "1y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Calculate historical portfolio returns, performance, and per-stock contribution breakdown.

    Use when the user asks about:
    - How the portfolio performed / did
    - Returns, gains, losses, or profit
    - Performance over a specific period or date range
    - "How much did I make / lose?"
    - "Which stocks performed well or poorly?"
    - "What is dragging down my portfolio?"
    - "Which holdings are underperforming?"
    - Individual stock contributions to portfolio performance
    - Performance breakdown by holding

    Args:
        tickers: List of stock tickers (e.g. ["RELIANCE.NS", "TCS.NS"])
        weights: Corresponding weights as decimals summing to 1.0 (e.g. [0.6, 0.4])
        period: "1mo", "3mo", "6mo", "1y", "3y", or "5y" — used when no date range given
        start_date: Start date in YYYY-MM-DD format (overrides period when paired with end_date)
        end_date: End date in YYYY-MM-DD format (overrides period when paired with start_date)
    """
    return calculate_portfolio_returns(tickers, weights, period, start_date, end_date)


@tool
def sharpe_ratio_tool(
    tickers: List[str],
    weights: List[float],
    period: Optional[str] = "1y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Calculate Sharpe ratio and risk-adjusted return metrics.

    Use when the user asks about:
    - Sharpe ratio or risk-adjusted returns
    - Volatility or standard deviation
    - Whether returns justify the risk taken

    Args:
        tickers: List of stock tickers (e.g. ["RELIANCE.NS", "TCS.NS"])
        weights: Corresponding weights as decimals summing to 1.0 (e.g. [0.6, 0.4])
        period: "1mo", "3mo", "6mo", "1y", "3y", or "5y" — used when no date range given
        start_date: Start date in YYYY-MM-DD format (overrides period when paired with end_date)
        end_date: End date in YYYY-MM-DD format (overrides period when paired with start_date)
    """
    return calculate_sharpe_ratio(tickers, weights, period, start_date=start_date, end_date=end_date)


@tool
def max_drawdown_tool(
    tickers: List[str],
    weights: List[float],
    period: Optional[str] = "1y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Calculate maximum drawdown — the worst peak-to-trough decline.

    Use when the user asks about:
    - Drawdown or maximum drawdown
    - Worst loss or biggest historical drop
    - How bad things got / peak and trough dates

    Args:
        tickers: List of stock tickers (e.g. ["RELIANCE.NS", "TCS.NS"])
        weights: Corresponding weights as decimals summing to 1.0 (e.g. [0.6, 0.4])
        period: "1mo", "3mo", "6mo", "1y", "3y", or "5y" — used when no date range given
        start_date: Start date in YYYY-MM-DD format (overrides period when paired with end_date)
        end_date: End date in YYYY-MM-DD format (overrides period when paired with start_date)
    """
    return calculate_max_drawdown(tickers, weights, period, start_date=start_date, end_date=end_date)


@tool
def var_tool(
    tickers: List[str],
    weights: List[float],
    period: Optional[str] = "1y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Calculate Value at Risk (VaR) and tail risk metrics.

    Use when the user asks about:
    - Value at Risk (VaR)
    - How much they could lose on a bad day
    - Tail risk, downside risk, or expected shortfall (CVaR)
    - Worst-case daily loss scenarios

    Args:
        tickers: List of stock tickers (e.g. ["RELIANCE.NS", "TCS.NS"])
        weights: Corresponding weights as decimals summing to 1.0 (e.g. [0.6, 0.4])
        period: "1mo", "3mo", "6mo", "1y", "3y", or "5y" — used when no date range given
        start_date: Start date in YYYY-MM-DD format (overrides period when paired with end_date)
        end_date: End date in YYYY-MM-DD format (overrides period when paired with start_date)
    """
    return calculate_var(tickers, weights, period, start_date=start_date, end_date=end_date)


@tool
def beta_tool(
    tickers: List[str],
    weights: List[float],
    period: Optional[str] = "1y",
    benchmark: str = "^NSEI",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Calculate portfolio beta, alpha, and market sensitivity relative to a benchmark.

    Use when the user asks about:
    - Beta or market sensitivity / systematic risk
    - How much the portfolio moves with the market
    - Alpha — excess return over the benchmark after adjusting for risk
    - "Is my portfolio aggressive or defensive?"
    - "How correlated is my portfolio to the Nifty / S&P 500?"
    - R-squared or how much of returns are explained by the market

    Beta interpretation:
        beta > 1  → more volatile than the market (aggressive)
        beta = 1  → moves in line with the market
        beta < 1  → less volatile than the market (defensive)
        beta < 0  → moves opposite to the market (hedge)

    Args:
        tickers: List of stock tickers (e.g. ["RELIANCE.NS", "TCS.NS"])
        weights: Corresponding weights as decimals summing to 1.0 (e.g. [0.6, 0.4])
        period: "1mo", "3mo", "6mo", "1y", "3y", or "5y" — used when no date range given
        benchmark: Index ticker or alias. Common values:
            "^NSEI"    / "nifty50"   → Nifty 50  (default)
            "^BSESN"   / "sensex"    → BSE Sensex
            "^GSPC"    / "sp500"     → S&P 500
            "^IXIC"    / "nasdaq"    → NASDAQ Composite
            "^DJI"     / "dow"       → Dow Jones
        start_date: Start date in YYYY-MM-DD format (overrides period when paired with end_date)
        end_date: End date in YYYY-MM-DD format (overrides period when paired with start_date)
    """
    return calculate_beta(tickers, weights, period, benchmark, start_date=start_date, end_date=end_date)


@tool
def rolling_metrics_tool(
    tickers: List[str],
    weights: List[float],
    period: Optional[str] = "1y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Compute rolling risk metrics over time to show how portfolio risk has evolved.

    Use when the user asks about:
    - "Has my portfolio gotten riskier / safer over time?"
    - Rolling risk, rolling volatility, rolling Sharpe ratio
    - "How has risk changed over [period]?"
    - "Is my portfolio more volatile now than before?"
    - "Show me the risk trend over the past year"
    - "Has volatility increased recently?"

    Args:
        tickers: List of stock tickers (e.g. ["RELIANCE.NS", "TCS.NS"])
        weights: Corresponding weights as decimals summing to 1.0 (e.g. [0.6, 0.4])
        period: "1mo", "3mo", "6mo", "1y", "3y", or "5y" — used when no date range given
        start_date: Start date in YYYY-MM-DD format (overrides period when paired with end_date)
        end_date: End date in YYYY-MM-DD format (overrides period when paired with start_date)
    """
    return calculate_rolling_metrics(tickers, weights, period, start_date, end_date)


@tool
def benchmark_tool(
    tickers: List[str],
    weights: List[float],
    period: Optional[str] = "1y",
    benchmark: str = "^NSEI",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
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
        period: "1mo", "3mo", "6mo", "1y", "3y", or "5y" — used when no date range given
        benchmark: Index ticker or alias. Common values:
            "^NSEI"    / "nifty50"   → Nifty 50  (default)
            "^BSESN"   / "sensex"    → BSE Sensex
            "^GSPC"    / "sp500"     → S&P 500
            "^IXIC"    / "nasdaq"    → NASDAQ Composite
            "^DJI"     / "dow"       → Dow Jones
            "^NSEBANK" / "niftybank" → Nifty Bank
        start_date: Start date in YYYY-MM-DD format (overrides period when paired with end_date)
        end_date: End date in YYYY-MM-DD format (overrides period when paired with start_date)
    """
    return benchmark_portfolio(tickers, weights, period, benchmark, start_date, end_date)


@tool
def correlation_tool(
    tickers: List[str],
    weights: List[float],
    period: Optional[str] = "1y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
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
        period: "1mo", "3mo", "6mo", "1y", "3y", or "5y" — used when no date range given
        start_date: Start date in YYYY-MM-DD format (overrides period when paired with end_date)
        end_date: End date in YYYY-MM-DD format (overrides period when paired with start_date)
    """
    return calculate_correlation(tickers, weights, period, start_date, end_date)


@tool
def concentration_tool(tickers: List[str], weights: List[float], breakdown_type: str = "sector") -> dict:
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
def portfolio_composition_tool(tickers: List[str], weights: List[float]) -> dict:
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
    tickers: List[str],
    weights: List[float],
    modifications: Dict[str, float],
) -> dict:
    """Simulate what-if portfolio changes and return the rebalanced portfolio.

    Use when the user asks "what if" questions or wants to modify the portfolio:
    - "What if I exit / remove / sell [stock]?"
    - "What if I didn't have [stock] or [sector]?"
    - "What if I added / bought [stock]?"
    - "Replace X with Y"
    - "Increase / decrease [stock] to X%"
    - "Remove all my energy / tech / banking stocks"
    - Compound: "Remove TCS, add INFY at 15%, and increase HDFC to 30%"

    Always encode ALL changes in a single call using the modifications dict.
    Never make multiple separate tool calls for one compound request.

    After calling this tool, tell the user what changed and what the new portfolio
    looks like. All subsequent analysis will automatically use the modified portfolio.

    Args:
        tickers: Current portfolio tickers exactly as shown in the portfolio context
        weights: Current portfolio weights (decimals summing to 1.0)
        modifications: Dict of {ticker: weight} encoding ALL changes at once.
            weight = 0.0  → remove ticker completely
            weight > 0    → update weight if already held, add if new stock
            Example: {"RELIANCE.NS": 0.0, "TCS.NS": 0.35, "INFY.NS": 0.15}
            MUST use correct yfinance ticker with exchange suffix:
            Indian NSE (.NS): "INFY.NS", "HDFCBANK.NS"  |  US (no suffix): "AAPL", "NVDA"
    """
    return modify_portfolio(tickers, weights, modifications)


@tool
def revert_to_original_portfolio_tool(confirm: bool = True) -> dict:
    """Revert the portfolio back to its original state, undoing all what-if modifications.

    Use when the user asks to:
    - Revert / undo / reset the portfolio
    - Go back to the original / previous portfolio
    - Restore the original holdings
    - Undo all changes
    - "Start fresh" or "cancel modifications"
    - "Use my original portfolio"
    - "Let's go back" / "take me back to original"

    Args:
        confirm: Always pass True.
    """
    return {"status": "reverted"}


@tool
def compare_portfolios_tool(
    original_tickers: List[str],
    original_weights: List[float],
    modified_tickers: List[str],
    modified_weights: List[float],
    period: Optional[str] = "1y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Compare the original portfolio against the current (modified) portfolio across key metrics.

    Use when the user asks about:
    - How the modified portfolio compares to the original
    - "Is my new portfolio better than the old one?"
    - "What changed after the modification?"
    - "Compare original vs current portfolio"
    - "Did removing / adding [stock] improve my portfolio?"
    - "Show me the impact of my changes"
    - "How does the what-if portfolio stack up against what I had before?"

    Args:
        original_tickers: Tickers of the ORIGINAL (unmodified) portfolio
        original_weights: Weights of the ORIGINAL portfolio (decimals summing to 1.0)
        modified_tickers: Tickers of the CURRENT (modified) portfolio
        modified_weights: Weights of the CURRENT portfolio (decimals summing to 1.0)
        period: "1mo", "3mo", "6mo", "1y", "3y", or "5y" — used when no date range given
        start_date: Start date in YYYY-MM-DD format (overrides period when paired with end_date)
        end_date: End date in YYYY-MM-DD format (overrides period when paired with start_date)
    """
    return compare_portfolios(
        original_tickers, original_weights,
        modified_tickers, modified_weights,
        period, start_date, end_date,
    )


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

_tools = [
    portfolio_returns_tool,
    sharpe_ratio_tool,
    max_drawdown_tool,
    var_tool,
    beta_tool,
    rolling_metrics_tool,
    portfolio_composition_tool,
    concentration_tool,
    correlation_tool,
    benchmark_tool,
    simulate_portfolio_change_tool,
    revert_to_original_portfolio_tool,
    compare_portfolios_tool,
]

_tool_map = {t.name: t for t in _tools}

_llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
_llm_with_tools = _llm.bind_tools(_tools)

_SYSTEM_PROMPT = """You are a portfolio analysis assistant.

Use the available tools to answer questions about the user's portfolio.
The portfolio's tickers and weights are in the human message — always pass
those exact values when calling tools.

Period mapping — use period= when the user says a relative window (default "1y"):
  "1 month" / "30 days"                          → period="1mo"
  "3 months" / "quarter"                          → period="3mo"
  "6 months" / "half year"                        → period="6mo"
  "1 year" / "last year" / "past year" / "yearly" → period="1y"
  "3 years"                                        → period="3y"
  "5 years"                                        → period="5y"

Date range mapping — use start_date= and end_date= (YYYY-MM-DD) for anything specific.
Today is 2026-05-24. Always prefer dates over period when the user names a window.
  "YTD" / "year to date" / "this year"           → start="2026-01-01"  end="2026-05-24"
  "Q1" / "first quarter" (current year)          → start="2026-01-01"  end="2026-03-31"
  "Q2" / "second quarter" (current year)         → start="2026-04-01"  end="2026-06-30"
  "Q3" / "third quarter" (current year)          → start="2026-07-01"  end="2026-09-30"
  "Q4" / "fourth quarter" (current year)         → start="2026-10-01"  end="2026-12-31"
  "Jan to June" / "Jan–June"                     → start="<year>-01-01" end="<year>-06-30"
  "from March to August"                         → start="<year>-03-01" end="<year>-08-31"
  "in 2023" / "full year 2023"                   → start="2023-01-01"  end="2023-12-31"
  "last quarter" (before Q2 2026)                → start="2026-01-01"  end="2026-03-31"
  "last month" (before May 2026)                 → start="2026-04-01"  end="2026-04-30"
  "COVID crash"                                   → start="2020-02-19"  end="2020-03-23"
  "COVID recovery"                               → start="2020-03-23"  end="2020-12-31"
  "2008 crisis" / "GFC" / "financial crisis"     → start="2008-09-01"  end="2009-03-31"
  "dot-com crash"                                → start="2000-03-01"  end="2002-10-09"
  If the user names a month without a year, assume the most recent occurrence of that month.

Ticker format rules — critical when adding new stocks:
- Look at the existing portfolio tickers to determine the exchange:
    Tickers ending in ".NS"  → Indian NSE. New additions MUST also use ".NS"
      e.g. "Infosys" → "INFY.NS", "HDFC Bank" → "HDFCBANK.NS", "Gold ETF" → "GOLDBEES.NS"
    Tickers ending in ".BO"  → Indian BSE. New additions MUST also use ".BO"
    No suffix                → US market. New additions have no suffix
      e.g. "Apple" → "AAPL", "Microsoft" → "MSFT", "Nvidia" → "NVDA"
- NEVER pass a plain company name ("Infosys", "Apple") as a ticker — always use the
  official yfinance symbol. If unsure, use the most widely recognised symbol for that exchange.
- CRITICAL — explicit ticker symbols vs company names:
    If the user provides something that already looks like a ticker (all-caps word, e.g. TMPV,
    INFY, HDFCBANK), only append the exchange suffix if it is missing. NEVER substitute it
    with a different company's ticker. TMPV → TMPV.NS (Indian portfolio), NOT TATAMOTORS.NS.
    Only look up and substitute a ticker when the user provides a plain company name in
    natural language (e.g. "Tata Motors", "Infosys", "Apple").

Revert intent — call revert_to_original_portfolio_tool whenever the user says anything like:
  "go back to original", "revert", "undo", "reset", "restore original", "original portfolio",
  "cancel changes", "start fresh", "take me back". Always pass confirm=True.

After calling a tool, respond conversationally in 2–3 sentences:
- Explain the results in plain English
- Embed the specific numbers from the tool result
- Provide brief context or interpretation
- For simulate_portfolio_change_tool: base your response entirely on the `changes_summary`
  field. Do not add doubts or caveats about ticker validity — the tool already validated each
  ticker. If `invalid_tickers` is empty, report all changes as successful without qualification."""

# Maps called tool name → canvas visualization type
_TOOL_VIZ_MAP = {
    "portfolio_returns_tool":     "line_chart",
    "sharpe_ratio_tool":          "metrics",
    "var_tool":                   "metrics",
    "max_drawdown_tool":          "drawdown_chart",
    "beta_tool":                  "metrics",
    "rolling_metrics_tool":       "rolling_metrics_chart",
    "portfolio_composition_tool": "portfolio_pie",
    "concentration_tool":         "concentration_pie",
    "correlation_tool":           "correlation_heatmap",
    "benchmark_tool":             "benchmark_chart",
    "compare_portfolios_tool":    "comparison_chart",
}

# Large array fields that are only needed for visualization, not for the LLM's text response.
# Stripping these from ToolMessages cuts token usage by ~2–5k tokens per call.
_VIZ_ONLY_KEYS = {
    "portfolio_returns_tool": ["daily_returns"],
    "max_drawdown_tool":      ["drawdown_series"],
    "benchmark_tool":         ["portfolio_cumulative", "benchmark_cumulative"],
    "rolling_metrics_tool":   ["rolling_volatility", "rolling_sharpe", "rolling_var_95"],
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
    if viz_type == "comparison_chart" and tool_results:
        return {"type": "comparison_chart", "data": tool_results}
    if viz_type == "rolling_metrics_chart" and tool_results:
        return {"type": "rolling_metrics_chart", "data": tool_results}
    return {"type": None, "data": {}}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_agent(
    portfolio: dict,
    question: str,
    accumulated_analysis: dict = None,
    is_initial: bool = False,
    original_portfolio: dict = None,
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
    orig = original_portfolio
    if orig and (orig["tickers"] != portfolio["tickers"] or orig["weights"] != portfolio["weights"]):
        orig_context = "\n".join(
            f"  - {t}: {w * 100:.1f}%"
            for t, w in zip(orig["tickers"], orig["weights"])
        )
        original_section = (
            f"\nOriginal portfolio (before modifications):\n{orig_context}\n"
            f"Original Tickers: {orig['tickers']}\n"
            f"Original Weights: {orig['weights']}\n"
        )
    else:
        original_section = ""
    formatted_input = (
        f"Current portfolio holdings:\n{portfolio_context}\n\n"
        f"Tickers: {portfolio['tickers']}\n"
        f"Weights: {portfolio['weights']}\n"
        f"{original_section}\n"
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
            final = _llm.invoke(messages)
            response_text = final.content or _FALLBACK_RESPONSE
        except Exception as e:
            raise RuntimeError(f"LLM follow-up call failed: {e}")
    else:
        response_text = response.content or _FALLBACK_RESPONSE

    # Handle portfolio simulation — build a fresh pie chart and return the new portfolio
    portfolio_update = None
    composition_for_suggestions = None  # set below when portfolio changes
    if called_tool == "simulate_portfolio_change_tool" and tool_results:
        new_tickers = tool_results["tickers"]
        new_weights  = tool_results["weights"]
        try:
            comp = get_portfolio_composition(new_tickers, new_weights)
            visualization = _build_visualization("portfolio_pie", comp, portfolio)
            composition_for_suggestions = comp
        except Exception:
            visualization = {"type": None, "data": {}}
        portfolio_update = {
            "tickers":         new_tickers,
            "weights":         new_weights,
            "period":          portfolio["period"],
            "changes_summary": tool_results.get("changes_summary", ""),
        }
    elif called_tool == "revert_to_original_portfolio_tool" and original_portfolio:
        orig_tickers = original_portfolio["tickers"]
        orig_weights = original_portfolio["weights"]
        try:
            comp = get_portfolio_composition(orig_tickers, orig_weights)
            visualization = _build_visualization("portfolio_pie", comp, portfolio)
            composition_for_suggestions = comp
        except Exception:
            visualization = {"type": None, "data": {}}
        portfolio_update = {
            "tickers":         orig_tickers,
            "weights":         orig_weights,
            "period":          portfolio["period"],
            "changes_summary": "Reverted to original portfolio",
            "is_revert":       True,
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

    # Use the freshly computed composition when the portfolio changed; fall back to accumulated.
    suggestions = generate_smart_suggestions(
        portfolio_composition=composition_for_suggestions or all_analysis.get("portfolio_composition"),
        sector_breakdown=all_analysis.get("sector_breakdown"),
        last_question=question,
        last_response=response_text,
    )

    return {
        "response":         response_text,
        "visualization":    visualization,
        "suggestions":      suggestions,
        "analysis_updates": analysis_updates,
        "portfolio_update": portfolio_update,
    }
