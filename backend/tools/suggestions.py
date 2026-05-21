_PERIOD_LABELS = {
    "1mo": "1 month", "3mo": "3 months", "6mo": "6 months",
    "1y": "1 year",   "3y": "3 years",   "5y": "5 years",
}

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}

# Fallback starters used to pad to 3 when fewer patterns fire
_FALLBACKS = [
    {
        "text": "How did my portfolio perform?",
        "query": "How has my portfolio performed over the last year?",
        "action": "_fallback_returns",
        "priority": "low",
    },
    {
        "text": "Compare vs Nifty 50",
        "query": "How does my portfolio compare against Nifty 50?",
        "action": "_fallback_benchmark",
        "priority": "low",
    },
    {
        "text": "Show my risk metrics",
        "query": "What is the risk profile of my portfolio?",
        "action": "_fallback_risk",
        "priority": "low",
    },
]


def generate_smart_suggestions(
    portfolio_composition: dict = None,
    sector_breakdown: dict = None,
    sharpe_data: dict = None,
    returns_data: dict = None,
    drawdown_data: dict = None,
    var_data: dict = None,
) -> list[dict]:
    """Deterministic pattern-matching over real tool outputs → contextual suggestions."""
    try:
        return _generate(portfolio_composition, sector_breakdown, sharpe_data,
                         returns_data, drawdown_data, var_data)
    except Exception:
        return []


def _generate(
    portfolio_composition, sector_breakdown, sharpe_data,
    returns_data, drawdown_data, var_data,
) -> list[dict]:
    candidates = []

    # Pattern 1: High sector concentration
    if sector_breakdown:
        breakdown = sector_breakdown.get("breakdown", [])
        if breakdown:
            top = breakdown[0]
            pct = top["weight_pct"]
            sector = top["label"]
            if pct > 40:
                priority = "high" if pct > 60 else "medium"
                candidates.append({
                    "text": f"{pct:.0f}% in {sector}",
                    "query": "Show me my sector concentration breakdown",
                    "action": "analyze_sector_risk",
                    "priority": priority,
                    "_score": pct,
                })

    # Pattern 2: Negative returns
    if returns_data:
        total_return = returns_data.get("total_return_pct", 0)
        period_label = _PERIOD_LABELS.get(
            returns_data.get("period", ""), returns_data.get("period", "this period")
        )
        if total_return < 0:
            priority = "high" if total_return < -10 else "medium"
            candidates.append({
                "text": f"Down {abs(total_return):.1f}% over {period_label}",
                "query": "Which holdings are dragging down my portfolio?",
                "action": "show_stock_breakdown",
                "priority": priority,
                "_score": abs(total_return),
            })

    # Pattern 3: Low Sharpe ratio
    if sharpe_data:
        sharpe = sharpe_data.get("sharpe_ratio", 1.0)
        if sharpe < 0.5:
            priority = "high" if sharpe < 0 else "medium"
            candidates.append({
                "text": f"Sharpe ratio {sharpe:.2f} — low risk-adjusted returns",
                "query": "How can I improve my portfolio's risk-adjusted returns?",
                "action": "suggest_diversification",
                "priority": priority,
                "_score": (0.5 - sharpe) * 20,
            })

    # Pattern 4: High maximum drawdown
    if drawdown_data:
        max_dd = drawdown_data.get("max_drawdown_pct", 0)
        if max_dd < -20:
            priority = "high" if max_dd < -30 else "medium"
            candidates.append({
                "text": f"Max drawdown {max_dd:.1f}%",
                "query": "How can I reduce my portfolio's downside risk?",
                "action": "analyze_risk_reduction",
                "priority": priority,
                "_score": abs(max_dd),
            })

    # Pattern 5: High tail risk
    if var_data:
        var_95 = var_data.get("var_95_pct", 0)
        if var_95 < -3.0:
            priority = "high" if var_95 < -5.0 else "medium"
            candidates.append({
                "text": f"VaR {var_95:.1f}% — high tail risk",
                "query": "How can I reduce my portfolio's tail risk exposure?",
                "action": "explore_var_strategies",
                "priority": priority,
                "_score": abs(var_95),
            })

    # Pattern 6 & 7: Holdings count
    if portfolio_composition:
        n = portfolio_composition.get("total_holdings", 10)
        if n < 5:
            priority = "high" if n < 3 else "medium"
            label = "stock" if n == 1 else "stocks"
            candidates.append({
                "text": f"Only {n} {label} — limited diversification",
                "query": "What are the benefits of adding more stocks to my portfolio?",
                "action": "suggest_more_holdings",
                "priority": priority,
                "_score": (5 - n) * 5,
            })
        elif n > 15:
            priority = "medium" if n > 20 else "low"
            candidates.append({
                "text": f"{n} holdings — consider simplifying",
                "query": "How can I simplify my portfolio while staying diversified?",
                "action": "suggest_consolidation",
                "priority": priority,
                "_score": n - 15,
            })

    # Pattern 8: Single-stock concentration (available on initial load)
    if portfolio_composition:
        largest = portfolio_composition.get("largest_holding", {})
        lw = largest.get("weight_pct", 0)
        ln = largest.get("name", "")
        already_added = any(c["action"] == "suggest_more_holdings" for c in candidates)
        if lw > 30 and not already_added:
            priority = "high" if lw > 50 else "medium"
            candidates.append({
                "text": f"{ln} is {lw:.0f}% of portfolio",
                "query": "What are the risks of having one stock dominate my portfolio?",
                "action": "single_stock_concentration",
                "priority": priority,
                "_score": lw,
            })

    candidates.sort(key=lambda c: (_PRIORITY_ORDER[c["priority"]], -c["_score"]))
    for c in candidates:
        del c["_score"]

    # Pad to 3 with fallback starters so there are always exactly 3 suggestions
    existing_actions = {c["action"] for c in candidates}
    for fb in _FALLBACKS:
        if len(candidates) >= 3:
            break
        if fb["action"] not in existing_actions:
            candidates.append(fb)

    return candidates[:3]
