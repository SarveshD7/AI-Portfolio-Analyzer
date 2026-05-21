def _match_ticker(target: str, portfolio: dict) -> str | None:
    """Return the portfolio key that best matches the target string."""
    t = target.strip()
    # Exact match
    if t in portfolio:
        return t
    # Case-insensitive exact
    tup = t.upper()
    for key in portfolio:
        if key.upper() == tup:
            return key
    # Strip exchange suffix (.NS, .BO, etc.) and compare base symbols
    t_base = tup.split(".")[0]
    for key in portfolio:
        if key.upper().split(".")[0] == t_base:
            return key
    return None


def modify_portfolio(
    tickers: list,
    weights: list,
    remove: list = None,
    add: dict = None,
    change_weight: dict = None,
) -> dict:
    """Apply what-if modifications to a portfolio and return the rebalanced result."""
    portfolio = {t: float(w) for t, w in zip(tickers, weights)}
    changes: list[str] = []

    # Removals
    for target in (remove or []):
        matched = _match_ticker(target, portfolio)
        if matched:
            changes.append(f"Removed {matched} ({portfolio[matched] * 100:.1f}%)")
            del portfolio[matched]

    # Weight changes
    for target, new_weight in (change_weight or {}).items():
        matched = _match_ticker(target, portfolio)
        if matched:
            old_w = portfolio[matched]
            portfolio[matched] = float(new_weight)
            changes.append(f"Changed {matched}: {old_w * 100:.1f}% → {float(new_weight) * 100:.1f}%")

    # Additions
    for ticker, weight in (add or {}).items():
        matched = _match_ticker(ticker, portfolio)
        key = matched if matched else ticker
        portfolio[key] = float(weight)
        changes.append(f"Added {key} ({float(weight) * 100:.1f}%)")

    if not portfolio:
        raise ValueError("Portfolio is empty after modifications — cannot proceed.")

    total = sum(portfolio.values())
    normalized = {t: w / total for t, w in portfolio.items()}

    new_tickers = list(normalized.keys())
    new_weights  = [round(w, 6) for w in normalized.values()]

    summary = ", ".join(changes) if changes else "No changes applied"
    if changes and abs(total - 1.0) > 0.01:
        summary += " (weights rebalanced to 100%)"

    return {
        "tickers":          new_tickers,
        "weights":          new_weights,
        "changes_summary":  summary,
    }
