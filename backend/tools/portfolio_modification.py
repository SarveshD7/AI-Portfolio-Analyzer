import yfinance as yf


def _validate_ticker(ticker: str) -> bool:
    try:
        info = yf.Ticker(ticker).fast_info
        return getattr(info, "last_price", None) is not None
    except Exception:
        return True


def _match_ticker(target: str, portfolio: dict) -> str | None:
    """Return the portfolio key that best matches the target string."""
    t = target.strip()
    if t in portfolio:
        return t
    tup = t.upper()
    for key in portfolio:
        if key.upper() == tup:
            return key
    t_base = tup.split(".")[0]
    for key in portfolio:
        if key.upper().split(".")[0] == t_base:
            return key
    return None


def modify_portfolio(
    tickers: list,
    weights: list,
    modifications: dict[str, float],
) -> dict:
    """Apply modifications to a portfolio and return the rebalanced result.

    modifications: {ticker: weight}
        weight == 0.0  → remove ticker
        weight > 0, ticker in portfolio → update weight
        weight > 0, ticker not in portfolio → validate via yfinance, then add
    """
    portfolio = {t: float(w) for t, w in zip(tickers, weights)}
    changes: list[str] = []
    invalid_tickers: list[str] = []
    pinned: dict[str, float] = {}  # tickers with explicit target weights

    for ticker, weight in (modifications or {}).items():
        matched = _match_ticker(ticker, portfolio)

        if weight == 0.0:
            if matched:
                old_pct = portfolio[matched] * 100
                changes.append(f"Removed {matched} (was {old_pct:.1f}%)")
                del portfolio[matched]
        elif matched:
            old_pct = portfolio[matched] * 100
            pinned[matched] = float(weight)
            changes.append(f"Changed {matched}: {old_pct:.1f}% → {float(weight) * 100:.1f}%")
        else:
            if not _validate_ticker(ticker):
                invalid_tickers.append(ticker)
                changes.append(f"{ticker} not found on yfinance — skipped")
                continue
            pinned[ticker] = float(weight)
            changes.append(f"Added {ticker} ({float(weight) * 100:.1f}%)")

    if not portfolio and not pinned:
        raise ValueError("Portfolio is empty after modifications — cannot proceed.")

    # Tickers not explicitly modified keep their original weights and scale to fill
    # whatever space the pinned tickers don't occupy.
    free = {t: w for t, w in portfolio.items() if t not in pinned}
    pinned_sum = sum(pinned.values())

    if pinned_sum < 1.0:
        free_sum = sum(free.values())
        remaining = 1.0 - pinned_sum
        scaled_free = (
            {t: (w / free_sum) * remaining for t, w in free.items()} if free_sum > 0 else {}
        )
        final = {**scaled_free, **pinned}
    else:
        # Pinned weights already fill ≥100% — include free tickers and normalize together
        final = {**free, **pinned}

    if not final:
        raise ValueError("Portfolio is empty after modifications — cannot proceed.")

    total = sum(final.values())
    normalized = {t: w / total for t, w in final.items()}

    new_tickers = list(normalized.keys())
    new_weights = [round(w, 6) for w in normalized.values()]

    summary = ", ".join(changes) if changes else "No changes applied"
    if changes and abs(total - 1.0) > 0.01:
        summary += " (weights rebalanced to 100%)"

    return {
        "tickers":         new_tickers,
        "weights":         new_weights,
        "changes_summary": summary,
        "invalid_tickers": invalid_tickers,
    }
