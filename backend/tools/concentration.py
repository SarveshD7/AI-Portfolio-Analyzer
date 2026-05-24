import yfinance as yf

VALID_BREAKDOWN_TYPES = {"sector", "asset_class", "factor"}


def _classify_asset_class(info: dict) -> str:
    qt = (info.get("quoteType") or "").upper()
    return {
        "EQUITY":         "Equity",
        "ETF":            "ETF",
        "MUTUALFUND":     "Mutual Fund",
        "FUTURE":         "Futures",
        "OPTION":         "Options",
        "INDEX":          "Index",
        "CURRENCY":       "Currency",
        "CRYPTOCURRENCY": "Cryptocurrency",
    }.get(qt, "Other")


def _classify_market_cap_factor(info: dict) -> str:
    mc = info.get("marketCap") or 0
    if mc >= 10_000_000_000:
        return "Large Cap"
    if mc >= 2_000_000_000:
        return "Mid Cap"
    if mc > 0:
        return "Small Cap"
    return "Unknown"


def analyze_concentration(tickers: list, weights: list, breakdown_type: str = "sector") -> dict:
    """Group portfolio weights by sector, asset class, or market-cap factor tier."""
    if breakdown_type not in VALID_BREAKDOWN_TYPES:
        raise ValueError(
            f"Invalid breakdown_type '{breakdown_type}'. Must be one of {sorted(VALID_BREAKDOWN_TYPES)}."
        )

    group_weights: dict[str, float] = {}

    for ticker, weight in zip(tickers, weights):
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            info = {}

        if breakdown_type == "sector":
            label = info.get("sector") or "Unknown"
        elif breakdown_type == "asset_class":
            label = _classify_asset_class(info)
        else:
            label = _classify_market_cap_factor(info)

        group_weights[label] = group_weights.get(label, 0.0) + weight

    breakdown = sorted(
        [{"label": k, "weight_pct": round(v * 100, 2)} for k, v in group_weights.items()],
        key=lambda x: x["weight_pct"],
        reverse=True,
    )

    dominant = breakdown[0] if breakdown else {"label": "N/A", "weight_pct": 0.0}

    return {
        "breakdown": breakdown,
        "breakdown_type": breakdown_type,
        "dominant": dominant,
        "concentration_warning": dominant["weight_pct"] > 30,
    }
