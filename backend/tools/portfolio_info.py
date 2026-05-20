import yfinance as yf


def get_portfolio_composition(tickers: list, weights: list) -> dict:
    """Fetch company names and build a sorted holdings breakdown."""
    holdings = []
    for ticker, weight in zip(tickers, weights):
        try:
            info = yf.Ticker(ticker).info
            name = info.get("longName") or info.get("shortName") or ticker
        except Exception:
            name = ticker
        holdings.append({
            "ticker": ticker,
            "name": name,
            "weight_pct": round(weight * 100, 2),
        })

    holdings.sort(key=lambda h: h["weight_pct"], reverse=True)

    return {
        "holdings": holdings,
        "total_holdings": len(holdings),
        "largest_holding": {
            "name": holdings[0]["name"],
            "weight_pct": holdings[0]["weight_pct"],
        },
        "smallest_holding": {
            "name": holdings[-1]["name"],
            "weight_pct": holdings[-1]["weight_pct"],
        },
    }
