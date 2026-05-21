import yfinance as yf
import pandas as pd


VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "3y", "5y"}


def calculate_portfolio_returns(tickers: list, weights: list, period: str = "1y") -> dict:
    if period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Must be one of {sorted(VALID_PERIODS)}.")

    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data from yfinance: {e}")

    if raw.empty:
        raise ValueError("No data returned. Check tickers and network connection.")

    # yfinance returns MultiIndex columns when multiple tickers are requested
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]] if "Close" in raw.columns else raw

    # Drop tickers that came back empty
    close = close.dropna(axis=1, how="all")
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise ValueError(f"No data found for ticker(s): {missing}")

    # Align weights to the order of surviving columns
    ticker_weight = dict(zip(tickers, weights))
    aligned_weights = [ticker_weight[t] for t in close.columns]

    daily_pct = close.pct_change().dropna()

    # Weighted daily portfolio return
    portfolio_daily = daily_pct.dot(aligned_weights)

    # Total return: compound the daily returns
    total_return = ((1 + portfolio_daily).prod() - 1) * 100

    daily_returns = [
        {"date": str(date.date()), "return": round(float(val) * 100, 4)}
        for date, val in portfolio_daily.items()
    ]

    # Per-stock total returns and contributions
    stock_total_pct = ((1 + daily_pct).prod() - 1) * 100

    stock_contributions = []
    for ticker in close.columns:
        weight = ticker_weight[ticker]
        ret = round(float(stock_total_pct[ticker]), 2)
        contribution = round(weight * ret, 2)
        stock_contributions.append({
            "ticker": ticker,
            "weight_pct": round(weight * 100, 1),
            "return_pct": ret,
            "contribution_pct": contribution,
        })

    # Sort worst contribution first (most negative at top)
    stock_contributions.sort(key=lambda x: x["contribution_pct"])

    best = max(stock_contributions, key=lambda x: x["return_pct"])
    worst = min(stock_contributions, key=lambda x: x["return_pct"])

    return {
        "total_return_pct": round(float(total_return), 2),
        "daily_returns": daily_returns,
        "period": period,
        "stock_contributions": stock_contributions,
        "best_performer": {"ticker": best["ticker"], "return_pct": best["return_pct"]},
        "worst_performer": {"ticker": worst["ticker"], "return_pct": worst["return_pct"]},
    }
