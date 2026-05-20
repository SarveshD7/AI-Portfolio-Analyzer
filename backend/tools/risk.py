import math

import pandas as pd
import yfinance as yf


VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "3y", "5y"}


def calculate_sharpe_ratio(
    tickers: list,
    weights: list,
    period: str = "1y",
    risk_free_rate: float = 0.06,
) -> dict:
    if period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Must be one of {sorted(VALID_PERIODS)}.")

    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data from yfinance: {e}")

    if raw.empty:
        raise ValueError("No data returned. Check tickers and network connection.")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]] if "Close" in raw.columns else raw

    close = close.dropna(axis=1, how="all")
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise ValueError(f"No data found for ticker(s): {missing}")

    ticker_weight = dict(zip(tickers, weights))
    aligned_weights = [ticker_weight[t] for t in close.columns]

    daily_pct = close.pct_change().dropna()

    if len(daily_pct) < 5:
        raise ValueError("Insufficient data for Sharpe ratio calculation (need at least 5 trading days).")

    portfolio_daily = daily_pct.dot(aligned_weights)

    # Annualized return
    total_return = float((1 + portfolio_daily).prod() - 1)
    n_days = len(portfolio_daily)
    annualized_return = float(((1 + total_return) ** (252 / n_days)) - 1)

    # Annualized volatility
    daily_vol = float(portfolio_daily.std())
    annualized_vol = daily_vol * math.sqrt(252)

    sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0.0

    return {
        "sharpe_ratio": round(sharpe, 2),
        "annualized_return_pct": round(annualized_return * 100, 2),
        "annualized_volatility_pct": round(annualized_vol * 100, 2),
        "risk_free_rate_pct": round(risk_free_rate * 100, 2),
        "period": period,
    }
