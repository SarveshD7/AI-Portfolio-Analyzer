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

    return {
        "total_return_pct": round(float(total_return), 2),
        "daily_returns": daily_returns,
        "period": period,
    }
