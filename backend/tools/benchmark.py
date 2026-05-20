import numpy as np
import pandas as pd
import yfinance as yf

VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "3y", "5y"}

# Lowercase alias → yfinance ticker
_ALIASES: dict[str, str] = {
    "nifty50":   "^NSEI",
    "nifty 50":  "^NSEI",
    "nifty":     "^NSEI",
    "sensex":    "^BSESN",
    "bse":       "^BSESN",
    "bsesensex": "^BSESN",
    "sp500":     "^GSPC",
    "s&p500":    "^GSPC",
    "s&p 500":   "^GSPC",
    "nasdaq":    "^IXIC",
    "dow":       "^DJI",
    "dowjones":  "^DJI",
    "niftybank": "^NSEBANK",
    "nifty bank":"^NSEBANK",
    "niftymid":  "^NSEMDCP50",
}

_NAMES: dict[str, str] = {
    "^NSEI":     "Nifty 50",
    "^BSESN":    "BSE Sensex",
    "^GSPC":     "S&P 500",
    "^IXIC":     "NASDAQ Composite",
    "^DJI":      "Dow Jones",
    "^NSEBANK":  "Nifty Bank",
    "^NSEMDCP50":"Nifty Midcap 50",
}


def benchmark_portfolio(
    tickers: list,
    weights: list,
    period: str = "1y",
    benchmark: str = "^NSEI",
) -> dict:
    """Compare portfolio returns against a market index."""
    if period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Must be one of {sorted(VALID_PERIODS)}.")

    bm_ticker = _ALIASES.get(benchmark.lower().strip(), benchmark)
    bm_name   = _NAMES.get(bm_ticker, bm_ticker)

    try:
        raw = yf.download(tickers + [bm_ticker], period=period, auto_adjust=True, progress=False)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data from yfinance: {e}")

    if raw.empty:
        raise ValueError("No data returned. Check tickers and network connection.")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else (
        raw[["Close"]] if "Close" in raw.columns else raw
    )
    close = close.dropna(axis=1, how="all")

    if bm_ticker not in close.columns:
        raise ValueError(f"No data found for benchmark '{bm_ticker}' ({bm_name}).")

    valid_tickers = [t for t in tickers if t in close.columns]
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise ValueError(f"No data found for ticker(s): {missing}")

    tw = dict(zip(tickers, weights))
    aligned_weights = [tw[t] for t in valid_tickers]

    port_daily = close[valid_tickers].pct_change().dropna().dot(aligned_weights)
    bm_daily   = close[bm_ticker].pct_change().dropna()

    idx = port_daily.index.intersection(bm_daily.index)
    port_daily = port_daily.loc[idx]
    bm_daily   = bm_daily.loc[idx]

    port_cum = (1 + port_daily).cumprod() * 100
    bm_cum   = (1 + bm_daily).cumprod() * 100

    port_total = float((1 + port_daily).prod() - 1) * 100
    bm_total   = float((1 + bm_daily).prod() - 1) * 100

    cov   = np.cov(port_daily.values, bm_daily.values)
    beta  = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 0.0
    alpha = port_total - bm_total          # simple excess return over period

    return {
        "portfolio_cumulative": [
            {"date": str(d.date()), "value": round(float(v), 4)}
            for d, v in port_cum.items()
        ],
        "benchmark_cumulative": [
            {"date": str(d.date()), "value": round(float(v), 4)}
            for d, v in bm_cum.items()
        ],
        "benchmark_name":             bm_name,
        "benchmark_ticker":           bm_ticker,
        "portfolio_total_return_pct": round(port_total, 2),
        "benchmark_total_return_pct": round(bm_total, 2),
        "alpha_pct":                  round(alpha, 2),
        "beta":                       round(beta, 4),
        "outperformed":               port_total > bm_total,
        "period":                     period,
    }
