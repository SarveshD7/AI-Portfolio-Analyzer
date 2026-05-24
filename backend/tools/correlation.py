import pandas as pd
import yfinance as yf

VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "3y", "5y"}


def calculate_correlation(
    tickers: list,
    weights: list,
    period: str = "1y",
    start_date: str = None,
    end_date: str = None,
) -> dict:
    """Compute pairwise Pearson correlation of daily returns across all portfolio assets."""
    period = period or "1y"
    dl_kwargs = {"start": start_date, "end": end_date} if (start_date and end_date) else {"period": period}
    if not (start_date and end_date) and period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Must be one of {sorted(VALID_PERIODS)}.")
    period_label = f"{start_date} → {end_date}" if (start_date and end_date) else period

    if len(tickers) < 2:
        raise ValueError("Need at least 2 tickers to compute a correlation matrix.")

    try:
        raw = yf.download(tickers, auto_adjust=True, progress=False, **dl_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data from yfinance: {e}")

    if raw.empty:
        raise ValueError("No data returned. Check tickers and network connection.")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else (
        raw[["Close"]] if "Close" in raw.columns else raw
    )
    close = close.dropna(axis=1, how="all")

    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise ValueError(f"No data found for ticker(s): {missing}")

    close = close[tickers]  # preserve requested order
    corr = close.pct_change().dropna().corr()

    matrix = [
        [round(float(corr.loc[t1, t2]), 4) for t2 in tickers]
        for t1 in tickers
    ]

    high_corr, low_corr = [], []
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if j <= i:
                continue
            val = float(corr.loc[t1, t2])
            entry = {"assets": [t1, t2], "correlation": round(val, 4)}
            if val > 0.70:
                high_corr.append(entry)
            elif val < 0.30:
                low_corr.append(entry)

    high_corr.sort(key=lambda x: -x["correlation"])
    low_corr.sort(key=lambda x: x["correlation"])

    return {
        "tickers": tickers,
        "matrix": matrix,
        "period": period_label,
        "n_assets": len(tickers),
        "high_correlations": high_corr,
        "low_correlations": low_corr,
    }
