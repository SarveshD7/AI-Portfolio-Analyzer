import math

import numpy as np
import pandas as pd
import yfinance as yf


VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "3y", "5y"}


def _portfolio_metrics(tickers: list, weights: list, period: str, start_date=None, end_date=None) -> dict:
    dl_kwargs = {"start": start_date, "end": end_date} if (start_date and end_date) else {"period": period}
    try:
        raw = yf.download(tickers, auto_adjust=True, progress=False, **dl_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data: {e}")

    if raw.empty:
        raise ValueError("No data returned. Check tickers and network connection.")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]] if "Close" in raw.columns else raw

    close = close.dropna(axis=1, how="all")
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise ValueError(f"No data for ticker(s): {missing}")

    ticker_weight = dict(zip(tickers, weights))
    aligned_weights = [ticker_weight[t] for t in close.columns]
    daily = close.pct_change().dropna().dot(aligned_weights)

    total_return = float((1 + daily).prod() - 1) * 100
    n_days = len(daily)
    annualized_return = float(((1 + total_return / 100) ** (252 / n_days)) - 1) * 100
    annualized_vol = float(daily.std()) * math.sqrt(252) * 100
    sharpe = (annualized_return / 100 - 0.06) / (annualized_vol / 100) if annualized_vol > 0 else 0.0

    cum = (1 + daily).cumprod()
    max_drawdown = float(((cum - cum.cummax()) / cum.cummax()).min()) * 100

    return {
        "total_return_pct":        round(total_return, 2),
        "annualized_return_pct":   round(annualized_return, 2),
        "annualized_volatility_pct": round(annualized_vol, 2),
        "sharpe_ratio":            round(sharpe, 2),
        "max_drawdown_pct":        round(max_drawdown, 2),
        "num_holdings":            len(tickers),
    }


def compare_portfolios(
    original_tickers: list,
    original_weights: list,
    modified_tickers: list,
    modified_weights: list,
    period: str = "1y",
    start_date: str = None,
    end_date: str = None,
) -> dict:
    period = period or "1y"
    if not (start_date and end_date) and period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Must be one of {sorted(VALID_PERIODS)}.")
    period_label = f"{start_date} → {end_date}" if (start_date and end_date) else period

    orig = _portfolio_metrics(original_tickers, original_weights, period, start_date, end_date)
    mod  = _portfolio_metrics(modified_tickers, modified_weights, period, start_date, end_date)

    orig_map = dict(zip(original_tickers, original_weights))
    mod_map  = dict(zip(modified_tickers,  modified_weights))
    orig_set = set(original_tickers)
    mod_set  = set(modified_tickers)

    weight_changes = []
    for t in sorted(orig_set & mod_set):
        delta = round((mod_map[t] - orig_map[t]) * 100, 1)
        if abs(delta) > 0.1:
            weight_changes.append({
                "ticker":               t,
                "original_weight_pct":  round(orig_map[t] * 100, 1),
                "modified_weight_pct":  round(mod_map[t] * 100, 1),
                "delta_pct":            delta,
            })

    metrics = [
        {
            "label":           "Total Return",
            "original":        orig["total_return_pct"],
            "modified":        mod["total_return_pct"],
            "delta":           round(mod["total_return_pct"] - orig["total_return_pct"], 2),
            "unit":            "%",
            "higher_is_better": True,
        },
        {
            "label":           "Ann. Return",
            "original":        orig["annualized_return_pct"],
            "modified":        mod["annualized_return_pct"],
            "delta":           round(mod["annualized_return_pct"] - orig["annualized_return_pct"], 2),
            "unit":            "%",
            "higher_is_better": True,
        },
        {
            "label":           "Sharpe Ratio",
            "original":        orig["sharpe_ratio"],
            "modified":        mod["sharpe_ratio"],
            "delta":           round(mod["sharpe_ratio"] - orig["sharpe_ratio"], 2),
            "unit":            "",
            "higher_is_better": True,
        },
        {
            "label":           "Volatility",
            "original":        orig["annualized_volatility_pct"],
            "modified":        mod["annualized_volatility_pct"],
            "delta":           round(mod["annualized_volatility_pct"] - orig["annualized_volatility_pct"], 2),
            "unit":            "%",
            "higher_is_better": False,
        },
        {
            "label":           "Max Drawdown",
            "original":        orig["max_drawdown_pct"],
            "modified":        mod["max_drawdown_pct"],
            "delta":           round(mod["max_drawdown_pct"] - orig["max_drawdown_pct"], 2),
            "unit":            "%",
            "higher_is_better": True,  # values are negative; -5% > -20% means smaller drawdown = better
        },
    ]

    return {
        "original":         orig,
        "modified":         mod,
        "metrics":          metrics,
        "holdings_changes": {
            "added":          sorted(mod_set - orig_set),
            "removed":        sorted(orig_set - mod_set),
            "weight_changes": weight_changes,
        },
        "period":           period_label,
    }
