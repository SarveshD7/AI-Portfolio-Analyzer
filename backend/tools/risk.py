import math

import numpy as np
import pandas as pd
import yfinance as yf


VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "3y", "5y"}


def _download_portfolio_daily(tickers, weights, period):
    """Shared helper: returns aligned (portfolio_daily, aligned_weights, close)."""
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
    portfolio_daily = close.pct_change().dropna().dot(aligned_weights)
    return portfolio_daily, aligned_weights


def calculate_sharpe_ratio(
    tickers: list,
    weights: list,
    period: str = "1y",
    risk_free_rate: float = 0.06,
) -> dict:
    if period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Must be one of {sorted(VALID_PERIODS)}.")

    portfolio_daily, _ = _download_portfolio_daily(tickers, weights, period)

    if len(portfolio_daily) < 5:
        raise ValueError("Insufficient data (need at least 5 trading days).")

    total_return = float((1 + portfolio_daily).prod() - 1)
    n_days = len(portfolio_daily)
    annualized_return = float(((1 + total_return) ** (252 / n_days)) - 1)
    annualized_vol = float(portfolio_daily.std()) * math.sqrt(252)
    sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0.0

    return {
        "sharpe_ratio": round(sharpe, 2),
        "annualized_return_pct": round(annualized_return * 100, 2),
        "annualized_volatility_pct": round(annualized_vol * 100, 2),
        "risk_free_rate_pct": round(risk_free_rate * 100, 2),
        "period": period,
    }


def calculate_max_drawdown(tickers: list, weights: list, period: str = "1y") -> dict:
    """Calculate maximum drawdown — the largest peak-to-trough decline — for the portfolio."""
    if period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Must be one of {sorted(VALID_PERIODS)}.")

    portfolio_daily, _ = _download_portfolio_daily(tickers, weights, period)

    if len(portfolio_daily) < 5:
        raise ValueError("Insufficient data (need at least 5 trading days).")

    cum = (1 + portfolio_daily).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max * 100

    max_drawdown_val = float(drawdown.min())
    trough_idx = drawdown.idxmin()

    # Peak date: last time a new high was set before the trough
    is_new_high = running_max.diff().fillna(1) > 1e-10
    pre_trough_highs = is_new_high[:trough_idx][is_new_high[:trough_idx]]
    peak_idx = pre_trough_highs.index[-1] if not pre_trough_highs.empty else cum.index[0]

    # Recovery date: first day after trough where portfolio climbs back to the peak value
    peak_value = float(running_max[trough_idx])
    post_trough = cum[trough_idx:].iloc[1:]
    recovered = post_trough[post_trough >= peak_value * (1 - 1e-6)]
    if not recovered.empty:
        recovery_date = str(recovered.index[0].date())
        currently_in_drawdown = False
    else:
        recovery_date = None
        currently_in_drawdown = float(drawdown.iloc[-1]) < -0.01

    drawdown_series = [
        {"date": str(idx.date()), "drawdown": round(float(val), 4)}
        for idx, val in drawdown.items()
    ]

    return {
        "max_drawdown_pct": round(max_drawdown_val, 2),
        "peak_date": str(peak_idx.date()),
        "trough_date": str(trough_idx.date()),
        "recovery_date": recovery_date,
        "currently_in_drawdown": currently_in_drawdown,
        "drawdown_series": drawdown_series,
        "period": period,
    }


def calculate_var(
    tickers: list,
    weights: list,
    period: str = "1y",
    confidence: float = 0.95,
) -> dict:
    """Calculate Value at Risk (VaR) and tail risk metrics using the historical method."""
    if period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Must be one of {sorted(VALID_PERIODS)}.")

    portfolio_daily, _ = _download_portfolio_daily(tickers, weights, period)

    if len(portfolio_daily) < 20:
        raise ValueError("Insufficient data for VaR (need at least 20 trading days).")

    returns_pct = portfolio_daily * 100

    # Historical VaR: nth percentile of the daily return distribution
    var_95 = float(np.percentile(returns_pct, 5))   # 5th percentile → 1-in-20 bad day
    var_99 = float(np.percentile(returns_pct, 1))   # 1st percentile → 1-in-100 bad day

    # CVaR (Expected Shortfall) at 95%: mean of all returns worse than VaR 95
    tail_mask = returns_pct <= var_95
    cvar_95 = float(returns_pct[tail_mask].mean()) if tail_mask.any() else var_95

    # Worst single day
    worst_idx = portfolio_daily.idxmin()
    worst_day_loss = float(portfolio_daily[worst_idx] * 100)

    return {
        "var_95_pct": round(var_95, 2),
        "var_99_pct": round(var_99, 2),
        "cvar_95_pct": round(cvar_95, 2),
        "worst_day_loss_pct": round(worst_day_loss, 2),
        "worst_day_date": str(worst_idx.date()),
        "period": period,
        "confidence_level": confidence,
    }
