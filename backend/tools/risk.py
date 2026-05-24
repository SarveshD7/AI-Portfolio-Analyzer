import math

import numpy as np
import pandas as pd
import yfinance as yf


VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "3y", "5y"}


def _download_portfolio_daily(tickers, weights, period, start_date=None, end_date=None):
    """Shared helper: returns aligned (portfolio_daily, aligned_weights, close)."""
    period = period or "1y"
    dl_kwargs = {"start": start_date, "end": end_date} if (start_date and end_date) else {"period": period}
    try:
        raw = yf.download(tickers, auto_adjust=True, progress=False, **dl_kwargs)
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
    start_date: str = None,
    end_date: str = None,
) -> dict:
    period = period or "1y"
    if not (start_date and end_date) and period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Must be one of {sorted(VALID_PERIODS)}.")
    period_label = f"{start_date} → {end_date}" if (start_date and end_date) else period

    portfolio_daily, _ = _download_portfolio_daily(tickers, weights, period, start_date, end_date)

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
        "period": period_label,
    }


def calculate_max_drawdown(
    tickers: list,
    weights: list,
    period: str = "1y",
    start_date: str = None,
    end_date: str = None,
) -> dict:
    """Calculate maximum drawdown — the largest peak-to-trough decline — for the portfolio."""
    period = period or "1y"
    if not (start_date and end_date) and period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Must be one of {sorted(VALID_PERIODS)}.")
    period_label = f"{start_date} → {end_date}" if (start_date and end_date) else period

    portfolio_daily, _ = _download_portfolio_daily(tickers, weights, period, start_date, end_date)

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
        "period": period_label,
    }


def calculate_beta(
    tickers: list,
    weights: list,
    period: str = "1y",
    benchmark: str = "^NSEI",
    risk_free_rate: float = 0.06,
    start_date: str = None,
    end_date: str = None,
) -> dict:
    """Calculate portfolio beta and alpha relative to a benchmark index."""
    period = period or "1y"
    if not (start_date and end_date) and period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Must be one of {sorted(VALID_PERIODS)}.")
    period_label = f"{start_date} → {end_date}" if (start_date and end_date) else period
    dl_kwargs = {"start": start_date, "end": end_date} if (start_date and end_date) else {"period": period}

    all_tickers = tickers + [benchmark]
    try:
        raw = yf.download(all_tickers, auto_adjust=True, progress=False, **dl_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data: {e}")

    if raw.empty:
        raise ValueError("No data returned. Check tickers and network connection.")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.dropna(axis=1, how="all")

    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise ValueError(f"No data for ticker(s): {missing}")
    if benchmark not in close.columns:
        raise ValueError(f"No data for benchmark '{benchmark}'.")

    daily = close.pct_change().dropna()
    bench_returns = daily[benchmark]

    ticker_weight = dict(zip(tickers, weights))
    aligned = [t for t in tickers if t in close.columns]
    aligned_weights = np.array([ticker_weight[t] for t in aligned])
    port_returns = daily[aligned].dot(aligned_weights)

    cov_matrix = np.cov(port_returns, bench_returns)
    beta = float(cov_matrix[0, 1] / cov_matrix[1, 1]) if cov_matrix[1, 1] != 0 else 0.0

    n_days = len(port_returns)
    ann_port = float(((1 + port_returns).prod() ** (252 / n_days)) - 1)
    ann_bench = float(((1 + bench_returns).prod() ** (252 / n_days)) - 1)
    alpha = ann_port - (risk_free_rate + beta * (ann_bench - risk_free_rate))

    correlation = float(np.corrcoef(port_returns, bench_returns)[0, 1])
    r_squared = round(correlation ** 2 * 100, 2)

    return {
        "beta":                    round(beta, 3),
        "alpha_pct":               round(alpha * 100, 2),
        "correlation":             round(correlation, 3),
        "r_squared_pct":           r_squared,
        "annualized_port_return_pct":  round(ann_port * 100, 2),
        "annualized_bench_return_pct": round(ann_bench * 100, 2),
        "benchmark":               benchmark,
        "period":                  period_label,
    }


def calculate_var(
    tickers: list,
    weights: list,
    period: str = "1y",
    confidence: float = 0.95,
    start_date: str = None,
    end_date: str = None,
) -> dict:
    """Calculate Value at Risk (VaR) and tail risk metrics using the historical method."""
    period = period or "1y"
    if not (start_date and end_date) and period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Must be one of {sorted(VALID_PERIODS)}.")
    period_label = f"{start_date} → {end_date}" if (start_date and end_date) else period

    portfolio_daily, _ = _download_portfolio_daily(tickers, weights, period, start_date, end_date)

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
        "period": period_label,
        "confidence_level": confidence,
    }


def calculate_rolling_metrics(
    tickers: list,
    weights: list,
    period: str = "1y",
    start_date: str = None,
    end_date: str = None,
    risk_free_rate: float = 0.06,
) -> dict:
    """Compute rolling annualised volatility, Sharpe ratio, and VaR 95% over a sliding window."""
    period = period or "1y"
    if not (start_date and end_date) and period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Must be one of {sorted(VALID_PERIODS)}.")
    period_label = f"{start_date} → {end_date}" if (start_date and end_date) else period

    portfolio_daily, _ = _download_portfolio_daily(tickers, weights, period, start_date, end_date)

    # Derive window from data length so ~10 rolling points always appear regardless of period
    window_days = max(5, len(portfolio_daily) // 10)

    # Rolling annualised volatility (%)
    rolling_vol = portfolio_daily.rolling(window_days).std() * math.sqrt(252) * 100

    # Rolling annualised Sharpe
    rolling_mean = portfolio_daily.rolling(window_days).mean() * 252
    rolling_std  = portfolio_daily.rolling(window_days).std()  * math.sqrt(252)
    rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std.replace(0, float("nan"))

    # Rolling VaR 95% (%) — 5th percentile of daily returns in window
    rolling_var = (portfolio_daily * 100).rolling(window_days).quantile(0.05)

    # Align all three series to the same valid (non-NaN) index
    common_idx = rolling_vol.dropna().index
    vol_s      = rolling_vol[common_idx]
    sharpe_s   = rolling_sharpe[common_idx]
    var_s      = rolling_var[common_idx]

    def _to_list(s: pd.Series) -> list:
        return [
            {"date": str(idx.date()), "value": round(float(v), 4)}
            for idx, v in s.items()
            if not (v != v)  # exclude NaN
        ]

    # Risk trend: compare mean vol of most-recent 30 days vs prior 30 days
    if len(vol_s) >= 60:
        recent_mean = float(vol_s.iloc[-30:].mean())
        prior_mean  = float(vol_s.iloc[-60:-30].mean())
    else:
        half        = max(1, len(vol_s) // 2)
        recent_mean = float(vol_s.iloc[half:].mean())
        prior_mean  = float(vol_s.iloc[:half].mean())

    diff = recent_mean - prior_mean
    if diff > 1.0:
        risk_trend = "increasing"
    elif diff < -1.0:
        risk_trend = "decreasing"
    else:
        risk_trend = "stable"

    return {
        "period":             period_label,
        "window_days":        window_days,
        "rolling_volatility": _to_list(vol_s),
        "rolling_sharpe":     _to_list(sharpe_s),
        "rolling_var_95":     _to_list(var_s),
        "risk_trend":         risk_trend,
        "current_vol_pct":    round(float(vol_s.iloc[-1]),    2),
        "current_sharpe":     round(float(sharpe_s.iloc[-1]), 2),
        "current_var_95_pct": round(float(var_s.iloc[-1]),    2),
    }
