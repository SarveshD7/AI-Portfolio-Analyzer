# Rolling Metrics Feature — Design Spec

**Date:** 2026-05-24  
**Status:** Approved

---

## Overview

Add a rolling metrics tool that answers queries like "has my portfolio gotten riskier over time?" by computing and visualising three risk metrics as rolling time series: annualised volatility, Sharpe ratio, and VaR 95%. The canvas renders three stacked Plotly subplots sharing a common x-axis.

---

## Architecture

### Backend — `backend/tools/risk.py`

New function `calculate_rolling_metrics(tickers, weights, period, window_days, start_date, end_date)`.

- Downloads price data once via the existing `_download_portfolio_daily` helper.
- Computes three rolling series over a `window_days`-day window (default 30 trading days):
  - **Rolling annualised volatility** — `std() * sqrt(252)` over the window, expressed as %.
  - **Rolling Sharpe** — `(annualised_return - risk_free_rate) / annualised_vol` over the window.
  - **Rolling VaR 95%** — 5th percentile of daily returns within the window, expressed as %.
- Derives `risk_trend` by comparing the mean rolling volatility of the most recent 30 days against the 30 days before that:
  - `"increasing"` if recent mean > prior mean by more than 1 pp
  - `"decreasing"` if recent mean < prior mean by more than 1 pp
  - `"stable"` otherwise

**Return shape:**
```json
{
  "period": "1y",
  "window_days": 30,
  "rolling_volatility": [{"date": "YYYY-MM-DD", "value": 18.4}, ...],
  "rolling_sharpe":     [{"date": "YYYY-MM-DD", "value": 0.82}, ...],
  "rolling_var_95":     [{"date": "YYYY-MM-DD", "value": -1.23}, ...],
  "risk_trend": "increasing",
  "current_vol_pct": 22.1,
  "current_sharpe": 0.61,
  "current_var_95_pct": -1.8
}
```

Accepts `period` OR `start_date`+`end_date`, identical to all other risk tools. Valid periods: `1mo`, `3mo`, `6mo`, `1y`, `3y`, `5y`.

---

### Backend — `backend/agent.py`

**New tool `rolling_metrics_tool`:**

```python
@tool
def rolling_metrics_tool(
    tickers, weights,
    period="1y",
    start_date=None,
    end_date=None,
    window_days=30,
) -> dict:
```

Docstring triggers:
- "has my portfolio gotten riskier / safer"
- "rolling risk", "rolling volatility", "rolling Sharpe"
- "how has risk changed over time"
- "is my portfolio more volatile now"
- "risk trend over [period]"

**`_TOOL_VIZ_MAP` addition:**
```python
"rolling_metrics_tool": "rolling_metrics_chart"
```

**`_VIZ_ONLY_KEYS` addition** — strips the three series arrays from the ToolMessage so the LLM sees only summary scalars:
```python
"rolling_metrics_tool": ["rolling_volatility", "rolling_sharpe", "rolling_var_95"]
```

The LLM's text response is written using `risk_trend` and the three `current_*` scalars.

---

### Frontend — `frontend/app.py`

New branch in `render_canvas` for `viz_type == "rolling_metrics_chart"`.

**Chart:** `make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06)`

| Row | Metric | Fill colour | "Riskier" direction |
|-----|--------|-------------|---------------------|
| 1 | Rolling Volatility (%) | Orange `#d97706` | Up |
| 2 | Rolling Sharpe | Blue `#2563eb` (positive) / Red `#dc2626` (negative) | Down |
| 3 | Rolling VaR 95% (%) | Red `#dc2626` | Down (more negative) |

Each panel uses `fill="tozeroy"` with 6% opacity fill. All three share the x-axis; x-tick labels show only on the bottom panel.

**Summary row below chart:** Three columns showing `current_vol_pct`, `current_sharpe`, `current_var_95_pct` as metric cards.

**Risk trend badge:** Coloured indicator below the summary row:
- `"increasing"` → danger red, "Risk increasing ↑"
- `"decreasing"` → success green, "Risk decreasing ↓"  
- `"stable"` → muted grey, "Risk stable →"

---

## Data Flow

```
User: "has my portfolio gotten riskier this year?"
  → LLM selects rolling_metrics_tool (start_date=2026-01-01, end_date=2026-05-24)
  → calculate_rolling_metrics(...) runs in a single yfinance download pass
  → Tool returns full payload including series + summary scalars
  → _VIZ_ONLY_KEYS strips series arrays from ToolMessage
  → LLM writes 2–3 sentence verdict using risk_trend + current_* values
  → _build_visualization("rolling_metrics_chart", tool_results, portfolio)
    passes full payload (including series) to frontend
  → render_canvas draws 3-panel subplot + summary + badge
```

---

## Constraints

- Minimum data requirement: `window_days * 2` trading days must be available, otherwise raise `ValueError` with a clear message.
- Rolling series starts from index `window_days` (first valid window), so the x-axis is shorter than the full period by one window.
- `window_days` defaults to 30; the LLM always passes the default unless the user explicitly requests a different window.
- No new API endpoint — uses the existing `/analyze` endpoint.
- No new session state keys required.
