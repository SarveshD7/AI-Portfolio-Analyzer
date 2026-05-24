# Rolling Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a rolling metrics tool that computes and visualises 30-day rolling volatility, Sharpe ratio, and VaR 95% over any user-specified period or date range, answering queries like "has my portfolio gotten riskier?"

**Architecture:** New `calculate_rolling_metrics` function in `risk.py` downloads prices once and computes all three rolling series in a single pass. The agent exposes this as `rolling_metrics_tool` with the same period/date-range signature as all other risk tools. The frontend renders a 3-panel Plotly subplot (shared x-axis) plus a risk trend badge.

**Tech Stack:** yfinance, pandas rolling, numpy, Plotly make_subplots, Streamlit, FastAPI (existing `/analyze` endpoint)

---

## File Map

| File | Change |
|------|--------|
| `backend/tools/risk.py` | Add `calculate_rolling_metrics` function at end of file |
| `backend/agent.py` | Update import line; add `rolling_metrics_tool`; update `_TOOL_VIZ_MAP`, `_VIZ_ONLY_KEYS`, `_tools`, `_build_visualization` |
| `frontend/app.py` | Add `rolling_metrics_chart` branch inside `render_canvas` |

---

## Task 1: `calculate_rolling_metrics` in `backend/tools/risk.py`

**Files:**
- Modify: `backend/tools/risk.py` — append new function at the bottom

- [ ] **Step 1: Append `calculate_rolling_metrics` to `backend/tools/risk.py`**

Add after the last function in the file (`calculate_var`):

```python
def calculate_rolling_metrics(
    tickers: list,
    weights: list,
    period: str = "1y",
    window_days: int = 30,
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

    if len(portfolio_daily) < window_days * 2:
        raise ValueError(
            f"Insufficient data for rolling metrics — need at least {window_days * 2} trading days, "
            f"got {len(portfolio_daily)}. Try a longer period or a smaller window."
        )

    # Rolling annualised volatility (%)
    rolling_vol = portfolio_daily.rolling(window_days).std() * math.sqrt(252) * 100

    # Rolling annualised Sharpe
    rolling_mean = portfolio_daily.rolling(window_days).mean() * 252
    rolling_std  = portfolio_daily.rolling(window_days).std()  * math.sqrt(252)
    rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std.replace(0, float("nan"))

    # Rolling VaR 95% (%) — 5th percentile of daily returns in window
    rolling_var = (portfolio_daily * 100).rolling(window_days).quantile(0.05)

    # Align all three to the same valid (non-NaN) index
    common_idx   = rolling_vol.dropna().index
    vol_s        = rolling_vol[common_idx]
    sharpe_s     = rolling_sharpe[common_idx]
    var_s        = rolling_var[common_idx]

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
```

- [ ] **Step 2: Verify the function is importable**

Start a Python REPL inside the `backend/` directory and run:
```python
from tools.risk import calculate_rolling_metrics
print("OK")
```
Expected: `OK` with no import errors.

---

## Task 2: Wire the tool in `backend/agent.py`

**Files:**
- Modify: `backend/agent.py`

- [ ] **Step 1: Add `calculate_rolling_metrics` to the import line**

Find this line (line 17):
```python
from tools.risk import calculate_sharpe_ratio, calculate_max_drawdown, calculate_var, calculate_beta
```

Replace with:
```python
from tools.risk import calculate_sharpe_ratio, calculate_max_drawdown, calculate_var, calculate_beta, calculate_rolling_metrics
```

- [ ] **Step 2: Add `rolling_metrics_tool` after the `beta_tool` definition**

Insert this block immediately after `beta_tool` (before `benchmark_tool`):

```python
@tool
def rolling_metrics_tool(
    tickers: List[str],
    weights: List[float],
    period: Optional[str] = "1y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    window_days: int = 30,
) -> dict:
    """Compute rolling risk metrics over time to show how portfolio risk has evolved.

    Use when the user asks about:
    - "Has my portfolio gotten riskier / safer over time?"
    - Rolling risk, rolling volatility, rolling Sharpe ratio
    - "How has risk changed over [period]?"
    - "Is my portfolio more volatile now than before?"
    - "Show me the risk trend over the past year"
    - "Has volatility increased recently?"

    Args:
        tickers: List of stock tickers (e.g. ["RELIANCE.NS", "TCS.NS"])
        weights: Corresponding weights as decimals summing to 1.0 (e.g. [0.6, 0.4])
        period: "1mo", "3mo", "6mo", "1y", "3y", or "5y" — used when no date range given
        start_date: Start date in YYYY-MM-DD format (overrides period when paired with end_date)
        end_date: End date in YYYY-MM-DD format (overrides period when paired with start_date)
        window_days: Rolling window in trading days (default 30; do not change unless user asks)
    """
    return calculate_rolling_metrics(tickers, weights, period, window_days, start_date, end_date)
```

- [ ] **Step 3: Add `rolling_metrics_tool` to the `_tools` list**

Find `_tools = [` and add `rolling_metrics_tool` after `beta_tool`:

```python
_tools = [
    portfolio_returns_tool,
    sharpe_ratio_tool,
    max_drawdown_tool,
    var_tool,
    beta_tool,
    rolling_metrics_tool,       # ← add this line
    portfolio_composition_tool,
    concentration_tool,
    correlation_tool,
    benchmark_tool,
    simulate_portfolio_change_tool,
    revert_to_original_portfolio_tool,
    compare_portfolios_tool,
]
```

- [ ] **Step 4: Add entry to `_TOOL_VIZ_MAP`**

Find `_TOOL_VIZ_MAP = {` and add:
```python
    "rolling_metrics_tool":    "rolling_metrics_chart",
```
Place it after the `"beta_tool"` entry.

- [ ] **Step 5: Add entry to `_VIZ_ONLY_KEYS`**

Find `_VIZ_ONLY_KEYS = {` and add:
```python
    "rolling_metrics_tool": ["rolling_volatility", "rolling_sharpe", "rolling_var_95"],
```
Place it after the `"benchmark_tool"` entry. This strips the large series arrays from the ToolMessage so the LLM only sees summary scalars.

- [ ] **Step 6: Add `rolling_metrics_chart` case in `_build_visualization`**

Find `_build_visualization` and add this branch before the final `return {"type": None, "data": {}}`:

```python
    if viz_type == "rolling_metrics_chart" and tool_results:
        return {"type": "rolling_metrics_chart", "data": tool_results}
```

---

## Task 3: Render `rolling_metrics_chart` in `frontend/app.py`

**Files:**
- Modify: `frontend/app.py`

- [ ] **Step 1: Add the `plotly.subplots` import**

At the top of `frontend/app.py`, after the existing `import plotly.graph_objects as go` line, add:

```python
from plotly.subplots import make_subplots
```

- [ ] **Step 2: Add the `rolling_metrics_chart` renderer branch**

Inside `render_canvas`, find the `# ── Empty state` comment. Add the new branch immediately **before** that block:

```python
    # ── Rolling metrics chart ─────────────────────────────────────────────────
    elif viz_type == "rolling_metrics_chart":
        vol_df    = pd.DataFrame(viz_data.get("rolling_volatility", []))
        sharpe_df = pd.DataFrame(viz_data.get("rolling_sharpe",     []))
        var_df    = pd.DataFrame(viz_data.get("rolling_var_95",     []))

        if not vol_df.empty:
            period     = viz_data.get("period", "")
            window     = viz_data.get("window_days", 30)

            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.07,
                subplot_titles=[
                    "Rolling Volatility (Ann. %)",
                    "Rolling Sharpe Ratio",
                    "Rolling VaR 95% (%)",
                ],
            )

            # Row 1 — Volatility
            fig.add_trace(go.Scatter(
                x=vol_df["date"], y=vol_df["value"],
                mode="lines",
                line=dict(color=WARNING, width=2),
                fill="tozeroy",
                fillcolor="rgba(217,119,6,0.06)",
                name="Volatility",
                hovertemplate="%{x}<br><b>%{y:.2f}%</b><extra>Volatility</extra>",
            ), row=1, col=1)

            # Row 2 — Sharpe
            fig.add_trace(go.Scatter(
                x=sharpe_df["date"], y=sharpe_df["value"],
                mode="lines",
                line=dict(color=PRIMARY, width=2),
                fill="tozeroy",
                fillcolor="rgba(37,99,235,0.06)",
                name="Sharpe",
                hovertemplate="%{x}<br><b>%{y:.2f}</b><extra>Sharpe</extra>",
            ), row=2, col=1)
            fig.add_hline(y=0, line_color=BORDER, line_width=1, line_dash="dot", row=2, col=1)

            # Row 3 — VaR 95%
            fig.add_trace(go.Scatter(
                x=var_df["date"], y=var_df["value"],
                mode="lines",
                line=dict(color=DANGER, width=2),
                fill="tozeroy",
                fillcolor="rgba(220,38,38,0.06)",
                name="VaR 95%",
                hovertemplate="%{x}<br><b>%{y:.2f}%</b><extra>VaR 95%</extra>",
            ), row=3, col=1)

            fig.update_layout(
                height=540,
                margin=dict(t=52, b=36, l=8, r=8),
                plot_bgcolor=SURFACE,
                paper_bgcolor=SURFACE,
                font=dict(family="Inter, sans-serif", size=11, color=MUTED),
                title=dict(
                    text=f"Rolling Risk Metrics  ·  {window}-day window  ·  {period}",
                    font=dict(size=13, color=TEXT, family="Inter, sans-serif"),
                    x=0, xanchor="left",
                    pad=dict(l=4, t=2),
                ),
                showlegend=False,
                hoverlabel=dict(
                    bgcolor=SURFACE, bordercolor=BORDER,
                    font=dict(size=12, family="Inter, sans-serif", color=TEXT),
                ),
                modebar=dict(bgcolor="rgba(0,0,0,0)", color=MUTED_LT, activecolor=PRIMARY),
            )
            fig.update_xaxes(
                showgrid=True, gridcolor="#f1f5f9", gridwidth=1,
                linecolor=BORDER, linewidth=1, zeroline=False,
                tickfont=dict(size=11, color=MUTED),
            )
            fig.update_yaxes(
                showgrid=True, gridcolor="#f1f5f9", gridwidth=1,
                linecolor=BORDER, linewidth=1, zeroline=False,
                tickfont=dict(size=11, color=MUTED),
            )
            fig.update_yaxes(ticksuffix="%", row=1, col=1)
            fig.update_yaxes(ticksuffix="%", row=3, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # Summary metric row
            current_vol    = viz_data.get("current_vol_pct",    0)
            current_sharpe = viz_data.get("current_sharpe",     0)
            current_var    = viz_data.get("current_var_95_pct", 0)
            risk_trend     = viz_data.get("risk_trend",         "stable")

            c1, c2, c3 = st.columns(3)
            c1.metric("Current Volatility", f"{current_vol:.2f}%")
            c2.metric("Current Sharpe",     f"{current_sharpe:.2f}")
            c3.metric("Current VaR 95%",    f"{current_var:.2f}%")

            # Risk trend badge
            trend_cfg = {
                "increasing": (DANGER,  DANGER_BG,  "Risk increasing ↑"),
                "decreasing": (SUCCESS, SUCCESS_BG, "Risk decreasing ↓"),
                "stable":     (MUTED,   GRAY_BG,    "Risk stable →"),
            }
            t_fg, t_bg, t_label = trend_cfg.get(risk_trend, (MUTED, GRAY_BG, "Risk stable →"))
            st.markdown(
                f"""<div style="display:inline-flex;align-items:center;gap:8px;
                  background:{t_bg};border:1px solid {t_fg}33;
                  border-radius:8px;padding:8px 18px;margin-top:10px">
                  <span style="font-size:14px;font-weight:700;color:{t_fg}">{t_label}</span>
                </div>""",
                unsafe_allow_html=True,
            )
```

- [ ] **Step 3: Smoke test end-to-end**

1. Start the backend: `uvicorn main:app --reload` (from `backend/`)
2. Start the frontend: `streamlit run app.py` (from `frontend/`)
3. Upload any portfolio CSV
4. Ask: **"Has my portfolio gotten riskier over the past year?"**

Expected:
- Canvas shows 3 stacked panels (Volatility / Sharpe / VaR 95%) with a shared date x-axis
- Three metric summary cards appear below
- A coloured risk trend badge ("Risk increasing ↑" / "Risk decreasing ↓" / "Risk stable →") appears
- The chat response mentions the current volatility, Sharpe, and whether risk is trending up or down

5. Ask: **"Show me rolling risk from January 2026 to today"**

Expected: Same chart but x-axis scoped to Jan–May 2026 (date range overrides period).
