import html as html_lib
import io

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Kalpi Portfolio Analyzer", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in [
    ("portfolio", None),
    ("messages", []),
    ("portfolio_uploaded", False),
    ("current_visualization_type", None),
    ("current_visualization_data", {}),
    ("canvas_updated", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chat_bubble(role: str, content: str):
    escaped = html_lib.escape(content)
    if role == "user":
        st.markdown(
            f"""<div style="display:flex;justify-content:flex-end;margin:4px 0">
              <div style="background:#0084ff;color:#fff;padding:9px 13px;
                border-radius:18px 18px 4px 18px;max-width:88%;
                font-size:14px;line-height:1.4;word-wrap:break-word">
                {escaped}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div style="display:flex;justify-content:flex-start;margin:4px 0">
              <div style="background:#f0f2f6;color:#1a1a1a;padding:9px 13px;
                border-radius:18px 18px 18px 4px;max-width:88%;
                font-size:14px;line-height:1.4;word-wrap:break-word">
                {escaped}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_canvas(viz_type: str, viz_data: dict):
    if viz_type == "line_chart":
        daily = pd.DataFrame(viz_data.get("daily_returns", []))
        if not daily.empty:
            fig = go.Figure(
                go.Scatter(x=daily["date"], y=daily["return"], mode="lines",
                           line=dict(color="#0084ff", width=2), name="Daily Return (%)")
            )
            fig.update_layout(
                title=f"Daily Portfolio Returns  ·  {viz_data.get('period', '')}",
                xaxis_title="Date", yaxis_title="Return (%)",
                margin=dict(t=48, b=32, l=8, r=8),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
            fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Total Return", f"{viz_data.get('total_return_pct', 0):+.2f}%")

    elif viz_type == "metrics" and "var_95_pct" in viz_data:
        period = viz_data.get("period", "")
        st.caption(f"Value at Risk (Historical Method)  ·  {period}")

        def _risk_card(label, value, sublabel, bg, fg):
            st.markdown(
                f"""<div style="background:{bg};border-radius:10px;padding:14px 12px;
                  text-align:center;margin-bottom:6px">
                  <div style="font-size:11px;color:{fg};font-weight:600;
                    text-transform:uppercase;letter-spacing:.4px">{label}</div>
                  <div style="font-size:26px;font-weight:700;color:{fg};line-height:1.1">
                    {value}%
                  </div>
                  <div style="font-size:11px;color:{fg};margin-top:4px;opacity:.8">
                    {sublabel}
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _risk_card("VaR 95%", f"{viz_data['var_95_pct']:.2f}",
                       "Bad day (1 in 20)", "#f8d7da", "#842029")
        with c2:
            _risk_card("VaR 99%", f"{viz_data['var_99_pct']:.2f}",
                       "Very bad day (1 in 100)", "#dc3545", "#fff")
        with c3:
            _risk_card("CVaR 95%", f"{viz_data['cvar_95_pct']:.2f}",
                       "Avg tail loss", "#f0c0c4", "#6b0e17")
        with c4:
            _risk_card(f"Worst Day ({viz_data.get('worst_day_date', '')})",
                       f"{viz_data['worst_day_loss_pct']:.2f}",
                       "Single-day low", "#fce8e8", "#842029")

        var95 = abs(viz_data["var_95_pct"])
        var99 = abs(viz_data["var_99_pct"])
        st.info(
            f"On a typical bad day (1 in 20), you could expect to lose around **{var95:.1f}%**. "
            f"In extreme cases (1 in 100), losses could reach **{var99:.1f}%**. "
            f"The CVaR figure shows the average loss you'd experience beyond the VaR threshold "
            f"— the 'average of the worst days'."
        )

    elif viz_type == "metrics":
        sharpe = viz_data.get("sharpe_ratio", 0)
        period = viz_data.get("period", "")

        if sharpe > 1.0:
            s_color, s_bg, s_label, interp = (
                "#1e7e34", "#e6f4ea", "Good",
                "Strong risk-adjusted returns. The portfolio is well-compensated for the volatility it carries.",
            )
        elif sharpe >= 0.5:
            s_color, s_bg, s_label, interp = (
                "#856404", "#fff3cd", "Moderate",
                "Acceptable risk-adjusted returns. There may be room to improve the return-to-risk profile.",
            )
        else:
            s_color, s_bg, s_label, interp = (
                "#842029", "#f8d7da", "Poor",
                "Weak risk-adjusted returns. The portfolio may not be adequately compensating for the risk taken.",
            )

        st.markdown(
            f"""<div style="text-align:center;padding:24px 16px;background:{s_bg};
              border-radius:12px;margin-bottom:20px">
              <div style="font-size:12px;color:{s_color};font-weight:600;
                letter-spacing:.5px;text-transform:uppercase;margin-bottom:6px">
                Sharpe Ratio &nbsp;·&nbsp; {period}
              </div>
              <div style="font-size:52px;font-weight:700;color:{s_color};line-height:1">
                {sharpe:.2f}
              </div>
              <div style="font-size:13px;color:{s_color};margin-top:8px;
                background:rgba(0,0,0,.07);display:inline-block;
                padding:2px 10px;border-radius:10px;font-weight:600">
                {s_label}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Annualized Return", f"{viz_data.get('annualized_return_pct', 0):+.2f}%")
        c2.metric("Annualized Volatility", f"{viz_data.get('annualized_volatility_pct', 0):.2f}%")
        c3.metric("Risk-Free Rate", f"{viz_data.get('risk_free_rate_pct', 6.0):.1f}%")

        st.info(interp)

    elif viz_type == "drawdown_chart":
        series = viz_data.get("drawdown_series", [])
        if series:
            daily = pd.DataFrame(series)
            max_dd = viz_data.get("max_drawdown_pct", 0)
            trough_date = viz_data.get("trough_date", "")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily["date"], y=daily["drawdown"],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(220,53,69,0.15)",
                line=dict(color="#dc3545", width=1.5),
                name="Drawdown (%)",
            ))
            if trough_date:
                fig.add_trace(go.Scatter(
                    x=[trough_date], y=[max_dd],
                    mode="markers+text",
                    marker=dict(color="#dc3545", size=11, symbol="circle"),
                    text=[f"  {max_dd:.1f}%"],
                    textposition="middle right",
                    textfont=dict(color="#dc3545", size=12),
                    name="Max Drawdown",
                    showlegend=False,
                ))
            fig.update_layout(
                title=f"Portfolio Drawdown  ·  {viz_data.get('period', '')}",
                xaxis_title="Date", yaxis_title="Drawdown (%)",
                margin=dict(t=48, b=32, l=8, r=8),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(ticksuffix="%"),
            )
            fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
            fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(
                f"""<div style="background:#f8d7da;border-radius:8px;padding:12px 8px;text-align:center">
                  <div style="font-size:11px;color:#842029;font-weight:600;text-transform:uppercase">Max Drawdown</div>
                  <div style="font-size:24px;font-weight:700;color:#842029">{max_dd:.1f}%</div>
                </div>""",
                unsafe_allow_html=True,
            )
            c2.metric("Peak Date", viz_data.get("peak_date", "—"))
            c3.metric("Trough Date", trough_date or "—")
            if viz_data.get("currently_in_drawdown"):
                c4.metric("Status", "In drawdown")
                st.warning("Portfolio is currently still in a drawdown phase.")
            else:
                c4.metric("Recovery Date", viz_data.get("recovery_date") or "—")
                st.success("Portfolio has fully recovered from the maximum drawdown.")

    elif viz_type == "benchmark_chart":
        port_series = pd.DataFrame(viz_data.get("portfolio_cumulative", []))
        bm_series   = pd.DataFrame(viz_data.get("benchmark_cumulative", []))
        bm_name     = viz_data.get("benchmark_name", "Benchmark")

        if not port_series.empty and not bm_series.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=port_series["date"], y=port_series["value"],
                mode="lines", name="My Portfolio",
                line=dict(color="#0084ff", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=bm_series["date"], y=bm_series["value"],
                mode="lines", name=bm_name,
                line=dict(color="#f97316", width=2, dash="dot"),
            ))
            fig.add_hline(y=100, line_dash="dash", line_color="#ccc", line_width=1)
            fig.update_layout(
                title=f"Portfolio vs {bm_name}  ·  {viz_data.get('period', '')}  (indexed to 100)",
                xaxis_title="Date",
                yaxis_title="Growth of ₹100",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=64, b=32, l=8, r=8),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
            fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
            st.plotly_chart(fig, use_container_width=True)

            port_ret = viz_data.get("portfolio_total_return_pct", 0)
            bm_ret   = viz_data.get("benchmark_total_return_pct", 0)
            alpha    = viz_data.get("alpha_pct", 0)
            beta     = viz_data.get("beta", 0)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Portfolio Return", f"{port_ret:+.2f}%")
            c2.metric(f"{bm_name} Return", f"{bm_ret:+.2f}%")
            c3.metric("Alpha (excess return)", f"{alpha:+.2f}%",
                      delta="Outperformed" if alpha > 0 else "Underperformed",
                      delta_color="normal" if alpha > 0 else "inverse")
            c4.metric("Beta", f"{beta:.2f}",
                      delta="Low sensitivity" if beta < 0.8 else ("High sensitivity" if beta > 1.2 else "Market-like"),
                      delta_color="off")

            if viz_data.get("outperformed"):
                st.success(
                    f"Your portfolio outperformed {bm_name} by **{alpha:.2f}%** over this period."
                )
            else:
                st.warning(
                    f"Your portfolio underperformed {bm_name} by **{abs(alpha):.2f}%** over this period."
                )

    elif viz_type == "correlation_heatmap":
        matrix = viz_data.get("matrix", [])
        tickers = viz_data.get("tickers", [])
        if matrix and tickers:
            text_annotations = [[f"{v:.2f}" for v in row] for row in matrix]
            fig = go.Figure(go.Heatmap(
                z=matrix,
                x=tickers,
                y=tickers,
                colorscale="RdBu_r",   # red = high correlation, blue = low / negative
                zmin=-1,
                zmax=1,
                text=text_annotations,
                texttemplate="%{text}",
                textfont=dict(size=11),
                colorbar=dict(title="r", thickness=14),
            ))
            fig.update_layout(
                title=f"Asset Correlation Matrix  ·  {viz_data.get('period', '')}",
                margin=dict(t=48, b=80, l=100, r=8),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(tickangle=-35, side="bottom"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

            high_corr = viz_data.get("high_correlations", [])
            low_corr  = viz_data.get("low_correlations", [])

            if high_corr:
                pairs = ", ".join(
                    f"**{p['assets'][0]}** & **{p['assets'][1]}** ({p['correlation']:.2f})"
                    for p in high_corr[:3]
                )
                st.warning(f"High-correlation pairs (move together — limited diversification): {pairs}")

            if low_corr:
                pairs = ", ".join(
                    f"**{p['assets'][0]}** & **{p['assets'][1]}** ({p['correlation']:.2f})"
                    for p in low_corr[:3]
                )
                st.success(f"Low-correlation pairs (good diversifiers): {pairs}")

            if not high_corr and not low_corr:
                st.info(
                    "All pairs have moderate correlation (0.30–0.70), "
                    "suggesting reasonable but not exceptional diversification."
                )

    elif viz_type == "concentration_pie":
        breakdown = viz_data.get("breakdown", [])
        if breakdown:
            labels = [b["label"] for b in breakdown]
            values = [b["weight_pct"] for b in breakdown]
            breakdown_type = viz_data.get("breakdown_type", "sector")
            title_map = {
                "sector":      "Sector Concentration",
                "asset_class": "Asset Class Breakdown",
                "factor":      "Market-Cap Factor Exposure",
            }
            title = title_map.get(breakdown_type, "Portfolio Concentration")

            fig = go.Figure(go.Pie(
                labels=labels,
                values=values,
                hole=0.35,
                textinfo="label+percent",
                textposition="auto",
                marker=dict(line=dict(color="white", width=2)),
            ))
            fig.update_layout(
                title=title,
                margin=dict(t=48, b=32, l=8, r=8),
                paper_bgcolor="white",
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

            dominant = viz_data.get("dominant", {})
            label_display = breakdown_type.replace("_", " ")
            if viz_data.get("concentration_warning"):
                st.warning(
                    f"Concentration alert: **{dominant.get('label')}** accounts for "
                    f"**{dominant.get('weight_pct', 0):.1f}%** of your portfolio — "
                    f"you may be over-indexed to this {label_display}."
                )
            else:
                st.success(
                    f"Your portfolio looks reasonably diversified across {label_display}s. "
                    f"The largest exposure (**{dominant.get('label')}**) is "
                    f"{dominant.get('weight_pct', 0):.1f}%."
                )

            st.markdown("**Breakdown**")
            col_name = {"sector": "Sector", "asset_class": "Asset Class", "factor": "Factor"}.get(
                breakdown_type, "Category"
            )
            df = pd.DataFrame(breakdown)
            df.columns = [col_name, "Weight (%)"]
            st.dataframe(df, use_container_width=True, hide_index=True)

    elif viz_type == "pie_chart":
        tickers = viz_data.get("tickers", [])
        weights = viz_data.get("weights", [])
        if tickers and weights:
            fig = go.Figure(
                go.Pie(labels=tickers, values=weights, hole=0.35,
                       marker=dict(line=dict(color="white", width=2)))
            )
            fig.update_layout(
                title="Portfolio Allocation",
                margin=dict(t=48, b=32, l=8, r=8),
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "portfolio_pie":
        holdings = viz_data.get("holdings", [])
        if holdings:
            names = [h["name"] for h in holdings]
            wts = [h["weight_pct"] for h in holdings]

            fig = go.Figure(go.Pie(
                labels=names,
                values=wts,
                hole=0.35,
                textinfo="label+percent",
                textposition="auto",
                marker=dict(line=dict(color="white", width=2)),
            ))
            fig.update_layout(
                title="Portfolio Composition",
                margin=dict(t=48, b=32, l=8, r=8),
                paper_bgcolor="white",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Total Holdings", viz_data.get("total_holdings", 0))
            largest = viz_data.get("largest_holding", {})
            c2.metric(
                "Largest Position",
                f"{largest.get('weight_pct', 0):.1f}%",
                delta=largest.get("name", ""),
                delta_color="off",
            )

            st.markdown("**All Holdings**")
            df = pd.DataFrame(holdings)[["name", "ticker", "weight_pct"]]
            df.columns = ["Company", "Ticker", "Weight (%)"]
            st.dataframe(df, use_container_width=True, hide_index=True)

    else:
        st.markdown(
            """<div style="display:flex;align-items:center;justify-content:center;
              height:320px;color:#aaa;font-size:15px;
              border:2px dashed #ddd;border-radius:12px;margin-top:16px">
              Ask a question to see analysis here
            </div>""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Step 1: Upload
# ---------------------------------------------------------------------------
if not st.session_state.portfolio_uploaded:
    st.title("Kalpi Portfolio Analyzer")
    st.subheader("Upload Your Portfolio")

    uploaded = st.file_uploader("Portfolio CSV (columns: Ticker, Weight)", type="csv")

    if st.button("Analyze"):
        if uploaded is None:
            st.error("Please upload a CSV file first.")
        else:
            df = pd.read_csv(io.StringIO(uploaded.getvalue().decode("utf-8")))
            if "Ticker" not in df.columns or "Weight" not in df.columns:
                st.error("CSV must have 'Ticker' and 'Weight' columns.")
            else:
                df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
                if df["Weight"].isna().any():
                    st.error("All weights must be numeric.")
                elif abs(df["Weight"].sum() - 1.0) > 0.01:
                    st.error(f"Weights must sum to 1.0 (got {df['Weight'].sum():.4f}).")
                else:
                    st.session_state.portfolio = {
                        "tickers": df["Ticker"].tolist(),
                        "weights": df["Weight"].tolist(),
                        "period": "1y",
                    }
                    st.session_state.portfolio_uploaded = True
                    st.rerun()

# ---------------------------------------------------------------------------
# Step 2: Chat + Canvas
# ---------------------------------------------------------------------------
else:
    portfolio = st.session_state.portfolio

    # Top bar
    top_left, top_right = st.columns([5, 1])
    with top_left:
        holdings = "  |  ".join(
            f"{t} {w * 100:.0f}%" for t, w in zip(portfolio["tickers"], portfolio["weights"])
        )
        st.markdown(f"**Kalpi Portfolio Analyzer** &nbsp;·&nbsp; <span style='color:#666;font-size:13px'>{holdings}</span>",
                    unsafe_allow_html=True)
    with top_right:
        if st.button("New Analysis"):
            for key in ("portfolio", "messages", "current_visualization_type", "current_visualization_data"):
                st.session_state[key] = None if key == "portfolio" else ([] if key == "messages" else {})
            st.session_state.portfolio_uploaded = False
            st.session_state.canvas_updated = False
            st.rerun()

    st.divider()

    # Two-column layout
    chat_col, canvas_col = st.columns([1, 2])

    # ── Left: Chat ──────────────────────────────────────────────────────────
    with chat_col:
        st.markdown(
            "<div style='font-weight:600;font-size:15px;margin-bottom:8px'>Chat</div>",
            unsafe_allow_html=True,
        )

        # Fixed-height scrollable message area
        with st.container(height=420):
            for msg in st.session_state.messages:
                chat_bubble(msg["role"], msg["content"])

        # Auto-scroll the height-constrained container to the bottom on every rerun.
        # Streamlit wraps st.container(height=N) in a div with data-testid="stVerticalBlockBorderWrapper".
        components.html(
            """<script>
            setTimeout(function () {
                try {
                    var containers = window.parent.document.querySelectorAll(
                        '[data-testid="stVerticalBlockBorderWrapper"]'
                    );
                    containers.forEach(function (el) {
                        if (el.scrollHeight > el.clientHeight) {
                            el.scrollTop = el.scrollHeight;
                        }
                    });
                } catch (e) {}
            }, 120);
            </script>""",
            height=0,
        )

        # Input form
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "message", label_visibility="collapsed",
                placeholder="Ask about your portfolio…"
            )
            submitted = st.form_submit_button("Send", use_container_width=True)

        if submitted and user_input.strip():
            prompt = user_input.strip()
            st.session_state.messages.append({"role": "user", "content": prompt})

            payload = {
                "tickers": portfolio["tickers"],
                "weights": portfolio["weights"],
                "period": portfolio["period"],
                "question": prompt,
            }

            with st.spinner("Thinking…"):
                try:
                    resp = requests.post("http://localhost:8000/analyze", json=payload)
                    if resp.status_code == 200:
                        result = resp.json()
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["response"],
                        })
                        viz = result.get("visualization", {})
                        st.session_state.current_visualization_type = viz.get("type")
                        st.session_state.current_visualization_data = viz.get("data", {})
                        st.session_state.canvas_updated = True
                    else:
                        detail = resp.json().get("detail", resp.text)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error: {detail}",
                        })
                except requests.exceptions.ConnectionError:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Could not connect to backend. Is it running on port 8000?",
                    })

            st.rerun()

    # ── Right: Canvas ────────────────────────────────────────────────────────
    with canvas_col:
        canvas_header = "Portfolio Analysis Canvas"
        if st.session_state.canvas_updated:
            st.markdown(
                f"<div style='font-weight:600;font-size:15px;margin-bottom:8px'>"
                f"{canvas_header} "
                f"<span style='background:#e6f4ea;color:#1e7e34;font-size:11px;"
                f"padding:2px 8px;border-radius:10px;font-weight:500'>Updated</span></div>",
                unsafe_allow_html=True,
            )
            st.session_state.canvas_updated = False
        else:
            st.markdown(
                f"<div style='font-weight:600;font-size:15px;margin-bottom:8px'>{canvas_header}</div>",
                unsafe_allow_html=True,
            )

        render_canvas(
            st.session_state.current_visualization_type,
            st.session_state.current_visualization_data,
        )
