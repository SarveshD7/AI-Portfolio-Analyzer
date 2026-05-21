import html as html_lib
import io

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Kalpi · Portfolio Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in [
    ("portfolio", None),
    ("original_portfolio", None),
    ("current_portfolio", None),
    ("modification_history", []),
    ("portfolio_is_modified", False),
    ("messages", []),
    ("portfolio_uploaded", False),
    ("current_visualization_type", None),
    ("current_visualization_data", {}),
    ("canvas_updated", False),
    ("current_suggestions", []),
    ("explored_suggestions", []),
    ("accumulated_analysis", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
PRIMARY    = "#2563eb"
PRIMARY_BG = "#eff6ff"
SUCCESS    = "#059669"
SUCCESS_BG = "#ecfdf5"
WARNING    = "#d97706"
WARNING_BG = "#fffbeb"
DANGER     = "#dc2626"
DANGER_BG  = "#fef2f2"
SURFACE    = "#ffffff"
GRAY_BG    = "#f8fafc"
BORDER     = "#e2e8f0"
TEXT       = "#1e293b"
MUTED      = "#64748b"
MUTED_LT   = "#94a3b8"
WHITE      = "#ffffff"

CHART_COLORS = [
    "#2563eb", "#059669", "#d97706", "#7c3aed",
    "#0891b2", "#db2777", "#dc2626", "#78716c",
]

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}}

#MainMenu {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ display: none !important; }}
footer {{ display: none !important; }}
.stDeployButton {{ display: none !important; }}

.block-container {{
    padding-top: 1.4rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1440px !important;
}}

::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: #f1f5f9; }}
::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 99px; }}
::-webkit-scrollbar-thumb:hover {{ background: #94a3b8; }}

/* Buttons */
.stButton > button {{
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
    padding: 6px 16px !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.01em !important;
}}
.stButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.12) !important;
}}
.stButton > button[kind="secondary"] {{
    background: {SURFACE} !important;
    color: {TEXT} !important;
    border: 1.5px solid {BORDER} !important;
}}
.stButton > button[kind="secondary"]:hover {{
    border-color: #94a3b8 !important;
    background: {GRAY_BG} !important;
}}

.stDownloadButton > button {{
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
}}
.stDownloadButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.1) !important;
}}

.stFormSubmitButton > button {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.15s ease !important;
}}
.stFormSubmitButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.25) !important;
}}

/* Text input */
.stTextInput > div > div > input {{
    border-radius: 8px !important;
    font-size: 14px !important;
    border: 1.5px solid {BORDER} !important;
    padding: 9px 12px !important;
    background: #fafafa !important;
    color: #000000 !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}}
.stTextInput > div > div > input::placeholder {{
    color: #9ca3af !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: {PRIMARY} !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.08) !important;
    background: white !important;
    color: #000000 !important;
    outline: none !important;
}}

/* File uploader */
[data-testid="stFileUploader"] section {{
    border: 2px dashed #d1d5db !important;
    border-radius: 12px !important;
    background: {GRAY_BG} !important;
    transition: all 0.15s ease !important;
}}
[data-testid="stFileUploader"] section:hover {{
    border-color: {PRIMARY} !important;
    background: {PRIMARY_BG} !important;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    border-radius: 10px !important;
    overflow: hidden !important;
}}

/* Metrics */
[data-testid="metric-container"] {{
    background: {GRAY_BG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
}}
[data-testid="stMetricLabel"] p {{
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: {TEXT} !important;
}}
[data-testid="stMetricValue"] {{
    font-size: 22px !important;
    font-weight: 700 !important;
    color: {TEXT} !important;
}}
[data-testid="stMetricDelta"] {{
    color: {MUTED} !important;
}}

/* Expander */
[data-testid="stExpander"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}}

/* Alerts */
.stAlert {{
    border-radius: 10px !important;
    font-size: 13.5px !important;
}}

hr {{ border: none; border-top: 1px solid {BORDER}; margin: 0; }}

/* Animations */
@keyframes fadeSlideIn {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes badgePop {{
    0%   {{ transform: scale(0.8); opacity: 0; }}
    65%  {{ transform: scale(1.06); opacity: 1; }}
    100% {{ transform: scale(1);   opacity: 1; }}
}}
@keyframes dotGlow {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(34,197,94,0.5); }}
    50%       {{ box-shadow: 0 0 0 5px rgba(34,197,94,0); }}
}}

.msg-fade   {{ animation: fadeSlideIn 0.22s ease forwards; }}
.badge-pop  {{ animation: badgePop 0.35s ease forwards; display: inline-flex; align-items: center; }}
.dot-active {{ animation: dotGlow 2.4s ease-in-out infinite; }}

/* Suggestion pills */
.sug-pill > button {{
    background: #2d2d2d !important;
    color: #e0e0e0 !important;
    border: none !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 5px 13px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    cursor: pointer !important;
    transition: background 0.15s ease, transform 0.15s ease !important;
    letter-spacing: 0.01em !important;
    animation: fadeSlideIn 0.2s ease forwards !important;
}}
.sug-pill > button:hover {{
    background: #3d3d3d !important;
    transform: scale(1.02) translateY(0) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25) !important;
}}
.sug-pill-high > button   {{ border-left: 3px solid #d97706 !important; border-radius: 4px 20px 20px 4px !important; }}
.sug-pill-medium > button {{ border-left: 3px solid #2563eb !important; border-radius: 4px 20px 20px 4px !important; }}
.sug-pill-low > button    {{ border-left: 3px solid #64748b !important; border-radius: 4px 20px 20px 4px !important; }}
</style>
"""


def _inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chart layout helper
# ---------------------------------------------------------------------------
def _chart_layout(title="", period="", height=380):
    label = f"{title}  ·  {period}" if period else title
    return dict(
        title=dict(
            text=label,
            font=dict(size=13, color=TEXT, family="Inter, sans-serif"),
            x=0, xanchor="left",
            pad=dict(l=4, t=2),
        ),
        height=height,
        margin=dict(t=52, b=36, l=8, r=8),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(family="Inter, sans-serif", size=11, color=MUTED),
        xaxis=dict(
            showgrid=True, gridcolor="#f1f5f9", gridwidth=1,
            linecolor=BORDER, linewidth=1, zeroline=False,
            tickfont=dict(size=11, color=MUTED),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#f1f5f9", gridwidth=1,
            linecolor=BORDER, linewidth=1, zeroline=False,
            tickfont=dict(size=11, color=MUTED),
        ),
        hoverlabel=dict(
            bgcolor=SURFACE, bordercolor=BORDER,
            font=dict(size=12, family="Inter, sans-serif", color=TEXT),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(size=11, family="Inter, sans-serif", color=TEXT),
        ),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=MUTED_LT, activecolor=PRIMARY),
    )


# ---------------------------------------------------------------------------
# Chat bubble
# ---------------------------------------------------------------------------
def chat_bubble(role: str, content: str):
    escaped = html_lib.escape(content)
    if role == "user":
        st.markdown(
            f"""<div class="msg-fade" style="display:flex;justify-content:flex-end;
              margin:6px 0 6px 20px">
              <div style="background:{PRIMARY};color:#fff;padding:10px 14px;
                border-radius:16px 4px 16px 16px;max-width:88%;font-size:13.5px;
                line-height:1.55;word-break:break-word;
                box-shadow:0 1px 4px rgba(37,99,235,0.2)">
                {escaped}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div class="msg-fade" style="display:flex;align-items:flex-start;
              gap:8px;margin:6px 20px 6px 0">
              <div style="min-width:28px;height:28px;background:{PRIMARY};border-radius:50%;
                display:flex;align-items:center;justify-content:center;color:#fff;
                font-size:11px;font-weight:700;flex-shrink:0;margin-top:2px;
                box-shadow:0 2px 6px rgba(37,99,235,0.28)">K</div>
              <div style="background:#f1f5f9;color:{TEXT};padding:10px 14px;
                border-radius:4px 16px 16px 16px;max-width:88%;font-size:13.5px;
                line-height:1.55;word-break:break-word">
                {escaped}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Canvas renderer
# ---------------------------------------------------------------------------
def render_canvas(viz_type: str, viz_data: dict):

    # ── Line chart ────────────────────────────────────────────────────────────
    if viz_type == "line_chart":
        daily = pd.DataFrame(viz_data.get("daily_returns", []))
        if not daily.empty:
            period = viz_data.get("period", "")
            total  = viz_data.get("total_return_pct", 0)
            color  = SUCCESS if total >= 0 else DANGER

            fig = go.Figure(go.Scatter(
                x=daily["date"], y=daily["return"],
                mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)",
                name="Daily Return (%)",
                hovertemplate="%{x}<br><b>%{y:.2f}%</b><extra></extra>",
            ))
            fig.add_hline(y=0, line_color=BORDER, line_width=1.5, line_dash="dot")
            fig.update_layout(**_chart_layout("Daily Portfolio Returns", period))
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig, use_container_width=True)

            sign = "+" if total >= 0 else ""
            badge_bg = SUCCESS_BG if total >= 0 else DANGER_BG
            badge_fg = SUCCESS if total >= 0 else DANGER
            st.markdown(
                f"""<div style="display:inline-flex;align-items:center;gap:6px;
                  background:{badge_bg};border:1px solid {badge_fg}22;
                  border-radius:8px;padding:8px 16px;margin-top:4px">
                  <span style="font-size:11px;font-weight:600;text-transform:uppercase;
                    letter-spacing:.04em;color:{badge_fg}">Total Return</span>
                  <span style="font-size:20px;font-weight:700;color:{badge_fg}">
                    {sign}{total:.2f}%
                  </span>
                </div>""",
                unsafe_allow_html=True,
            )

            # Stock contributions table
            contribs = viz_data.get("stock_contributions", [])
            best_t   = viz_data.get("best_performer", {}).get("ticker")
            worst_t  = viz_data.get("worst_performer", {}).get("ticker")
            if contribs:
                st.markdown(f"<div style='height:14px'></div><div style='font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:#ffffff;margin-bottom:8px'>Stock Contributions</div>", unsafe_allow_html=True)
                rows_html = ""
                for row in contribs:
                    is_best  = row["ticker"] == best_t
                    is_worst = row["ticker"] == worst_t
                    row_bg   = SUCCESS_BG if is_best else (DANGER_BG if is_worst else GRAY_BG)
                    ret_fg   = SUCCESS if row["return_pct"] >= 0 else DANGER
                    cnt_fg   = SUCCESS if row["contribution_pct"] >= 0 else DANGER
                    ret_sign = "+" if row["return_pct"] >= 0 else ""
                    cnt_sign = "+" if row["contribution_pct"] >= 0 else ""
                    badge = ""
                    if is_best:
                        badge = f'<span style="font-size:10px;background:{SUCCESS};color:white;padding:1px 6px;border-radius:20px;margin-left:6px">Best</span>'
                    elif is_worst:
                        badge = f'<span style="font-size:10px;background:{DANGER};color:white;padding:1px 6px;border-radius:20px;margin-left:6px">Worst</span>'
                    rows_html += (
                        f'<div style="display:flex;align-items:center;justify-content:space-between;'
                        f'background:{row_bg};border-radius:8px;padding:8px 12px;margin-bottom:4px;'
                        f'border:1px solid {BORDER}">'
                        f'<div style="display:flex;align-items:center;gap:4px">'
                        f'<span style="font-size:13px;font-weight:600;color:{TEXT}">{html_lib.escape(row["ticker"])}</span>'
                        f'{badge}'
                        f'<span style="font-size:11px;color:{MUTED};margin-left:4px">{row["weight_pct"]:.1f}%</span>'
                        f'</div>'
                        f'<div style="display:flex;gap:20px;font-size:12.5px">'
                        f'<span style="color:{ret_fg};font-weight:600">{ret_sign}{row["return_pct"]:.2f}%</span>'
                        f'<span style="color:{cnt_fg};font-weight:500">{cnt_sign}{row["contribution_pct"]:.2f}% contrib</span>'
                        f'</div>'
                        f'</div>'
                    )
                st.markdown(rows_html, unsafe_allow_html=True)

    # ── VaR metrics ───────────────────────────────────────────────────────────
    elif viz_type == "metrics" and "var_95_pct" in viz_data:
        period = viz_data.get("period", "")
        st.markdown(
            f"""<div style="font-size:11px;font-weight:600;text-transform:uppercase;
              letter-spacing:.05em;color:#ffffff;margin-bottom:12px">
              Value at Risk (Historical Method)  ·  {period}
            </div>""",
            unsafe_allow_html=True,
        )

        def _var_card(col, label, value, sub, bg, fg):
            col.markdown(
                f"""<div style="background:{bg};border-radius:12px;padding:16px 12px;
                  text-align:center;border:1px solid {fg}22">
                  <div style="font-size:10.5px;color:{fg};font-weight:600;
                    text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px">
                    {label}
                  </div>
                  <div style="font-size:28px;font-weight:700;color:{fg};line-height:1">
                    {value}%
                  </div>
                  <div style="font-size:11px;color:{fg};margin-top:6px;opacity:.75">
                    {sub}
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

        c1, c2, c3, c4 = st.columns(4)
        _var_card(c1, "VaR 95%",  f"{viz_data['var_95_pct']:.2f}",  "1-in-20 bad day",   DANGER_BG,        DANGER)
        _var_card(c2, "VaR 99%",  f"{viz_data['var_99_pct']:.2f}",  "1-in-100 bad day",  "#fecdd3",        "#9f1239")
        _var_card(c3, "CVaR 95%", f"{viz_data['cvar_95_pct']:.2f}", "Avg of worst days", "#fee2e2",        "#b91c1c")
        _var_card(c4,
            f"Worst Day",
            f"{viz_data['worst_day_loss_pct']:.2f}",
            viz_data.get("worst_day_date", ""),
            "#fef2f2", "#dc2626",
        )
        var95 = abs(viz_data["var_95_pct"])
        var99 = abs(viz_data["var_99_pct"])
        st.info(
            f"On a typical bad day (1-in-20), expect to lose ~**{var95:.1f}%**. "
            f"In extreme scenarios (1-in-100), losses could reach **{var99:.1f}%**. "
            f"CVaR shows the average loss beyond the VaR threshold — the 'average of the worst days'."
        )

    # ── Sharpe metrics ────────────────────────────────────────────────────────
    elif viz_type == "metrics":
        sharpe = viz_data.get("sharpe_ratio", 0)
        period = viz_data.get("period", "")

        if sharpe > 1.0:
            s_fg, s_bg, s_label, interp = (
                SUCCESS, SUCCESS_BG, "Good",
                "Strong risk-adjusted returns — the portfolio is well-compensated for the volatility it carries.",
            )
        elif sharpe >= 0.5:
            s_fg, s_bg, s_label, interp = (
                WARNING, WARNING_BG, "Moderate",
                "Acceptable risk-adjusted returns. There may be room to improve the return-to-risk profile.",
            )
        else:
            s_fg, s_bg, s_label, interp = (
                DANGER, DANGER_BG, "Poor",
                "Weak risk-adjusted returns — the portfolio may not adequately compensate for the risk taken.",
            )

        st.markdown(
            f"""<div style="text-align:center;padding:28px 16px;background:{s_bg};
              border-radius:14px;margin-bottom:20px;border:1px solid {s_fg}22">
              <div style="font-size:11px;color:{s_fg};font-weight:600;
                letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px">
                Sharpe Ratio  ·  {period}
              </div>
              <div style="font-size:56px;font-weight:700;color:{s_fg};line-height:1;
                letter-spacing:-1px">
                {sharpe:.2f}
              </div>
              <div style="display:inline-block;margin-top:10px;background:{s_fg};
                color:white;font-size:12px;font-weight:600;padding:3px 12px;
                border-radius:20px;letter-spacing:.03em">
                {s_label}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Annualized Return",     f"{viz_data.get('annualized_return_pct', 0):+.2f}%")
        c2.metric("Annualized Volatility", f"{viz_data.get('annualized_volatility_pct', 0):.2f}%")
        c3.metric("Risk-Free Rate",        f"{viz_data.get('risk_free_rate_pct', 6.0):.1f}%")
        st.info(interp)

    # ── Drawdown chart ────────────────────────────────────────────────────────
    elif viz_type == "drawdown_chart":
        series = viz_data.get("drawdown_series", [])
        if series:
            daily      = pd.DataFrame(series)
            max_dd     = viz_data.get("max_drawdown_pct", 0)
            trough_dt  = viz_data.get("trough_date", "")
            period     = viz_data.get("period", "")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily["date"], y=daily["drawdown"],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(220,38,38,0.08)",
                line=dict(color=DANGER, width=1.5),
                name="Drawdown (%)",
                hovertemplate="%{x}<br><b>%{y:.2f}%</b><extra></extra>",
            ))
            if trough_dt:
                fig.add_trace(go.Scatter(
                    x=[trough_dt], y=[max_dd],
                    mode="markers+text",
                    marker=dict(color=DANGER, size=10, symbol="circle",
                                line=dict(color="white", width=2)),
                    text=[f"  {max_dd:.1f}%"],
                    textposition="middle right",
                    textfont=dict(color=DANGER, size=12, family="Inter, sans-serif"),
                    name="Max Drawdown",
                    showlegend=False,
                    hovertemplate=f"Max Drawdown<br><b>{max_dd:.2f}%</b><extra></extra>",
                ))
            layout = _chart_layout("Portfolio Drawdown", period)
            layout["yaxis"]["ticksuffix"] = "%"
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(
                f"""<div style="background:{DANGER_BG};border-radius:10px;
                  padding:14px 10px;text-align:center;border:1px solid {DANGER}22">
                  <div style="font-size:10.5px;color:{DANGER};font-weight:600;
                    text-transform:uppercase;letter-spacing:.04em">Max Drawdown</div>
                  <div style="font-size:26px;font-weight:700;color:{DANGER};margin-top:4px">
                    {max_dd:.1f}%
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )
            c2.metric("Peak Date",   viz_data.get("peak_date", "—"))
            c3.metric("Trough Date", trough_dt or "—")
            if viz_data.get("currently_in_drawdown"):
                c4.metric("Status", "In drawdown")
                st.warning("Portfolio is currently still in a drawdown phase.")
            else:
                c4.metric("Recovery Date", viz_data.get("recovery_date") or "—")
                st.success("Portfolio has fully recovered from the maximum drawdown.")

    # ── Benchmark chart ───────────────────────────────────────────────────────
    elif viz_type == "benchmark_chart":
        port_series = pd.DataFrame(viz_data.get("portfolio_cumulative", []))
        bm_series   = pd.DataFrame(viz_data.get("benchmark_cumulative", []))
        bm_name     = viz_data.get("benchmark_name", "Benchmark")
        period      = viz_data.get("period", "")

        if not port_series.empty and not bm_series.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=port_series["date"], y=port_series["value"],
                mode="lines", name="My Portfolio",
                line=dict(color=PRIMARY, width=2.5),
                hovertemplate="%{x}<br><b>%{y:.1f}</b><extra>Portfolio</extra>",
            ))
            fig.add_trace(go.Scatter(
                x=bm_series["date"], y=bm_series["value"],
                mode="lines", name=bm_name,
                line=dict(color=WARNING, width=2, dash="dot"),
                hovertemplate="%{x}<br><b>%{y:.1f}</b><extra>" + bm_name + "</extra>",
            ))
            fig.add_hline(y=100, line_dash="dash", line_color=BORDER, line_width=1)
            layout = _chart_layout(f"Portfolio vs {bm_name}", period)
            layout["legend"] = dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                font=dict(size=11, family="Inter, sans-serif", color=TEXT),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=BORDER, borderwidth=1,
            )
            layout["yaxis_title"] = "Growth of ₹100"
            fig.update_layout(**layout)
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
                st.success(f"Your portfolio **outperformed** {bm_name} by **{alpha:.2f}%** over this period.")
            else:
                st.warning(f"Your portfolio **underperformed** {bm_name} by **{abs(alpha):.2f}%** over this period.")

    # ── Correlation heatmap ───────────────────────────────────────────────────
    elif viz_type == "correlation_heatmap":
        matrix  = viz_data.get("matrix", [])
        tickers = viz_data.get("tickers", [])
        if matrix and tickers:
            annotations = [[f"{v:.2f}" for v in row] for row in matrix]
            fig = go.Figure(go.Heatmap(
                z=matrix, x=tickers, y=tickers,
                colorscale="RdBu_r", zmin=-1, zmax=1,
                text=annotations,
                texttemplate="%{text}",
                textfont=dict(size=11, family="Inter, sans-serif", color="white"),
                colorbar=dict(
                    title=dict(text="r", font=dict(size=11, family="Inter, sans-serif", color=TEXT)),
                    thickness=12, tickfont=dict(size=10, color=TEXT),
                ),
                hovertemplate="%{y} & %{x}<br>r = <b>%{z:.3f}</b><extra></extra>",
            ))
            layout = _chart_layout("Asset Correlation Matrix", viz_data.get("period", ""))
            layout["margin"] = dict(t=52, b=80, l=100, r=8)
            layout["xaxis"] = dict(tickangle=-35, side="bottom",
                                   tickfont=dict(size=11, color=MUTED))
            layout["yaxis"] = dict(autorange="reversed",
                                   tickfont=dict(size=11, color=MUTED))
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

            high_corr = viz_data.get("high_correlations", [])
            low_corr  = viz_data.get("low_correlations", [])
            if high_corr:
                pairs = ", ".join(
                    f"**{p['assets'][0]}** & **{p['assets'][1]}** ({p['correlation']:.2f})"
                    for p in high_corr[:3]
                )
                st.warning(f"High-correlation pairs (limited diversification): {pairs}")
            if low_corr:
                pairs = ", ".join(
                    f"**{p['assets'][0]}** & **{p['assets'][1]}** ({p['correlation']:.2f})"
                    for p in low_corr[:3]
                )
                st.success(f"Low-correlation pairs (good diversifiers): {pairs}")
            if not high_corr and not low_corr:
                st.info("All pairs have moderate correlation (0.30–0.70), suggesting reasonable diversification.")

    # ── Concentration pie ─────────────────────────────────────────────────────
    elif viz_type == "concentration_pie":
        breakdown      = viz_data.get("breakdown", [])
        breakdown_type = viz_data.get("breakdown_type", "sector")
        if breakdown:
            labels = [b["label"] for b in breakdown]
            values = [b["weight_pct"] for b in breakdown]
            title_map = {
                "sector":      "Sector Concentration",
                "asset_class": "Asset Class Breakdown",
                "factor":      "Market-Cap Factor Exposure",
            }
            fig = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.4,
                textinfo="percent",
                textfont=dict(size=12, family="Inter, sans-serif", color="white"),
                textposition="inside",
                marker=dict(colors=CHART_COLORS, line=dict(color=SURFACE, width=2.5)),
                hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
                pull=[0.04] + [0] * (len(labels) - 1),
            ))
            layout = _chart_layout(title_map.get(breakdown_type, "Concentration"))
            layout["legend"] = dict(
                orientation="v", yanchor="middle", y=0.5,
                xanchor="left", x=1.02,
                font=dict(size=11, family="Inter, sans-serif", color=TEXT),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=BORDER, borderwidth=1,
            )
            layout["margin"] = dict(t=52, b=32, l=8, r=100)
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

            dominant     = viz_data.get("dominant", {})
            label_word   = breakdown_type.replace("_", " ")
            if viz_data.get("concentration_warning"):
                st.warning(
                    f"Concentration alert: **{dominant.get('label')}** accounts for "
                    f"**{dominant.get('weight_pct', 0):.1f}%** of your portfolio — "
                    f"consider diversifying across {label_word}s."
                )
            else:
                st.success(
                    f"Well-diversified across {label_word}s. Largest exposure "
                    f"(**{dominant.get('label')}**) is {dominant.get('weight_pct', 0):.1f}%."
                )
            col_name = {"sector": "Sector", "asset_class": "Asset Class", "factor": "Factor"}.get(
                breakdown_type, "Category"
            )
            df = pd.DataFrame(breakdown)
            df.columns = [col_name, "Weight (%)"]
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Portfolio pie ─────────────────────────────────────────────────────────
    elif viz_type == "portfolio_pie":
        holdings = viz_data.get("holdings", [])
        if holdings:
            names = [h["name"] for h in holdings]
            wts   = [h["weight_pct"] for h in holdings]

            fig = go.Figure(go.Pie(
                labels=names, values=wts, hole=0.4,
                textinfo="percent",
                textfont=dict(size=12, family="Inter, sans-serif", color="white"),
                textposition="inside",
                marker=dict(colors=CHART_COLORS, line=dict(color=SURFACE, width=2.5)),
                hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
            ))
            layout = _chart_layout("Portfolio Composition")
            layout["legend"] = dict(
                orientation="v", yanchor="middle", y=0.5,
                xanchor="left", x=1.02,
                font=dict(size=11, family="Inter, sans-serif", color=TEXT),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=BORDER, borderwidth=1,
            )
            layout["margin"] = dict(t=52, b=32, l=8, r=120)
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            largest = viz_data.get("largest_holding", {})
            c1.markdown(
                f"""<div style="background:{PRIMARY_BG};border:1px solid #bfdbfe;
                  border-radius:10px;padding:14px 16px;text-align:center">
                  <div style="font-size:11px;font-weight:600;text-transform:uppercase;
                    letter-spacing:.05em;color:{PRIMARY};margin-bottom:4px">Total Holdings</div>
                  <div style="font-size:26px;font-weight:700;color:{PRIMARY}">
                    {viz_data.get("total_holdings", 0)}
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )
            c2.markdown(
                f"""<div style="background:{PRIMARY_BG};border:1px solid #bfdbfe;
                  border-radius:10px;padding:14px 16px;text-align:center">
                  <div style="font-size:11px;font-weight:600;text-transform:uppercase;
                    letter-spacing:.05em;color:{PRIMARY};margin-bottom:4px">Largest Position</div>
                  <div style="font-size:26px;font-weight:700;color:{PRIMARY}">
                    {largest.get("weight_pct", 0):.1f}%
                  </div>
                  <div style="font-size:11.5px;color:{PRIMARY};opacity:.75;margin-top:3px">
                    {html_lib.escape(largest.get("name", ""))}
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )
            df = pd.DataFrame(holdings)[["name", "ticker", "weight_pct"]]
            df.columns = ["Company", "Ticker", "Weight (%)"]
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Empty state ───────────────────────────────────────────────────────────
    else:
        st.markdown(
            f"""<div style="display:flex;flex-direction:column;align-items:center;
              justify-content:center;height:340px;background:{GRAY_BG};
              border:2px dashed {BORDER};border-radius:16px;margin-top:8px">
              <div style="font-size:40px;margin-bottom:14px;opacity:0.45">📈</div>
              <div style="font-size:15px;font-weight:600;color:{MUTED};margin-bottom:6px">
                No analysis yet
              </div>
              <div style="font-size:13px;color:{MUTED_LT};text-align:center;
                max-width:240px;line-height:1.6">
                Ask a question about your portfolio to see charts and metrics here
              </div>
            </div>""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Shared API call helper
# ---------------------------------------------------------------------------
def _call_api(portfolio: dict, prompt: str):
    """Call /analyze, update session state, return True on success."""
    p = st.session_state.current_portfolio or portfolio
    payload = {
        "tickers": p["tickers"],
        "weights": p["weights"],
        "period":  p["period"],
        "question": prompt,
        "accumulated_analysis": st.session_state.accumulated_analysis,
    }
    try:
        resp = requests.post("http://localhost:8000/analyze", json=payload, timeout=120)
        if resp.status_code == 200:
            result = resp.json()
            st.session_state.messages.append({"role": "assistant", "content": result["response"]})
            viz = result.get("visualization", {})
            st.session_state.current_visualization_type = viz.get("type")
            st.session_state.current_visualization_data = viz.get("data", {})
            st.session_state.canvas_updated = True
            updates = result.get("analysis_updates", {})
            if updates:
                st.session_state.accumulated_analysis.update(updates)
            portfolio_update = result.get("portfolio_update")
            if portfolio_update:
                st.session_state.current_portfolio = {
                    "tickers": portfolio_update["tickers"],
                    "weights": portfolio_update["weights"],
                    "period":  portfolio_update.get("period", p["period"]),
                }
                st.session_state.portfolio_is_modified = True
                st.session_state.modification_history.append({
                    "summary": portfolio_update.get("changes_summary", ""),
                    "tickers": portfolio_update["tickers"],
                    "weights": portfolio_update["weights"],
                })
                st.session_state.current_suggestions = [
                    {"text": "Check returns", "query": "How did my portfolio perform over the last year?", "action": "_mod_compare", "priority": "high"},
                    {"text": "Check risk profile", "query": "What is the Sharpe ratio of my portfolio?", "action": "_mod_risk", "priority": "medium"},
                    {"text": "Show composition", "query": "Show me my portfolio composition", "action": "_mod_composition", "priority": "low"},
                ]
            else:
                explored_set = set(st.session_state.explored_suggestions)
                st.session_state.current_suggestions = [
                    s for s in result.get("suggestions", [])
                    if s["action"] not in explored_set
                ]
            return True
        else:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {detail}"})
            st.session_state.current_suggestions = []
    except requests.exceptions.ConnectionError:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Could not connect to the backend. Make sure it's running on port 8000.",
        })
        st.session_state.current_suggestions = []
    except requests.exceptions.Timeout:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "The request timed out. This can happen with large portfolios — please try again.",
        })
        st.session_state.current_suggestions = []
    return False


# ===========================================================================
# STEP 1 — Upload page
# ===========================================================================
if not st.session_state.portfolio_uploaded:
    _inject_css()

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown(
        f"""<div style="text-align:center;padding:48px 0 36px">
          <div style="display:inline-flex;align-items:center;gap:8px;
            background:{PRIMARY_BG};border:1px solid #bfdbfe;border-radius:50px;
            padding:5px 14px 5px 8px;margin-bottom:20px">
            <div style="width:26px;height:26px;background:{PRIMARY};border-radius:50%;
              display:flex;align-items:center;justify-content:center;
              color:white;font-size:12px;font-weight:700">K</div>
            <span style="color:{PRIMARY};font-weight:600;font-size:13px;
              letter-spacing:.02em">KALPI</span>
          </div>
          <h1 style="font-size:38px;font-weight:700;color:#ffffff;margin:0 0 10px;
            letter-spacing:-0.5px;line-height:1.15">Portfolio Analyzer</h1>
          <p style="font-size:16px;color:rgba(255,255,255,0.65);margin:0;font-weight:400">
            AI-Powered Portfolio Analysis · Instant Charts · Smart Insights
          </p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Two-column layout: instructions + upload ──────────────────────────────
    _, content_col, _ = st.columns([0.4, 3, 0.4])
    with content_col:
        left_col, right_col = st.columns([1, 1], gap="large")

        # ── Left: How it works + format ───────────────────────────────────────
        with left_col:
            st.markdown(
                f"""<div style="background:{SURFACE};border:1px solid {BORDER};
                  border-radius:14px;padding:24px 22px;height:100%">
                  <div style="font-size:13px;font-weight:700;color:{TEXT};
                    margin-bottom:16px;letter-spacing:.01em">How it works</div>
                  <div style="display:flex;flex-direction:column;gap:14px">
                    <div style="display:flex;align-items:flex-start;gap:10px">
                      <div style="min-width:24px;height:24px;background:{PRIMARY_BG};
                        border-radius:50%;display:flex;align-items:center;
                        justify-content:center;font-size:11px;font-weight:700;
                        color:{PRIMARY}">1</div>
                      <div>
                        <div style="font-size:13px;font-weight:600;color:{TEXT}">
                          Upload your portfolio CSV</div>
                        <div style="font-size:12px;color:{MUTED};margin-top:2px">
                          Two columns: Ticker and Weight (decimal, summing to 1.0)</div>
                      </div>
                    </div>
                    <div style="display:flex;align-items:flex-start;gap:10px">
                      <div style="min-width:24px;height:24px;background:{PRIMARY_BG};
                        border-radius:50%;display:flex;align-items:center;
                        justify-content:center;font-size:11px;font-weight:700;
                        color:{PRIMARY}">2</div>
                      <div>
                        <div style="font-size:13px;font-weight:600;color:{TEXT}">
                          Ask in plain English</div>
                        <div style="font-size:12px;color:{MUTED};margin-top:2px">
                          "What's my Sharpe ratio?" · "Compare to Nifty 50" ·
                          "Show drawdown"</div>
                      </div>
                    </div>
                    <div style="display:flex;align-items:flex-start;gap:10px">
                      <div style="min-width:24px;height:24px;background:{PRIMARY_BG};
                        border-radius:50%;display:flex;align-items:center;
                        justify-content:center;font-size:11px;font-weight:700;
                        color:{PRIMARY}">3</div>
                      <div>
                        <div style="font-size:13px;font-weight:600;color:{TEXT}">
                          Get instant analysis</div>
                        <div style="font-size:12px;color:{MUTED};margin-top:2px">
                          Interactive charts, risk metrics, and smart suggestions</div>
                      </div>
                    </div>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # CSV format example
            st.markdown(
                f"""<div style="font-size:12px;font-weight:600;color:{MUTED};
                  text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px">
                  CSV Format Example
                </div>""",
                unsafe_allow_html=True,
            )
            sample_df = pd.DataFrame({
                "Ticker": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"],
                "Weight": [0.35, 0.25, 0.20, 0.20],
            })
            st.dataframe(sample_df, use_container_width=True, hide_index=True, height=178)

            sample_csv = (
                "Ticker,Weight\n"
                "RELIANCE.NS,0.35\n"
                "TCS.NS,0.25\n"
                "INFY.NS,0.20\n"
                "HDFCBANK.NS,0.20\n"
            )
            st.download_button(
                "⬇  Download Sample CSV",
                data=sample_csv,
                file_name="sample_portfolio.csv",
                mime="text/csv",
                use_container_width=True,
                type="secondary",
            )

        # ── Right: Upload form ────────────────────────────────────────────────
        with right_col:
            st.markdown(
                f"""<div style="background:{SURFACE};border:1px solid {BORDER};
                  border-radius:14px;padding:24px 22px">
                  <div style="font-size:13px;font-weight:700;color:{TEXT};
                    margin-bottom:4px;letter-spacing:.01em">Upload Portfolio</div>
                  <div style="font-size:12px;color:{MUTED};margin-bottom:16px">
                    CSV file with <code style="background:#f1f5f9;padding:1px 5px;
                    border-radius:4px;font-size:11px">Ticker</code> and
                    <code style="background:#f1f5f9;padding:1px 5px;border-radius:4px;
                    font-size:11px">Weight</code> columns. Weights must sum to 1.0.
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

            uploaded = st.file_uploader(
                "Portfolio CSV",
                type="csv",
                label_visibility="collapsed",
            )

            if uploaded is not None:
                st.markdown(
                    f"""<div style="display:flex;align-items:center;gap:8px;
                      background:{SUCCESS_BG};border:1px solid #6ee7b7;
                      border-radius:8px;padding:8px 12px;margin:8px 0;font-size:13px">
                      <span style="color:{SUCCESS};font-size:16px">✓</span>
                      <span style="color:{SUCCESS};font-weight:500">{uploaded.name} ready</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            if st.button("Analyze Portfolio →", type="primary", use_container_width=True):
                if uploaded is None:
                    st.error("Please upload a CSV file first.")
                else:
                    df = pd.read_csv(io.StringIO(uploaded.getvalue().decode("utf-8")))
                    if "Ticker" not in df.columns or "Weight" not in df.columns:
                        st.error("CSV must have 'Ticker' and 'Weight' columns.")
                    else:
                        df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
                        if df["Weight"].isna().any():
                            st.error("All weights must be numeric values.")
                        elif abs(df["Weight"].sum() - 1.0) > 0.01:
                            st.error(
                                f"Weights must sum to 1.0 — got {df['Weight'].sum():.4f}. "
                                f"Adjust your weights and re-upload."
                            )
                        else:
                            tickers = df["Ticker"].tolist()
                            weights = df["Weight"].tolist()
                            _pdata = {"tickers": tickers, "weights": weights, "period": "1y"}
                            st.session_state.portfolio          = _pdata
                            st.session_state.original_portfolio = dict(_pdata)
                            st.session_state.current_portfolio  = dict(_pdata)
                            st.session_state.modification_history = []
                            st.session_state.portfolio_is_modified = False
                            with st.spinner("Loading portfolio and running initial analysis…"):
                                try:
                                    init_resp = requests.post(
                                        "http://localhost:8000/analyze",
                                        json={
                                            "tickers": tickers,
                                            "weights": weights,
                                            "period": "1y",
                                            "question": "",
                                            "is_initial": True,
                                            "accumulated_analysis": {},
                                        },
                                        timeout=120,
                                    )
                                    if init_resp.status_code == 200:
                                        r = init_resp.json()
                                        viz = r.get("visualization", {})
                                        st.session_state.current_visualization_type = viz.get("type")
                                        st.session_state.current_visualization_data = viz.get("data", {})
                                        st.session_state.accumulated_analysis = r.get("analysis_updates", {})
                                        st.session_state.current_suggestions  = r.get("suggestions", [])
                                        st.session_state.canvas_updated = True
                                except Exception:
                                    pass
                            st.session_state.portfolio_uploaded = True
                            st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        f"""<div style="text-align:center;padding:32px 0 8px;margin-top:24px;
          border-top:1px solid {BORDER}">
          <span style="font-size:12px;color:{MUTED_LT}">
            Built with <b style="color:{MUTED}">LangGraph</b> ·
            <b style="color:{MUTED}">FastAPI</b> ·
            <b style="color:{MUTED}">Streamlit</b>
          </span>
        </div>""",
        unsafe_allow_html=True,
    )


# ===========================================================================
# STEP 2 — Chat + Canvas
# ===========================================================================
else:
    _inject_css()
    portfolio = st.session_state.portfolio

    # ── Handle auto-submitted suggestion ─────────────────────────────────────
    if "pending_suggestion" in st.session_state:
        prompt = st.session_state.pop("pending_suggestion")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Analyzing your portfolio…"):
            _call_api(portfolio, prompt)
        st.rerun()

    # ── Header ───────────────────────────────────────────────────────────────
    hdr_left, hdr_right = st.columns([5, 1])
    with hdr_left:
        n = len(portfolio["tickers"])
        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:12px;padding:6px 0 14px">
              <div style="width:36px;height:36px;background:{PRIMARY};border-radius:10px;
                display:flex;align-items:center;justify-content:center;color:white;
                font-size:16px;font-weight:700;flex-shrink:0;
                box-shadow:0 2px 8px rgba(37,99,235,0.3)">K</div>
              <div>
                <div style="font-weight:700;font-size:16px;color:#ffffff;line-height:1.2">
                  Kalpi Portfolio Analyzer</div>
                <div style="font-size:11.5px;color:rgba(255,255,255,0.55)">
                  AI-Powered Portfolio Analysis</div>
              </div>
              <div style="margin-left:12px;display:inline-flex;align-items:center;
                gap:6px;background:{SUCCESS_BG};border:1px solid #86efac;
                border-radius:20px;padding:4px 10px 4px 8px">
                <div class="dot-active" style="width:7px;height:7px;background:#22c55e;
                  border-radius:50%;flex-shrink:0"></div>
                <span style="font-size:11px;color:#065f46;font-weight:500;
                  white-space:nowrap">{n} holdings loaded</span>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with hdr_right:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("↺  New Analysis", type="secondary", use_container_width=True):
            for k in ("portfolio", "original_portfolio", "current_portfolio",
                      "messages", "current_visualization_type",
                      "current_visualization_data", "current_suggestions",
                      "explored_suggestions", "accumulated_analysis",
                      "modification_history"):
                default_val = (
                    None if k in ("portfolio", "original_portfolio", "current_portfolio",
                                  "current_visualization_type") else
                    [] if k in ("messages", "current_suggestions", "explored_suggestions",
                                "modification_history") else {}
                )
                st.session_state[k] = default_val
            st.session_state.portfolio_uploaded    = False
            st.session_state.canvas_updated        = False
            st.session_state.portfolio_is_modified = False
            st.rerun()

    # Holdings bar
    max_show   = 7
    _is_mod    = st.session_state.portfolio_is_modified
    _orig_port = st.session_state.original_portfolio or portfolio
    _curr_port = st.session_state.current_portfolio  or portfolio

    if _is_mod:
        orig_map    = dict(zip(_orig_port["tickers"], _orig_port["weights"]))
        curr_map    = dict(zip(_curr_port["tickers"], _curr_port["weights"]))
        orig_set    = set(_orig_port["tickers"])
        curr_set    = set(_curr_port["tickers"])
        added_set   = curr_set - orig_set
        removed_set = orig_set - curr_set
        changed_set = {t for t in (orig_set & curr_set) if abs(orig_map[t] - curr_map.get(t, 0)) > 0.001}

        pills = ""
        for t, w in zip(_curr_port["tickers"][:max_show], _curr_port["weights"][:max_show]):
            if t in added_set:
                pill_bg = "#dcfce7"; pill_border = "#86efac"; pill_fg = SUCCESS
                pill_right = f'<span style="font-size:9.5px;background:{SUCCESS};color:white;padding:1px 5px;border-radius:10px;margin-left:2px">NEW</span>'
            elif t in changed_set:
                old_w = orig_map[t]
                pill_bg = "#eff6ff"; pill_border = "#bfdbfe"; pill_fg = PRIMARY
                pill_right = f'<span style="font-size:9.5px;color:{PRIMARY};margin-left:2px">{old_w*100:.0f}→{w*100:.0f}%</span>'
            else:
                pill_bg = GRAY_BG; pill_border = BORDER; pill_fg = TEXT
                pill_right = f'<span style="color:{MUTED_LT}">{w*100:.0f}%</span>'
            pills += (
                f'<span style="display:inline-flex;align-items:center;gap:3px;'
                f'background:{pill_bg};border:1px solid {pill_border};border-radius:20px;'
                f'padding:2px 8px 2px 6px;font-size:11.5px;white-space:nowrap">'
                f'<span style="font-weight:600;color:{pill_fg}">{html_lib.escape(t)}</span>'
                f'{pill_right}'
                f'</span>'
            )
        for t in list(removed_set)[:3]:
            pills += (
                f'<span style="display:inline-flex;align-items:center;gap:3px;'
                f'background:{DANGER_BG};border:1px solid #fca5a5;border-radius:20px;'
                f'padding:2px 8px 2px 6px;font-size:11.5px;white-space:nowrap;opacity:0.75">'
                f'<span style="font-weight:600;color:{DANGER};text-decoration:line-through">{html_lib.escape(t)}</span>'
                f'<span style="font-size:9.5px;color:{DANGER};margin-left:2px">removed</span>'
                f'</span>'
            )
        if len(_curr_port["tickers"]) > max_show:
            pills += f'<span style="font-size:11.5px;color:{MUTED_LT};padding:0 4px">+{len(_curr_port["tickers"])-max_show} more</span>'

        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;
              padding:10px 14px;background:{SURFACE};border:1px solid #bfdbfe;
              border-radius:10px;margin-bottom:8px">
              <span style="font-size:10.5px;font-weight:600;text-transform:uppercase;
                letter-spacing:.05em;color:{PRIMARY};margin-right:2px">Holdings</span>
              <span style="font-size:9.5px;background:{WARNING_BG};color:{WARNING};
                padding:1px 7px;border-radius:10px;font-weight:600;
                border:1px solid {WARNING}44;margin-right:4px">MODIFIED</span>
              {pills}
            </div>""",
            unsafe_allow_html=True,
        )
        last_summary = (
            st.session_state.modification_history[-1]["summary"]
            if st.session_state.modification_history else ""
        )
        _chg_col, _rev_col = st.columns([5, 1])
        with _chg_col:
            if last_summary:
                st.markdown(
                    f'<div style="font-size:11.5px;color:{MUTED};padding:2px 0 12px">'
                    f'📝 {html_lib.escape(last_summary)}</div>',
                    unsafe_allow_html=True,
                )
        with _rev_col:
            if st.button("↩ Revert", key="revert_btn", type="secondary", use_container_width=True):
                st.session_state.current_portfolio     = dict(_orig_port)
                st.session_state.portfolio_is_modified = False
                st.session_state.modification_history  = []
                st.session_state.current_suggestions   = []
                st.rerun()
    else:
        tickers = portfolio["tickers"]
        weights = portfolio["weights"]
        pills = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:4px;'
            f'background:{GRAY_BG};border:1px solid {BORDER};border-radius:20px;'
            f'padding:2px 8px 2px 6px;font-size:11.5px;white-space:nowrap">'
            f'<span style="font-weight:600;color:{TEXT}">{t}</span>'
            f'<span style="color:{MUTED_LT}">{w*100:.0f}%</span>'
            f'</span>'
            for t, w in zip(tickers[:max_show], weights[:max_show])
        )
        if len(tickers) > max_show:
            pills += (
                f'<span style="font-size:11.5px;color:{MUTED_LT};padding:0 4px">'
                f'+{len(tickers)-max_show} more</span>'
            )
        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;
              padding:10px 14px;background:{SURFACE};border:1px solid {BORDER};
              border-radius:10px;margin-bottom:18px">
              <span style="font-size:10.5px;font-weight:600;text-transform:uppercase;
                letter-spacing:.05em;color:{MUTED_LT};margin-right:4px">Holdings</span>
              {pills}
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Smart suggestions (below holdings, above chat+canvas) ─────────────────
    explored_set = set(st.session_state.explored_suggestions)
    active_sug = [
        s for s in st.session_state.current_suggestions
        if s["action"] not in explored_set
    ]
    if active_sug:
        st.markdown(
            f"<div style='font-size:11px;color:rgba(255,255,255,0.45);font-weight:500;"
            f"letter-spacing:.03em;margin-bottom:6px'>💡 Suggested Questions</div>",
            unsafe_allow_html=True,
        )
        pill_cols = st.columns(len(active_sug))
        for idx, s in enumerate(active_sug):
            priority_cls = f"sug-pill sug-pill-{s['priority']}"
            with pill_cols[idx]:
                st.markdown(f"<div class='{priority_cls}'>", unsafe_allow_html=True)
                if st.button(
                    s["text"],
                    key=f"sug_{idx}_{s['action']}",
                    use_container_width=True,
                ):
                    st.session_state.explored_suggestions.append(s["action"])
                    st.session_state.pending_suggestion = s.get("query", s["text"])
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Two-column layout ─────────────────────────────────────────────────────
    chat_col, canvas_col = st.columns([1, 2], gap="large")

    # ── Left: Chat ────────────────────────────────────────────────────────────
    with chat_col:
        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:8px;
              margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid {BORDER}">
              <span style="font-weight:700;font-size:14px;color:#ffffff">Chat</span>
              <span style="font-size:11.5px;color:rgba(255,255,255,0.55);margin-left:auto">
                Ask anything about your portfolio
              </span>
            </div>""",
            unsafe_allow_html=True,
        )

        # Scrollable message area
        with st.container(height=440):
            if not st.session_state.messages:
                st.markdown(
                    f"""<div style="display:flex;flex-direction:column;align-items:center;
                      justify-content:center;height:300px;text-align:center;padding:24px">
                      <div style="font-size:28px;margin-bottom:12px;opacity:0.4">💬</div>
                      <div style="font-size:13.5px;font-weight:600;color:{MUTED};
                        margin-bottom:6px">Start the conversation</div>
                      <div style="font-size:12.5px;color:{MUTED_LT};line-height:1.6;
                        max-width:200px">
                        Ask about returns, risk, Sharpe ratio, benchmarks, and more
                      </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            for msg in st.session_state.messages:
                chat_bubble(msg["role"], msg["content"])

        # Auto-scroll to bottom
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
                "message",
                label_visibility="collapsed",
                placeholder="Ask about your portfolio…",
            )
            submitted = st.form_submit_button("Send  →", use_container_width=True)

        if submitted and user_input.strip():
            prompt = user_input.strip()
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("Analyzing your portfolio…"):
                _call_api(portfolio, prompt)
            st.rerun()

    # ── Right: Canvas ─────────────────────────────────────────────────────────
    with canvas_col:
        # Canvas header
        _canvas_badges = ""
        if st.session_state.canvas_updated:
            _canvas_badges += (
                f'<span class="badge-pop" style="background:{SUCCESS_BG};color:{SUCCESS};'
                f'font-size:10.5px;padding:3px 8px;border-radius:20px;font-weight:600;'
                f'letter-spacing:.03em;border:1px solid #86efac">✓ UPDATED</span>'
            )
            st.session_state.canvas_updated = False
        if st.session_state.portfolio_is_modified:
            _canvas_badges += (
                f'<span style="background:{WARNING_BG};color:{WARNING};'
                f'font-size:10.5px;padding:3px 8px;border-radius:20px;font-weight:600;'
                f'letter-spacing:.03em;border:1px solid {WARNING}44;margin-left:4px">⚡ MODIFIED</span>'
            )
        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:8px;
              margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid {BORDER}">
              <span style="font-weight:700;font-size:14px;color:#ffffff">
                Portfolio Analysis Canvas</span>
              {_canvas_badges}
            </div>""",
            unsafe_allow_html=True,
        )

        render_canvas(
            st.session_state.current_visualization_type,
            st.session_state.current_visualization_data,
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        f"""<div style="text-align:center;padding:24px 0 4px;margin-top:28px;
          border-top:1px solid {BORDER}">
          <span style="font-size:12px;color:{MUTED_LT}">
            Built with <b style="color:{MUTED}">LangGraph</b> ·
            <b style="color:{MUTED}">FastAPI</b> ·
            <b style="color:{MUTED}">Streamlit</b>
          </span>
        </div>""",
        unsafe_allow_html=True,
    )
