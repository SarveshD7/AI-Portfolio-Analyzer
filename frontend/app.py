import html as html_lib
import io

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

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

        # Message history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                chat_bubble(msg["role"], msg["content"])

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
