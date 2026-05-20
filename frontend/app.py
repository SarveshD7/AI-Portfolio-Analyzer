import io

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

PERIOD_MAP = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "3 Years": "3y",
    "5 Years": "5y",
}

st.title("Kalpi Portfolio Analyzer")

uploaded = st.file_uploader("Upload portfolio CSV (columns: Ticker, Weight)", type="csv")

question = st.text_input(
    "Ask about your portfolio",
    placeholder="e.g. What's my 6-month return? / How did I do last year?",
)

# period_label = st.selectbox(
#     "Default period (used if not mentioned in your question)",
#     list(PERIOD_MAP.keys()),
#     index=3,
# )

if st.button("Analyze"):
    if uploaded is None:
        st.error("Please upload a portfolio CSV file.")
    elif not question.strip():
        st.error("Please enter a question about your portfolio.")
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
                payload = {
                    "tickers": df["Ticker"].tolist(),
                    "weights": df["Weight"].tolist(),
                    "period": "1y",
                    "question": question.strip(),
                }
                try:
                    with st.spinner("Analyzing..."):
                        resp = requests.post("http://localhost:8000/analyze", json=payload)

                    if resp.status_code == 200:
                        result = resp.json()

                        with st.chat_message("assistant"):
                            st.markdown(result["response"])

                        data = result.get("data", {})
                        if result.get("visualization") == "line_chart" and data.get("daily_returns"):
                            daily = pd.DataFrame(data["daily_returns"])
                            fig = go.Figure(
                                go.Scatter(
                                    x=daily["date"],
                                    y=daily["return"],
                                    mode="lines",
                                    name="Daily Return (%)",
                                )
                            )
                            fig.update_layout(
                                title=f"Daily Portfolio Returns ({data.get('period', '')})",
                                xaxis_title="Date",
                                yaxis_title="Return (%)",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Backend error: {resp.json().get('detail', resp.text)}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to backend. Is it running on port 8000?")
