import requests
import streamlit as st

st.title("Kalpi Portfolio Analyzer")

message = st.text_input("Message")

if st.button("Send to Backend"):
    try:
        response = requests.post(
            "http://localhost:8000/test",
            json={"message": message},
        )
        data = response.json()
        st.success(f"Echo: {data['echo']}")
        st.info(f"Timestamp: {data['timestamp']}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to backend. Is it running on port 8000?")
