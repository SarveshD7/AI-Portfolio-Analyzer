from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tools.returns import calculate_portfolio_returns

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


class Message(BaseModel):
    message: str


@app.post("/test")
def test(body: Message):
    return {"echo": body.message, "timestamp": datetime.utcnow().isoformat()}


class AnalyzeRequest(BaseModel):
    tickers: list[str]
    weights: list[float]
    period: str = "1y"
    # future fields: query: str, intent: str


@app.post("/analyze")
def analyze(body: AnalyzeRequest):
    if not body.tickers:
        raise HTTPException(status_code=422, detail="tickers list cannot be empty.")
    if len(body.tickers) != len(body.weights):
        raise HTTPException(status_code=422, detail="tickers and weights must have the same length.")
    total = round(sum(body.weights), 6)
    if abs(total - 1.0) > 0.01:
        raise HTTPException(status_code=422, detail=f"Weights must sum to 1.0, got {total}.")

    # LLM routing will go here — for now route directly to returns tool
    try:
        return calculate_portfolio_returns(body.tickers, body.weights, body.period)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
