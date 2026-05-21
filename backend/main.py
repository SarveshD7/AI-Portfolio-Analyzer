from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import run_agent

load_dotenv()

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
    question: str = "How has my portfolio performed?"
    accumulated_analysis: dict = Field(default_factory=dict)
    is_initial: bool = False


@app.post("/analyze")
def analyze(body: AnalyzeRequest):
    if not body.tickers:
        raise HTTPException(status_code=422, detail="tickers list cannot be empty.")
    if len(body.tickers) != len(body.weights):
        raise HTTPException(status_code=422, detail="tickers and weights must have the same length.")
    total = round(sum(body.weights), 6)
    if abs(total - 1.0) > 0.01:
        raise HTTPException(status_code=422, detail=f"Weights must sum to 1.0, got {total}.")

    portfolio = {
        "tickers": body.tickers,
        "weights": body.weights,
        "period": body.period,
    }

    try:
        return run_agent(
            portfolio,
            body.question,
            accumulated_analysis=body.accumulated_analysis,
            is_initial=body.is_initial,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
