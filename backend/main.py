from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


class Message(BaseModel):
    message: str


@app.post("/test")
def test(body: Message):
    return {"echo": body.message, "timestamp": datetime.utcnow().isoformat()}
