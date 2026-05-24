# Unified Portfolio Modification Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three-param (`remove`, `add`, `change_weight`) modification interface with a single `modifications: dict[str, float]` param where weight=0 means remove, positive weight means add-or-update.

**Architecture:** `modify_portfolio` is rewritten to iterate over the `modifications` dict in one pass, inferring intent from weight value and portfolio membership. `simulate_portfolio_change_tool` signature is updated to match. No other files change.

**Tech Stack:** Python, yfinance, LangChain `@tool`

---

## File Map

| Action | File |
|---|---|
| Rewrite | `backend/tools/portfolio_modification.py` |
| Modify  | `backend/agent.py` — `simulate_portfolio_change_tool` definition |

---

### Task 1: Rewrite `portfolio_modification.py`

**Files:**
- Modify: `backend/tools/portfolio_modification.py`

- [ ] **Step 1: Replace the entire file**

Overwrite `backend/tools/portfolio_modification.py` with:

```python
import yfinance as yf


def _validate_ticker(ticker: str) -> bool:
    try:
        info = yf.Ticker(ticker).fast_info
        return getattr(info, "last_price", None) is not None
    except Exception:
        return True


def _match_ticker(target: str, portfolio: dict) -> str | None:
    """Return the portfolio key that best matches the target string."""
    t = target.strip()
    if t in portfolio:
        return t
    tup = t.upper()
    for key in portfolio:
        if key.upper() == tup:
            return key
    t_base = tup.split(".")[0]
    for key in portfolio:
        if key.upper().split(".")[0] == t_base:
            return key
    return None


def modify_portfolio(
    tickers: list,
    weights: list,
    modifications: dict[str, float],
) -> dict:
    """Apply modifications to a portfolio and return the rebalanced result.

    modifications: {ticker: weight}
        weight == 0.0  → remove ticker
        weight > 0, ticker in portfolio → update weight
        weight > 0, ticker not in portfolio → validate via yfinance, then add
    """
    portfolio = {t: float(w) for t, w in zip(tickers, weights)}
    changes: list[str] = []
    invalid_tickers: list[str] = []

    for ticker, weight in (modifications or {}).items():
        matched = _match_ticker(ticker, portfolio)

        if weight == 0.0:
            if matched:
                old_pct = portfolio[matched] * 100
                changes.append(f"Removed {matched} (was {old_pct:.1f}%)")
                del portfolio[matched]
        elif matched:
            old_pct = portfolio[matched] * 100
            portfolio[matched] = float(weight)
            changes.append(f"Changed {matched}: {old_pct:.1f}% → {float(weight) * 100:.1f}%")
        else:
            if not _validate_ticker(ticker):
                invalid_tickers.append(ticker)
                changes.append(f"{ticker} not found on yfinance — skipped")
                continue
            portfolio[ticker] = float(weight)
            changes.append(f"Added {ticker} ({float(weight) * 100:.1f}%)")

    if not portfolio:
        raise ValueError("Portfolio is empty after modifications — cannot proceed.")

    total = sum(portfolio.values())
    normalized = {t: w / total for t, w in portfolio.items()}

    new_tickers = list(normalized.keys())
    new_weights = [round(w, 6) for w in normalized.values()]

    summary = ", ".join(changes) if changes else "No changes applied"
    if changes and abs(total - 1.0) > 0.01:
        summary += " (weights rebalanced to 100%)"

    return {
        "tickers":         new_tickers,
        "weights":         new_weights,
        "changes_summary": summary,
        "invalid_tickers": invalid_tickers,
    }
```

---

### Task 2: Update `simulate_portfolio_change_tool` in `agent.py`

**Files:**
- Modify: `backend/agent.py`

- [ ] **Step 1: Replace the tool definition**

Find the entire `simulate_portfolio_change_tool` function (from `@tool` through the closing `"""`) and replace it with:

```python
@tool
def simulate_portfolio_change_tool(
    tickers: List[str],
    weights: List[float],
    modifications: Dict[str, float],
) -> dict:
    """Simulate what-if portfolio changes and return the rebalanced portfolio.

    Use when the user asks "what if" questions or wants to modify the portfolio:
    - "What if I exit / remove / sell [stock]?"
    - "What if I didn't have [stock] or [sector]?"
    - "What if I added / bought [stock]?"
    - "Replace X with Y"
    - "Increase / decrease [stock] to X%"
    - "Remove all my energy / tech / banking stocks"
    - Compound: "Remove TCS, add INFY at 15%, and increase HDFC to 30%"

    Always encode ALL changes in a single call using the modifications dict.
    Never make multiple separate tool calls for one compound request.

    After calling this tool, tell the user what changed and what the new portfolio
    looks like. All subsequent analysis will automatically use the modified portfolio.

    Args:
        tickers: Current portfolio tickers exactly as shown in the portfolio context
        weights: Current portfolio weights (decimals summing to 1.0)
        modifications: Dict of {ticker: weight} encoding ALL changes at once.
            weight = 0.0  → remove ticker completely
            weight > 0    → update weight if already held, add if new stock
            Example: {"RELIANCE.NS": 0.0, "TCS.NS": 0.35, "INFY.NS": 0.15}
            MUST use correct yfinance ticker with exchange suffix:
            Indian NSE (.NS): "INFY.NS", "HDFCBANK.NS"  |  US (no suffix): "AAPL", "NVDA"
    """
    return modify_portfolio(tickers, weights, modifications)
```

---

### Task 3: Smoke test

- [ ] **Step 1: Start the backend**

```
cd backend
uvicorn main:app --reload
```

- [ ] **Step 2: Test a compound modification**

```
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
    "weights": [0.5, 0.3, 0.2],
    "period": "1y",
    "question": "Remove Reliance, increase TCS to 50%, and add HDFCBANK at 20%",
    "is_initial": false
  }' | python -m json.tool
```

Expected: `portfolio_update.tickers` contains `TCS.NS`, `INFY.NS`, `HDFCBANK.NS` but NOT `RELIANCE.NS`. `changes_summary` mentions all three operations.

- [ ] **Step 3: Test zero-weight removal**

```
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "NVDA"],
    "weights": [0.4, 0.35, 0.25],
    "period": "1y",
    "question": "What if I sold all my Apple shares?",
    "is_initial": false
  }' | python -m json.tool
```

Expected: `portfolio_update.tickers` contains `MSFT` and `NVDA` but NOT `AAPL`. Weights normalize to sum 1.0.

- [ ] **Step 4: Test invalid ticker in modifications**

```
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT"],
    "weights": [0.6, 0.4],
    "period": "1y",
    "question": "Add FAKESTK at 10%",
    "is_initial": false
  }' | python -m json.tool
```

Expected: `portfolio_update` is either absent or unchanged. `response` text mentions FAKESTK was not found.
