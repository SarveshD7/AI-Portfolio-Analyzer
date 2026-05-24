# Ticker Validation on Portfolio Addition

**Date:** 2026-05-24  
**Status:** Approved

## Problem

When a user asks to add a new stock to their portfolio via the what-if flow (e.g. "What if I added FAKESTK.NS?"), `modify_portfolio` blindly accepts the ticker without checking whether it exists on yfinance. The stock is added to the portfolio and all subsequent analysis silently fails or returns garbage data.

## Goal

Validate each new ticker in the `add` dict against yfinance before adding it. Valid tickers are added normally; invalid ones are skipped with a clear message surfaced through the existing LLM response chain.

## Scope

- Only new additions (`add` dict in `modify_portfolio`) are validated.
- Existing portfolio tickers (`remove`, `change_weight`) are never re-validated — they were already accepted at upload time.
- No changes to `main.py`, `agent.py`, or any other tool file.

## Design

### `_validate_ticker(ticker: str) -> bool`

A new private helper added to `backend/tools/portfolio_modification.py`.

```python
def _validate_ticker(ticker: str) -> bool:
    try:
        info = yf.Ticker(ticker).fast_info
        return getattr(info, "last_price", None) is not None
    except Exception:
        return True  # safe fallback: don't block on transient errors
```

**Why `fast_info`:** It is a lightweight, cached yfinance call (~100ms) that does not fetch full price history. It returns a `FastInfo` object with a `last_price` field that is `None` for unrecognised tickers.

**Why fallback to `True`:** A transient network error or yfinance API change should not block the user from adding a ticker. The safe failure mode is to let it through rather than silently drop a valid ticker.

### Changes to `modify_portfolio`

The additions loop is updated to:

1. Call `_validate_ticker(ticker)` for each entry in `add`.
2. If valid → add to portfolio and record in `changes` as today.
3. If invalid → append to `invalid_tickers: list[str]` and skip.
4. Extend `changes_summary` with a note per skipped ticker: `"FAKESTK.NS not found on yfinance — skipped"`.
5. Return `invalid_tickers` in the result dict.

### Return shape (updated)

```python
{
    "tickers":         [...],
    "weights":         [...],
    "changes_summary": "Added INFY.NS (10.0%), FAKESTK.NS not found on yfinance — skipped (weights rebalanced to 100%)",
    "invalid_tickers": ["FAKESTK.NS"],   # new field; empty list when all additions are valid
}
```

### LLM response chain

No changes to `agent.py`. The full result dict is already serialised into the ToolMessage the LLM reads. When `invalid_tickers` is non-empty the LLM sees the note in `changes_summary` and the explicit `invalid_tickers` list, and naturally responds with something like:

> "I added INFY.NS at 10% to your portfolio. FAKESTK.NS wasn't found on yfinance so it was skipped — please double-check the ticker symbol."

## Error handling

| Scenario | Behaviour |
|---|---|
| Ticker not on yfinance | Skipped; noted in `changes_summary` and `invalid_tickers` |
| `fast_info` throws (network) | Fallback: ticker treated as valid, added normally |
| All additions are invalid | Portfolio unchanged (existing holdings only); `changes_summary` explains why |
| `add` is empty or `None` | Existing behaviour unchanged |

## Files changed

| File | Change |
|---|---|
| `backend/tools/portfolio_modification.py` | Add `_validate_ticker`; update additions loop; update return dict |

## Out of scope

- Validating tickers at portfolio upload time (separate concern)
- A dedicated `/validate-ticker` endpoint (violates single-endpoint architecture)
- Frontend pre-validation (requires extra round-trips and frontend changes)
