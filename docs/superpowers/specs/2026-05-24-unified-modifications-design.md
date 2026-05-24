# Unified Portfolio Modification Interface Design

**Date:** 2026-05-24  
**Status:** Approved

## Problem

`simulate_portfolio_change_tool` has three separate parameters (`remove`, `add`, `change_weight`). For compound user requests ("remove X, add Y at 15%, and bump Z to 30%"), the LLM may make multiple separate tool calls. The agent loop overwrites `called_tool` and `tool_results` on each iteration — only the last call's result becomes the `portfolio_update`. Earlier changes are silently lost.

## Goal

Replace the three-param interface with a single `modifications: dict[str, float]` parameter where the weight value encodes the intent:

- `0.0` → remove the ticker
- `> 0` and already in portfolio → update weight
- `> 0` and not in portfolio → validate via yfinance, then add

## Scope

- Rewrite `modify_portfolio` in `backend/tools/portfolio_modification.py`
- Update `simulate_portfolio_change_tool` signature and docstring in `backend/agent.py`
- No changes to `main.py`, `app.py`, `revert_to_original_portfolio_tool`, or any other file

## Design

### `modify_portfolio` — new signature

```python
def modify_portfolio(
    tickers: list,
    weights: list,
    modifications: dict[str, float],
) -> dict:
```

#### Processing logic

Build `portfolio = {ticker: weight}` from current `tickers`/`weights`. Iterate over `modifications` in one pass:

| Condition | Action |
|---|---|
| `weight == 0.0` | Drop ticker from portfolio; record `"Removed {ticker} (was {old_pct}%)"` |
| `weight > 0` and ticker in portfolio | Update weight; record `"Changed {ticker}: {old}% → {new}%"` |
| `weight > 0` and ticker not in portfolio | Call `_validate_ticker`; if valid add, if not append to `invalid_tickers` and record skip message |

After processing all entries, raise `ValueError` if portfolio is empty. Normalize weights (divide each by sum), round to 6 decimal places. Append `" (weights rebalanced to 100%)"` to summary if the pre-normalization total deviates from 1.0 by more than 0.01.

#### Return shape (unchanged)

```python
{
    "tickers":         [...],
    "weights":         [...],          # normalized, rounded to 6dp
    "changes_summary": "...",
    "invalid_tickers": [...],          # empty list when all valid
}
```

#### `changes_summary` wording

- Remove: `"Removed RELIANCE.NS (was 30.0%)"`
- Update: `"Changed TCS.NS: 25.0% → 35.0%"`
- Add: `"Added INFY.NS (15.0%)"`
- Skip: `"FAKESTK not found on yfinance — skipped"`
- Combined example: `"Removed RELIANCE.NS (was 30.0%), Changed TCS.NS: 25.0% → 35.0%, Added INFY.NS (15.0%) (weights rebalanced to 100%)"`

### `simulate_portfolio_change_tool` — new signature

```python
@tool
def simulate_portfolio_change_tool(
    tickers: List[str],
    weights: List[float],
    modifications: Dict[str, float],
) -> dict:
    """Simulate what-if portfolio changes ...
    
    Args:
        tickers: Current portfolio tickers
        weights: Current portfolio weights (decimals summing to 1.0)
        modifications: Dict of {ticker: weight} encoding all changes at once.
            weight = 0.0  → remove the ticker completely
            weight > 0    → set new weight (updates if already held, adds if new)
            Always encode ALL changes in one call — remove, add, and weight updates together.
            Example: {"RELIANCE.NS": 0.0, "TCS.NS": 0.35, "INFY.NS": 0.15}
    """
```

### `_match_ticker` usage

`_match_ticker` is still used for existing-ticker lookups (case-insensitive, exchange-suffix tolerance). For the remove and update cases, the matched key is used. For new additions, the ticker is used as-is after validation.

## Files changed

| File | Change |
|---|---|
| `backend/tools/portfolio_modification.py` | Rewrite `modify_portfolio` with new single-param logic |
| `backend/agent.py` | Update `simulate_portfolio_change_tool` signature and docstring |
