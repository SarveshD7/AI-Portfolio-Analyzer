# LLM-Powered Smart Suggestions Design

**Date:** 2026-05-24  
**Status:** Approved

## Problem

The current `generate_smart_suggestions` function in `backend/tools/suggestions.py` uses hardcoded pattern-matching over portfolio metric data to produce 3 suggestion pills. This has two problems:

1. Suggestions don't adapt to what the user just asked — they're driven only by metric thresholds, not conversation context.
2. After a portfolio modification the frontend hardcodes its own 3 static suggestions (lines 911–915 of `frontend/app.py`), bypassing the backend entirely.

## Goal

Replace pattern-matching with an LLM call that generates 3 genuinely contextual next questions:
- **Initial load**: based on the actual portfolio composition and sector signals.
- **After each query**: based on the latest user question and the assistant's response.

## Scope

- Replace `generate_smart_suggestions` internals in `backend/tools/suggestions.py`.
- Update two call sites in `backend/agent.py` to pass the new context parameters.
- Remove the 3 hardcoded post-modification suggestions in `frontend/app.py`.
- No changes to `main.py`, other tools, or the frontend suggestion rendering logic.

## Design

### `generate_smart_suggestions` — new signature

```python
def generate_smart_suggestions(
    portfolio_composition: dict = None,
    sector_breakdown: dict = None,
    last_question: str = None,
    last_response: str = None,
) -> list[dict]:
```

All metric-specific parameters (`sharpe_data`, `returns_data`, `drawdown_data`, `var_data`) are removed. They were only used by the pattern-matching logic, which is being replaced.

### Two prompt branches

**Initial (no `last_question`):**

Gives the LLM:
- Holdings list: company name, ticker, weight %
- Top sectors from `sector_breakdown` (if available)

Asks for 3 first questions that highlight the most notable portfolio characteristics — concentration risks, diversification gaps, large single-stock positions, or general performance curiosity.

**After query (has `last_question` + `last_response`):**

Gives the LLM:
- Compact portfolio summary (top 5 holdings by weight)
- The user's last question
- The assistant's last response (truncated to 400 chars to control tokens)

Asks for 3 natural follow-up questions that flow from what was just discussed.

### LLM output format

The LLM is prompted to return a JSON array with exactly 3 items:

```json
[
  {"text": "40% in Financials", "query": "Am I overexposed to the Financial sector?", "priority": "high"},
  {"text": "Check returns", "query": "How has my portfolio performed over the last year?", "priority": "medium"},
  {"text": "Show risk metrics", "query": "What is the Sharpe ratio and VaR of my portfolio?", "priority": "low"}
]
```

- `text`: ≤6 words, shown on the pill button.
- `query`: Full question sent to the agent when clicked.
- `priority`: `"high"` / `"medium"` / `"low"` (controls pill border colour in the UI).

The backend adds `"action"` as a stable slug derived from the query text (`re.sub(r'\W+', '_', query.lower())[:40]`) so the frontend's explored-suggestion deduplication still works correctly.

### Fallback

If the LLM call throws, times out, or returns unparseable JSON, `generate_smart_suggestions` catches the exception and returns the existing hardcoded fallback starters. The function always returns exactly 3 suggestions.

### Model

Uses the same `ChatOpenAI` instance already configured in `agent.py` (OpenRouter / Llama 3.3 70B). `suggestions.py` imports and calls it directly, keeping the LLM configuration in one place.

To avoid circular imports (`agent.py` imports `suggestions.py` which would then need to import from `agent.py`), the LLM client is instantiated separately inside `suggestions.py` using the same env vars. Same model, same base URL, independent instance.

### `agent.py` — updated call sites

**`is_initial` path:**
```python
suggestions = generate_smart_suggestions(
    portfolio_composition=comp,
    sector_breakdown=sector,
)
```

**Main path (after tool call and response generation):**
```python
suggestions = generate_smart_suggestions(
    portfolio_composition=all_analysis.get("portfolio_composition"),
    sector_breakdown=all_analysis.get("sector_breakdown"),
    last_question=question,
    last_response=response_text,
)
```

The six metric-specific kwargs (`sharpe_data=`, `returns_data=`, etc.) are removed from both call sites.

### `frontend/app.py` — remove hardcoded post-modification suggestions

Lines 911–915 currently set `st.session_state.current_suggestions` to a fixed list of 3 pills whenever a portfolio modification is applied. These lines are removed. The `else` branch that uses `result.get("suggestions", [])` already handles this case correctly, so no replacement code is needed.

## Files changed

| File | Change |
|---|---|
| `backend/tools/suggestions.py` | Replace pattern-matching with LLM call; new signature; retain fallback |
| `backend/agent.py` | Update two `generate_smart_suggestions` call sites |
| `frontend/app.py` | Remove hardcoded post-modification suggestions (3 lines) |

## Out of scope

- Caching suggestions between turns (each turn gets a fresh LLM call)
- Showing more or fewer than 3 suggestions
- Changing the frontend suggestion pill rendering
