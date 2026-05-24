# LLM-Powered Smart Suggestions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded pattern-matching suggestion engine with an LLM call that generates 3 contextually relevant next questions — based on portfolio composition on initial load, and on the latest question + response on every subsequent turn.

**Architecture:** `suggestions.py` gets a new LLM-powered `generate_smart_suggestions` with a stripped signature. Two call sites in `agent.py` are updated to pass the right context. Three hardcoded post-modification suggestion lines in `app.py` are removed.

**Tech Stack:** Python, LangChain (`ChatOpenAI`), OpenRouter, Streamlit

---

## File Map

| Action | File |
|---|---|
| Rewrite | `backend/tools/suggestions.py` |
| Modify  | `backend/agent.py` — line ~416 (initial call) and lines ~536–543 (main call) |
| Modify  | `frontend/app.py` — lines 911–922 (remove hardcoded suggestions, unify into one branch) |

---

### Task 1: Rewrite `suggestions.py` with LLM-powered generation

**Files:**
- Modify: `backend/tools/suggestions.py`

- [ ] **Step 1: Replace the entire file with the new implementation**

Overwrite `backend/tools/suggestions.py` with:

```python
import json
import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

_llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

_FALLBACKS = [
    {
        "text": "How did my portfolio perform?",
        "query": "How has my portfolio performed over the last year?",
        "action": "_fallback_returns",
        "priority": "low",
    },
    {
        "text": "Compare vs Nifty 50",
        "query": "How does my portfolio compare against Nifty 50?",
        "action": "_fallback_benchmark",
        "priority": "low",
    },
    {
        "text": "Show my risk metrics",
        "query": "What is the risk profile of my portfolio?",
        "action": "_fallback_risk",
        "priority": "low",
    },
]


def _make_action(query: str) -> str:
    return re.sub(r'\W+', '_', query.lower())[:40]


def _parse_suggestions(raw: str) -> list[dict]:
    text = re.sub(r'```(?:json)?', '', raw).strip()
    start = text.find('[')
    end = text.rfind(']') + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array in LLM response")
    items = json.loads(text[start:end])
    result = []
    for item in items[:3]:
        query = item.get("query", "")
        result.append({
            "text":     item.get("text", query[:40]),
            "query":    query,
            "action":   _make_action(query),
            "priority": item.get("priority", "medium"),
        })
    return result


def _build_initial_prompt(portfolio_composition: dict, sector_breakdown: dict) -> str:
    holdings = portfolio_composition.get("holdings", [])
    holdings_lines = "\n".join(
        f"  - {h['name']} ({h['ticker']}): {h['weight_pct']:.1f}%"
        for h in holdings
    )
    sector_lines = ""
    if sector_breakdown:
        top_sectors = sector_breakdown.get("breakdown", [])[:3]
        if top_sectors:
            sector_lines = "\nTop sectors:\n" + "\n".join(
                f"  - {s['label']}: {s['weight_pct']:.1f}%"
                for s in top_sectors
            )
    return f"""You are a portfolio analysis assistant. A user just uploaded their portfolio.

Portfolio holdings:
{holdings_lines}{sector_lines}

Generate exactly 3 short, specific questions the user would most want to ask first.
Focus on the most notable characteristics: heavy concentration in a specific sector or stock,
lack of diversification, general returns, or risk.

Return ONLY a JSON array with exactly 3 items. No explanation, no markdown, just the array:
[
  {{"text": "<very short label, max 5 words>", "query": "<full question>", "priority": "<high|medium|low>"}},
  {{"text": "...", "query": "...", "priority": "..."}},
  {{"text": "...", "query": "...", "priority": "..."}}
]"""


def _build_followup_prompt(
    portfolio_composition: dict,
    last_question: str,
    last_response: str,
) -> str:
    holdings = portfolio_composition.get("holdings", []) if portfolio_composition else []
    top5 = holdings[:5]
    holdings_lines = "\n".join(
        f"  - {h['name']} ({h['ticker']}): {h['weight_pct']:.1f}%"
        for h in top5
    )
    n = len(portfolio_composition.get("holdings", [])) if portfolio_composition else 0
    summary = f"{n} holdings. Top positions:\n{holdings_lines}"
    truncated = last_response[:400] + "..." if len(last_response) > 400 else last_response
    return f"""You are a portfolio analysis assistant.

Portfolio summary: {summary}

User just asked: "{last_question}"
Assistant replied: "{truncated}"

Generate exactly 3 natural follow-up questions that flow from what was just discussed.

Return ONLY a JSON array with exactly 3 items. No explanation, no markdown, just the array:
[
  {{"text": "<very short label, max 5 words>", "query": "<full question>", "priority": "<high|medium|low>"}},
  {{"text": "...", "query": "...", "priority": "..."}},
  {{"text": "...", "query": "...", "priority": "..."}}
]"""


def generate_smart_suggestions(
    portfolio_composition: dict = None,
    sector_breakdown: dict = None,
    last_question: str = None,
    last_response: str = None,
) -> list[dict]:
    try:
        if last_question and portfolio_composition:
            prompt = _build_followup_prompt(
                portfolio_composition, last_question, last_response or ""
            )
        elif portfolio_composition:
            prompt = _build_initial_prompt(portfolio_composition, sector_breakdown)
        else:
            return list(_FALLBACKS)
        response = _llm.invoke(prompt)
        return _parse_suggestions(response.content)
    except Exception:
        return list(_FALLBACKS)
```

---

### Task 2: Update the two `generate_smart_suggestions` call sites in `agent.py`

**Files:**
- Modify: `backend/agent.py`

The current main call site (around line 536) passes six metric-specific kwargs that no longer exist in the new signature. The initial call site is already compatible but shown for completeness.

- [ ] **Step 1: Update the main call site**

Find this block (around line 536):

```python
    suggestions = generate_smart_suggestions(
        portfolio_composition=all_analysis.get("portfolio_composition"),
        sector_breakdown=all_analysis.get("sector_breakdown"),
        sharpe_data=all_analysis.get("sharpe_data"),
        returns_data=all_analysis.get("returns_data"),
        drawdown_data=all_analysis.get("drawdown_data"),
        var_data=all_analysis.get("var_data"),
    )
```

Replace it with:

```python
    suggestions = generate_smart_suggestions(
        portfolio_composition=all_analysis.get("portfolio_composition"),
        sector_breakdown=all_analysis.get("sector_breakdown"),
        last_question=question,
        last_response=response_text,
    )
```

- [ ] **Step 2: Verify the initial call site needs no change**

The `is_initial` path (around line 414) already matches the new signature:

```python
        suggestions = generate_smart_suggestions(
            portfolio_composition=comp,
            sector_breakdown=sector,
        )
```

No edit needed here — just confirm it matches.

---

### Task 3: Remove hardcoded post-modification suggestions from `frontend/app.py`

**Files:**
- Modify: `frontend/app.py`

Currently, when a portfolio modification comes back from the backend, the frontend ignores the backend's `suggestions` and hardcodes its own 3 pills. The fix is to unify both branches to use the backend suggestions.

- [ ] **Step 1: Replace the split if/else suggestions block**

Find this block (around lines 910–922 inside `_call_api`):

```python
            if portfolio_update:
                st.session_state.current_portfolio = {
                    "tickers": portfolio_update["tickers"],
                    "weights": portfolio_update["weights"],
                    "period":  portfolio_update.get("period", p["period"]),
                }
                st.session_state.portfolio_is_modified = True
                st.session_state.modification_history.append({
                    "summary": portfolio_update.get("changes_summary", ""),
                    "tickers": portfolio_update["tickers"],
                    "weights": portfolio_update["weights"],
                })
                st.session_state.current_suggestions = [
                    {"text": "Check returns", "query": "How did my portfolio perform over the last year?", "action": "_mod_compare", "priority": "high"},
                    {"text": "Check risk profile", "query": "What is the Sharpe ratio of my portfolio?", "action": "_mod_risk", "priority": "medium"},
                    {"text": "Show composition", "query": "Show me my portfolio composition", "action": "_mod_composition", "priority": "low"},
                ]
            else:
                explored_set = set(st.session_state.explored_suggestions)
                st.session_state.current_suggestions = [
                    s for s in result.get("suggestions", [])
                    if s["action"] not in explored_set
                ]
```

Replace it with:

```python
            if portfolio_update:
                st.session_state.current_portfolio = {
                    "tickers": portfolio_update["tickers"],
                    "weights": portfolio_update["weights"],
                    "period":  portfolio_update.get("period", p["period"]),
                }
                st.session_state.portfolio_is_modified = True
                st.session_state.modification_history.append({
                    "summary": portfolio_update.get("changes_summary", ""),
                    "tickers": portfolio_update["tickers"],
                    "weights": portfolio_update["weights"],
                })
            explored_set = set(st.session_state.explored_suggestions)
            st.session_state.current_suggestions = [
                s for s in result.get("suggestions", [])
                if s["action"] not in explored_set
            ]
```

The `else` is removed; the suggestions assignment now always uses the backend response.

---

### Task 4: Smoke test

- [ ] **Step 1: Start the backend**

```
cd backend
uvicorn main:app --reload
```

- [ ] **Step 2: Start the frontend**

In a second terminal:
```
cd frontend
streamlit run app.py
```

- [ ] **Step 3: Upload a portfolio and verify initial suggestions**

Upload a CSV with a few holdings. After the initial analysis loads, you should see 3 suggestion pills that reference specific characteristics of that portfolio (e.g. a stock or sector that dominates) rather than generic fallback text.

Expected: Pills are specific to the portfolio — not generic "How did my portfolio perform?" unless the portfolio has no notable signals.

- [ ] **Step 4: Click a suggestion and verify follow-up suggestions update**

Click one of the suggestion pills. After the response loads, the 3 pills should update to reflect natural follow-ups to that specific question and answer — not repeat the same pills.

Expected: New pills are contextually relevant to what was just discussed.

- [ ] **Step 5: Trigger a what-if modification and verify suggestions update**

Ask something like "What if I added AAPL at 10%?". After the portfolio updates, the suggestions should be backend-generated follow-ups (e.g. "Compare original vs modified", "Check new risk", "Show updated composition") — not the old hardcoded 3 static pills.

Expected: Suggestions come from the backend LLM call, not hardcoded text.
