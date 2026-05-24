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
    return f"""You are a portfolio risk analyst. A user just uploaded their portfolio. Identify the 3 most notable vulnerabilities, risks, or special observations specific to THIS portfolio.

Portfolio holdings:
{holdings_lines}{sector_lines}

Rules:
- Each observation must be concrete and data-driven — use actual stock names, tickers, and percentages from above.
- Cover distinct issues (e.g. single-stock concentration, sector imbalance, missing diversification, over-exposure to a volatile segment).
- The "text" field states the observation itself (max 10 words, e.g. "TCS alone is 40% of your portfolio").
- The "query" field is a question that investigates that specific observation further.
- Assign "high" priority to the most critical risk, "medium" and "low" to the rest.

Return ONLY a JSON array with exactly 3 items. No explanation, no markdown, just the array:
[
  {{"text": "<observation, max 10 words, with specific numbers>", "query": "<question investigating this observation>", "priority": "<high|medium|low>"}},
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
    n = len(holdings)
    summary = f"{n} holdings. Top positions:\n{holdings_lines}"
    truncated = last_response[:400] + "..." if len(last_response) > 400 else last_response
    return f"""You are a portfolio risk analyst reviewing a conversation about a user's portfolio.

Portfolio summary: {summary}

User just asked: "{last_question}"
Assistant replied: "{truncated}"

Identify 3 specific vulnerabilities, risks, or notable observations about this portfolio. Prioritise observations that are relevant to what was just discussed, but also surface other important portfolio-specific risks.

Rules:
- Each observation must name actual stocks, percentages, or sectors from the portfolio data — no generic statements.
- The "text" field states the observation (max 10 words, e.g. "HDFC Bank is 35% of your holdings").
- The "query" field is a question that investigates that specific observation further.
- All 3 observations must be distinct issues.

Return ONLY a JSON array with exactly 3 items. No explanation, no markdown, just the array:
[
  {{"text": "<observation, max 10 words, with specific numbers>", "query": "<question investigating this observation>", "priority": "<high|medium|low>"}},
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
        suggestions = _parse_suggestions(response.content)
        # Pad to 3 with fallbacks if LLM returned fewer items
        existing_actions = {s["action"] for s in suggestions}
        for fb in _FALLBACKS:
            if len(suggestions) >= 3:
                break
            if fb["action"] not in existing_actions:
                suggestions.append(fb)
        return suggestions[:3]
    except Exception:
        return list(_FALLBACKS)
