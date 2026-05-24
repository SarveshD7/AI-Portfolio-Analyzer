# Kalpi — AI Portfolio Analyzer

> Institutional-grade portfolio analysis through natural language conversation.

Kalpi is an AI-powered investment analysis tool that lets you interrogate your portfolio in plain English. Ask about returns, drawdowns, correlations, sector exposure, or stress-test against historical crises — all through a chat interface backed by deterministic financial calculations and dynamic visualizations.

---

## Demo
| Feature Screenshots |
|:---:|
| **Chat + Canvas Interface** |
| <img width="800" alt="Chat + Canvas Interface" src="https://github.com/user-attachments/assets/4d1816f7-8121-4e14-8680-3e7ea659a853" /> |
| **Rolling Risk Metrics** |
| <img width="800" alt="Rolling Risk Metrics" src="https://github.com/user-attachments/assets/b9ef8b82-4680-4eb6-8ead-ab980ac4a137" /> |
| **Portfolio Correlations** | 
| <img width="1917" height="971" alt="image" src="https://github.com/user-attachments/assets/a95a1e4e-5b6b-42cc-82bc-7d4e69e9a667" /> |

---

## Features

### Conversational Analysis
Ask questions the way you'd ask a financial analyst. The LLM routes each query to the right calculation tool and wraps the result in plain-English narrative. It never guesses numbers — all metrics come from deterministic Python functions on real market data.

### Performance Analytics
- **Historical returns** — total return and daily P&L curve for any timeframe
- **Per-stock contribution breakdown** — which holdings drove gains and which dragged
- **Best / worst performer** identification across any period

### Risk Metrics
- **Sharpe ratio** — risk-adjusted return vs a configurable risk-free rate
- **Maximum drawdown** — peak-to-trough decline with exact dates and recovery tracking
- **Value at Risk** — historical VaR at 95% and 99% confidence, plus CVaR (Expected Shortfall)
- **Beta & Alpha** — systematic risk and excess return relative to any benchmark
- **Rolling metrics** — how volatility, Sharpe, and VaR have evolved over time (rolling chart)

### Benchmark Comparison
Side-by-side cumulative return chart against any major index. Supported benchmarks:

| Alias | Index |
|---|---|
| `nifty50` / `^NSEI` | Nifty 50 (default) |
| `sensex` / `^BSESN` | BSE Sensex |
| `sp500` / `^GSPC` | S&P 500 |
| `nasdaq` / `^IXIC` | NASDAQ Composite |
| `dow` / `^DJI` | Dow Jones Industrial Average |
| `niftybank` / `^NSEBANK` | Nifty Bank |

### Portfolio Composition & Diversification
- **Holdings breakdown** — allocation weights with company names
- **Sector concentration** — detect over-indexing to any single sector
- **Market-cap factor analysis** — Large Cap / Mid Cap / Small Cap tiers
- **Asset class breakdown** — equity, ETF, crypto, mutual funds
- **Correlation heatmap** — pairwise correlation matrix across all holdings

### Historical Event Simulation
Replay your portfolio through named market crises by simply asking:

| Query | Date Range |
|---|---|
| "COVID crash" | 2020-02-19 → 2020-03-23 |
| "COVID recovery" | 2020-03-23 → 2020-12-31 |
| "2008 financial crisis" / "GFC" | 2008-09-01 → 2009-03-31 |
| "dot-com crash" | 2000-03-01 → 2002-10-09 |

### What-If Portfolio Modifications
Simulate rebalancing scenarios in natural language — add, remove, and reweight multiple holdings in a single message. The modified portfolio becomes the active state for all subsequent analysis. Revert to the original at any time.

```
"What if I removed TCS, added HDFCBANK at 20%, and increased RELIANCE to 35%?"
"Compare my modified portfolio against the original."
"Go back to my original portfolio."
```

### Smart AI Suggestions
After every analysis, the assistant surfaces three portfolio-specific observations — naming actual tickers, weights, and percentages — with a clickable follow-up question for each. Suggestions regenerate contextually after every query.

---

## Architecture

```
User Query
    │
    ▼
FastAPI  /analyze  (main.py)
    │
    ▼
LangChain Agent  (agent.py)          ← LLM decides which tool to call
    │
    ├─ portfolio_returns_tool         → returns.py
    ├─ sharpe_ratio_tool              → risk.py
    ├─ max_drawdown_tool              → risk.py
    ├─ var_tool                       → risk.py
    ├─ beta_tool                      → risk.py
    ├─ rolling_metrics_tool           → risk.py
    ├─ benchmark_tool                 → benchmark.py
    ├─ concentration_tool             → concentration.py
    ├─ correlation_tool               → correlation.py
    ├─ portfolio_composition_tool     → portfolio_info.py
    ├─ simulate_portfolio_change_tool → portfolio_modification.py
    ├─ compare_portfolios_tool        → portfolio_comparison.py
    └─ revert_to_original_tool
    │
    ▼
Deterministic Python  (pandas + numpy + yfinance)
    │
    ▼
Streamlit Frontend  (frontend/app.py)
    ├─ Chat panel (left)
    ├─ Canvas / Visualization (right)   ← Plotly charts, metric cards, heatmaps
    └─ Smart Suggestions panel
```

**Key design principles:**
- **LLM as orchestrator, not calculator.** The model selects the right tool and narrates results; it never performs arithmetic.
- **One tool per concern.** Each financial metric is an isolated, independently testable function.
- **Stateful what-if simulations.** Original and current portfolios are tracked separately; modifications are reversible.

---

## Portfolio State Management

Understanding how state flows through a session is important for anyone extending this codebase.

### Three tiers of portfolio state

The Streamlit frontend owns all session state. The backend is completely stateless — every `/analyze` request carries the full portfolio context it needs.

| State key | What it holds | When it changes |
|---|---|---|
| `original_portfolio` | Tickers + weights from the uploaded CSV | Set once on upload; never mutated |
| `current_portfolio` | The active portfolio used in all analysis | Updated whenever a what-if modification is applied |
| `portfolio_is_modified` | Boolean flag | Flipped True on first modification, False on revert |
| `accumulated_analysis` | Growing dict of all tool results this session | Appended after each tool call |
| `modification_history` | Log of change summaries | Appended on each modification |

### Request / response state flow

```
Frontend                              Backend (stateless)
────────                              ───────────────────
current_portfolio   ─────────────►
original_portfolio  ─────────────►   agent.py builds prompt with both portfolios
accumulated_analysis ────────────►   LLM selects and calls a tool
question            ─────────────►   Tool runs; result is returned

                    ◄─────────────   response (narrative text)
                    ◄─────────────   visualization (chart data)
                    ◄─────────────   suggestions (3 LLM-generated observations)
                    ◄─────────────   portfolio_update (only if portfolio changed)
                    ◄─────────────   analysis_updates (new tool result to accumulate)
```

If `portfolio_update` is present in the response, the frontend applies it to `current_portfolio`. If it carries `is_revert: true`, the frontend restores `original_portfolio` as the active state.

### What-if modification lifecycle

1. User asks: *"What if I removed TCS, added HDFCBANK at 20%, and increased RELIANCE to 35%?"*
2. The LLM calls `simulate_portfolio_change_tool` with all three changes encoded in a single `modifications` dict — never split across multiple calls.
3. The tool validates each ticker, applies the changes, and proportionally rebalances remaining weights to sum to `1.0`.
4. The backend returns `portfolio_update` with the new tickers, weights, and a `changes_summary` string.
5. The frontend saves the current portfolio as `original_portfolio` (if not already modified) and sets `current_portfolio` to the new state.
6. All subsequent analysis in the session — Sharpe, drawdown, correlation, etc. — automatically runs against `current_portfolio`.
7. When the user says *"go back to my original"*, `revert_to_original_portfolio_tool` is called, and `original_portfolio` is restored as `current_portfolio`.

### Accumulated analysis and suggestions

The `accumulated_analysis` dict grows across the session as tools are called. It is passed to every `/analyze` request so that the suggestions LLM has cumulative context about what has already been computed — sector breakdown, composition, returns data — without re-running any analysis. This allows suggestions to become progressively more specific as the conversation deepens.

---

## Implementation Notes

### LLM as orchestrator, not calculator

The LLM's only jobs are tool selection and result narration. It never performs arithmetic, interpolates missing data, or estimates metrics. Every number in a response comes from a deterministic Python function. This keeps outputs reproducible and auditable.

### Tool docstrings as intent-routing configuration

Each tool's docstring enumerates the natural language phrasings that should trigger it. The LLM reads these at inference time to decide which tool to call. Updating routing behaviour — adding new trigger phrases, disambiguating overlapping intents — means editing docstrings, not prompts or classifiers.

### Token optimisation: visualization-only arrays stripped from tool messages

After a tool runs, large array fields (`daily_returns`, `drawdown_series`, `portfolio_cumulative`, `rolling_volatility`, etc.) are stripped from the `ToolMessage` sent back to the LLM for narrative generation. The full arrays go directly to the frontend for charting. This reduces per-call token usage by 2–5k tokens while keeping the LLM's context focused on summary statistics.

### Single `/analyze` endpoint

All interactions — regardless of which tool is needed — go through one endpoint. This keeps the API surface minimal, avoids coupling the frontend to specific tool names, and lets the agent freely chain or switch tools without any frontend changes.

### Smart suggestions are LLM-generated, not pattern-matched

Suggestions are not produced by hardcoded rules like "if Sharpe < 1, show suggestion X." A second LLM call generates them, instructed to cite specific tickers, weights, and percentages from the actual portfolio data. This produces observations that are genuinely portfolio-specific rather than generic boilerplate.

### What-if changes are encoded in a single tool call

Compound modifications — remove one stock, add another, reweight a third — are encoded in a single `modifications` dict and handled in one `simulate_portfolio_change_tool` call. The tool resolves all changes atomically, then renormalises weights. This avoids intermediate invalid portfolio states and prevents multiple tool calls for a single user intent.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.3 70B via [OpenRouter](https://openrouter.ai) |
| Agent framework | LangChain (`bind_tools`) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Visualizations | Plotly |
| Market data | yfinance |
| Data processing | pandas, numpy |

---

## Setup

### Prerequisites
- Python 3.10+
- An [OpenRouter](https://openrouter.ai) API key (free tier works)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AI-Portfolio-Analyzer.git
cd AI-Portfolio-Analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and add your OpenRouter key:

```env
OPENROUTER_API_KEY=your-openrouter-api-key-here
```

Copy the same file into the backend directory:

```bash
cp .env backend/.env
```

---

## Running the App

The app consists of two processes — a FastAPI backend and a Streamlit frontend. Run them in separate terminals.

**Terminal 1 — Backend**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend**
```bash
cd frontend
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

### Uploading a Portfolio

Upload a CSV file with two columns: `Ticker` and `Weight`.

```csv
Ticker,Weight
RELIANCE.NS,0.40
MCX.NS,0.20
BSE.NS,0.08
ZYDUSLIFE.NS,0.07
HDFCAMC.NS,0.06
TCS.NS,0.05
INFY.NS,0.05
SBIN.NS,0.04
MARUTI.NS,0.03
360ONE.NS,0.02
```

A sample portfolio is included at [`sample_portfolio.csv`](./sample_portfolio.csv).

**Ticker format rules:**
- Indian NSE stocks: append `.NS` — e.g. `INFY.NS`, `HDFCBANK.NS`
- Indian BSE stocks: append `.BO` — e.g. `INFY.BO`
- US stocks: no suffix — e.g. `AAPL`, `MSFT`, `NVDA`

Weights must sum to `1.0`. The app normalises them automatically if they are close but not exact.

### Example Queries

```
How did my portfolio perform over the last year?
What is my Sharpe ratio?
Show me my worst drawdown and when it happened.
How does my portfolio compare against the Nifty 50?
Has my portfolio gotten riskier recently? Show rolling volatility.
Am I too concentrated in any sector?
How correlated are my holdings?
How would my portfolio have performed during the 2008 financial crisis?
What if I removed RELIANCE and added HDFCBANK at 25%?
Compare my modified portfolio against the original.
Go back to my original portfolio.
```

---

## Project Structure

```
AI-Portfolio-Analyzer/
├── backend/
│   ├── main.py                    # FastAPI app, /analyze endpoint
│   ├── agent.py                   # LangChain agent, tool routing, system prompt
│   └── tools/
│       ├── returns.py             # Historical return calculations
│       ├── risk.py                # Sharpe, drawdown, VaR, beta, rolling metrics
│       ├── benchmark.py           # Index benchmark comparison
│       ├── concentration.py       # Sector / factor / asset class breakdown
│       ├── correlation.py         # Pairwise correlation matrix
│       ├── portfolio_info.py      # Holdings composition
│       ├── portfolio_modification.py  # What-if simulation engine
│       ├── portfolio_comparison.py    # Original vs modified comparison
│       └── suggestions.py         # LLM-generated smart suggestions
├── frontend/
│   └── app.py                     # Streamlit UI, canvas, chat
├── sample_portfolio.csv
├── requirements.txt
└── .env.example
```

---

## API Reference

### `POST /analyze`

```json
{
  "tickers": ["RELIANCE.NS", "TCS.NS"],
  "weights": [0.6, 0.4],
  "period": "1y",
  "question": "What is my Sharpe ratio?",
  "accumulated_analysis": {},
  "is_initial": false,
  "original_tickers": [],
  "original_weights": []
}
```

**Response**
```json
{
  "response": "Your portfolio's Sharpe ratio is 1.24, meaning...",
  "visualization": { "type": "metrics", "data": { ... } },
  "suggestions": [ { "text": "...", "query": "...", "priority": "high" } ],
  "analysis_updates": {},
  "portfolio_update": null
}
```

### `GET /health`

Returns `{ "status": "ok" }`. Use for liveness checks.

---

## Contributing

Pull requests are welcome. For significant changes, please open an issue first to discuss the approach.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to the branch and open a Pull Request

---

## License

MIT
