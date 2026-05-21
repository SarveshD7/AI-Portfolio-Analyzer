Kalpi AI Portfolio Analyzer
An intelligent portfolio analysis platform that transforms raw investment data into actionable insights through conversational AI and dynamic visualizations.

🎯 Overview
The Kalpi AI Portfolio Analyzer is a production-ready AI agent system that provides institutional-grade portfolio analysis through natural language conversation. Built with LangGraph for agentic orchestration, it demonstrates advanced AI engineering patterns including multi-tool routing, stateful what-if simulations, and context-aware suggestions.
Live Demo: [Add deployment link]
Demo Video: [Add Loom link]

✨ Key Features
🤖 Conversational AI Analysis

Natural language queries for complex financial analysis
LangGraph-powered agent orchestration with intelligent tool routing
Multi-step reasoning for comprehensive portfolio insights
Context-aware responses based on actual calculated data

📊 Deep Financial Analysis
Return Performance

Historical returns across multiple timeframes (1mo - 5y)
Benchmark comparison vs Nifty 50
Stock-level performance breakdown
Best/worst performer identification

Risk & Vulnerabilities

Sharpe ratio (risk-adjusted returns)
Maximum drawdown analysis
Value at Risk (VaR) at 95% and 99% confidence
Tail risk metrics (CVaR)
Volatility tracking

Diversification & Exposure

Sector concentration analysis
Portfolio composition breakdown
Factor exposure (large-cap, value, momentum)
Concentration risk scoring

🔄 What-If Simulations

Natural language portfolio modifications
Chain multiple scenarios seamlessly
Real-time rebalancing with weight normalization
Revert to original portfolio anytime
Modification history tracking

💡 Smart Contextual Suggestions

Proactive question recommendations based on portfolio vulnerabilities
Data-driven pattern detection (sector concentration, low Sharpe, high drawdown)
Click-to-ask interaction
Dynamic updates after each analysis

🎨 Chat + Canvas Interface

Side-by-side layout: conversation on left, visualizations on right
Dynamic canvas updates based on conversation context
Multiple visualization types:

Line charts (returns, drawdowns)
Metric cards (Sharpe, VaR, returns)
Pie charts (sectors, holdings)
Comparison tables (stock breakdown)


Professional fintech dashboard aesthetic


🏗️ Architecture
Core Design Principles
1. LLM as Orchestrator, Not Calculator
User Question
    ↓
LangGraph Agent (decides which tool to call)
    ↓
Deterministic Python Tools (perform calculations)
    ↓
Agent (wraps results in natural language)
    ↓
User Response
The LLM never performs financial calculations or guesses metrics. All numerical analysis is done by deterministic Python functions.
2. Tool-Per-Concern Pattern
Each financial metric has its own isolated, testable tool:

calculate_portfolio_returns() - Returns analysis
calculate_sharpe_ratio() - Risk-adjusted performance
calculate_var() - Tail risk
calculate_max_drawdown() - Downside analysis
get_sector_breakdown() - Diversification
get_portfolio_composition() - Holdings overview
simulate_portfolio_change() - What-if scenarios

3. Stateful Conversation Management
pythonsession_state:
  - original_portfolio: Never modified (uploaded CSV)
  - current_portfolio: Active state (modified by what-ifs)
  - messages: Chat history
  - modification_history: What-if change log
  - accumulated_analysis: All tool results for smart suggestions

🛠️ Tech Stack
Backend

FastAPI - REST API framework
LangGraph - Agentic workflow orchestration
LangChain - Tool calling framework
Groq - LLM inference (Llama 3.3 70B)
yfinance - Market data retrieval
pandas/numpy - Financial calculations

Frontend

Streamlit - Interactive web interface
Plotly - Dynamic visualizations
Custom CSS - Professional styling

Data & Tools

Indian stock market focus (NSE/BSE tickers)
Nifty 50 benchmark comparison
Real-time market data via yfinance API

