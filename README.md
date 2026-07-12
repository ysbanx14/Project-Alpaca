# Quantitative ML Trading System

> An algorithmic trading platform designed for systematic, machine-learning-driven portfolio management.

## Project Overview & Goals

This system is a comprehensive quantitative trading platform designed to bridge the gap between rigorous backtesting and active live execution. 

**Operating Mode & Strategy:** 
The application operates **strictly in Alpaca Paper Trading mode** using a non-discretionary, machine-learning-based systematic strategy. It utilizes a Gradient Boosting Classifier to predict daily asset price movements based on technical indicators and dynamically deploys capital without requiring manual human intervention.

---

## Architecture Description

The codebase is engineered with a strict modular architecture, separating data engineering, predictive modeling, backtesting, and live execution into distinct operational layers:

- **`data_pipeline.py` (Data Pipeline):** Interfaces with the Alpaca API to fetch historical market data. It handles all native Pandas technical feature engineering (SMA, MACD, RSI, Bollinger Bands, OBV) and target variable alignment to ensure the model trains on diverse datasets.
- **`ml_model.py` (ML Strategy):** Manages the Scikit-Learn predictive pipeline. It scales features, performs Principal Component Analysis (PCA) for dimensionality reduction, and trains the heavily regularized `GradientBoostingClassifier`.
- **`backtester.py` (Backtester):** A highly optimized, vectorized backtesting engine that simulates historical equity curves and computes core quantitative performance metrics (Total Return, Sharpe Ratio, Max Drawdown, etc.) against a $100k baseline.
- **`paper_trader.py` (Execution Engine):** Interacts with Alpaca’s Paper Trading API to compute live position sizing, execute bracket orders, and handle real-time buying power constraints.
- **`portfolio_manager.py` (State Preservation):** Utilizes a local JSON ledger (`tracked_tickers.json`) to persist an unlimited number of deployed ticker models across browser refreshes and application restarts.
- **`main.py` (Streamlit UI):** The interactive front-end execution terminal. It orchestrates user inputs, visually renders backtest performance and PCA diagnostics, and spawns the background execution daemon for live portfolio trading.
- **`live_data_stream.py`:** Handles localized streaming components and continuous data hooks.

---

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.9+ installed. It is highly recommended to use a virtual environment. Install the required dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Configure Alpaca API Keys (Secure Setup)
This system requires a free Alpaca Trading API account (Paper). **Never hardcode your API keys in the source code or commit them to version control.**

In the root directory of the project, create a hidden file named `.env` and securely populate it with your Alpaca Paper credentials:

```env
# .env
APCA_API_KEY_ID=your_alpaca_paper_key_here
APCA_API_SECRET_KEY=your_alpaca_paper_secret_here
```

### 3. Run the Application
Launch the interactive Streamlit terminal by running:

```bash
streamlit run main.py
```

---

## Strategy Description & Risk Controls

### The Quantitative ML Strategy
- **Data Engineering:** The system fetches exactly 5 years of daily, **split-adjusted** IEX data via Alpaca to prevent artificial drawdowns on corporate actions.
- **Feature Engineering:** Technical indicators covering Momentum (RSI, MACD), Volatility (Bollinger Bands), Volume (OBV), and Trend (SMA_50) are engineered natively to create a multi-dimensional view of the asset.
- **Dimensionality Reduction:** The pipeline implements `StandardScaler` followed by PCA, dynamically keeping only the principal components necessary to explain `>= 80%` of the cumulative variance, effectively reducing noise and multi-collinearity.
- **Predictive Engine:** A strictly regularized `GradientBoostingClassifier` identifies complex, non-linear interactions between the PCA components to predict $T+1$ positive returns.
- **0.60 Probability Barrier:** To prevent over-trading in choppy markets, the system enforces a strict confidence threshold: signals trigger a Long (1) only if the model predicts a probability of an upward move strictly greater than **0.50**. Any prediction at or below 0.50 defaults to Flat (0).

### Hard Risk Controls
The system algorithmically enforces risk management rather than relying on human discretion:
- **Fixed Fractional Sizing:** Capital allocation enforces a strict **Fixed 5% Fractional Sizing** model. It dynamically calculates exactly 5% of the total available account equity for each trade execution.
- **Pre-Trade Constraints:** The execution engine queries the Alpaca account's available buying power *before* submitting orders. If the required 5% allocation exceeds available cash, the system gracefully skips the trade to prevent API rejection errors.
- **Automated Bracket Orders:** Every executed entry is strictly paired with an attached Bracket Order:
  - **Take-Profit (+10%):** Automatically closes the position in profit if the asset appreciates.
  - **Stop-Loss (-5%):** Strictly caps downside exposure, automatically exiting the position if the asset depreciates.

---

## Example Usage Walkthrough

### Phase 1: Research & Deploy
Navigate to the **Research & Deploy** tab. Enter a ticker symbol (e.g., `AAPL`) and run the pipeline. The system will fetch the data, train the ML model, and output the OOS backtest metrics and PCA variance charts. If satisfied with the strategy's statistical edge, click "Deploy ML Strategy" to save the ticker to your local portfolio state.

> <img width="3141" height="1252" alt="Screenshot 2026-07-12 at 6 17 41 PM" src="https://github.com/user-attachments/assets/2b53a45b-f12e-4d46-8cb5-9bb937d9d451" />

### Phase 2: Live Portfolio Tracking
Switch to the **Live Portfolio Tracking** tab. This acts as your command center. Select any deployed ticker from the dropdown to instantly view its live Alpaca position metrics (Shares Held, Avg Entry Price, Unrealized P&L), today's actionable ML probability, and historical trade execution logs.

> <img width="3141" height="1252" alt="Screenshot 2026-07-12 at 6 18 49 PM" src="https://github.com/user-attachments/assets/d9d57b7f-d99b-4457-a05e-35d448e75757" />


### Phase 3: Activating the Auto-Pilot Daemon
Navigate to the **Auto-Pilot** tab and toggle "Enable Auto-Pilot Trading" to True. This activates the non-blocking background execution daemon. The daemon continuously evaluates your deployed models and executes trades strictly once per day at **3:55 PM EST**, synchronizing the live environment with the daily closing price assumptions used during model training. You can monitor the real-time API responses in the live Execution Status Board DataFrame.

> <img width="3141" height="1252" alt="Screenshot 2026-07-12 at 6 19 27 PM" src="https://github.com/user-attachments/assets/ba4e74bc-0199-428c-9eaa-66055043e250" />
