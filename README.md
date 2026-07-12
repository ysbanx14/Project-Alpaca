# Quantitative ML Trading System

> An interactive, stateful machine learning execution terminal built for sophisticated quantitative portfolio management.

This application is a complete, end-to-end algorithmic trading platform. It seamlessly bridges the gap between rigorous quantitative research and active portfolio management, allowing users to backtest machine learning strategies and dynamically deploy them into a stateful, live paper-trading environment via the Alpaca API.

---

## Assignment Compliance Check

This system has been engineered to strictly satisfy all project requirements:

| Requirement | Implementation & Compliance Details |
| :--- | :--- |
| **5 Years of Historical Data** | Fetches exactly 5 years of daily, **split-adjusted** IEX data via Alpaca to prevent artificial drawdowns on corporate actions. |
| **Ticker Selection** | Users can dynamically input any US equity ticker via the interactive Streamlit sidebar. |
| **6 Technical Indicators** | Engineered using pure native Pandas math (avoiding library deprecation issues). Covers Momentum (RSI, MACD), Volatility (Bollinger Bands), Volume (OBV), and Trend (SMA_50, SMA_200). |
| **PCA Dimensionality Reduction** | Implements `StandardScaler` followed by PCA, dynamically keeping only the components necessary to explain **>= 80%** of the cumulative variance. |
| **Trained Classifier** | Utilizes a strictly regularized `GradientBoostingClassifier` (restricted `max_depth`, `n_estimators`, `subsample`) to combat overfitting. While the backtester uses an 80/20 split, the live deployed model is trained on 100% of the historical data to capture current market regimes. |
| **0.6 Probability Barrier** | Strict threshold enforcement: Signals trigger a Long (1) only if `P(Upward Move) > 0.6`, else Flat (0). |
| **Fixed 5% Fractional Sizing** | Trade execution enforces a fixed 5% fractional allocation per position. The system explicitly verifies available buying power before execution to prevent API rejections. |
| **Stop-Loss & Take-Profit** | Enforces hard risk limits by attaching Bracket Orders (Stop-Loss at -5%, Take-Profit at +10%) to all entries. |
| **$100k Initial Capital Backtest** | Vectorized OOS backtesting engine evaluates the ML strategy against a Buy & Hold baseline starting with exactly $100,000. |
| **7 Core Performance Metrics** | Reports Total Return, Annualized Return, Max Drawdown, Sharpe Ratio, Sortino Ratio, Win Rate, and Total Trades. |
| **Live Paper Execution** | Connects to Alpaca's Paper API to execute orders automatically, utilizing the non-blocking background Autopilot engine. |
| **Logging** | Integrates native Python file-based logging (`trading_system.log`) to track data fetches, network errors, rejected orders, and successful fills. |
| **Mandatory Warning String** | UI explicitly states: *"PAPER TRADING ONLY - NO REAL MONEY. This will execute an API call to Alpaca Paper Trading."* |

---

## System Architecture & Modular File Overview

The codebase is strictly modularized into distinct operational layers:

1. **`main_3.py` (The Execution Terminal)**
   - The Streamlit front-end featuring a dynamic, multi-view layout: **"Research & Deploy"** for backtesting, **"Live Portfolio Tracking"** for monitoring active deployments, and **"Auto-Pilot Logs"** for real-time execution status tracking.
2. **`data_pipeline.py` (Data Engineering)**
   - Interfaces with the Alpaca API for fetching 5-year split-adjusted bars. Handles all native Pandas technical indicator calculations and target variable alignment. Records detailed data fetch events to `trading_system.log`.
3. **`ml_model.py` (The Quant Brain)**
   - Manages the entire Scikit-Learn pipeline. Scales features, captures >80% variance via PCA, and trains the robust `GradientBoostingClassifier`.
4. **`backtester.py` (Performance Evaluation)**
   - A highly optimized, vectorized backtesting engine that simulates historical equity curves and computes the 7 core quantitative metrics.
5. **`paper_trader.py` (Order Execution Engine)**
   - Interfaces with Alpaca's Paper Trading client. Computes live integer quantities and submits bracket orders (Stop-Loss / Take-Profit) directly to the exchange. Logs all network responses and filled orders to `trading_system.log`.
6. **`portfolio_manager.py` (State Preservation Layer)**
   - Utilizes a local JSON ledger (`tracked_tickers.json`) to persist an unlimited number of deployed models across browser refreshes and app restarts.

---

## Quantitative Strategy & Risk Controls

### Financial Intuition & Market Behavior
- **Behaviors Exploited:** The strategy targets a composite of Momentum (identifying established trends via SMA and MACD), Mean-Reversion (exploiting price overreactions via RSI and Bollinger Bands), and Volume Confirmation (tracking "smart money" accumulation via On-Balance Volume).
- **Generating Returns:** Traditional rigid rules often fail due to market noise. This strategy generates returns by using a Gradient Boosting Classifier to identify statistically significant, non-linear interactions between these indicators. By enforcing a 0.60 probability barrier, capital is only deployed when these multi-dimensional states align to create a definitive mathematical edge.

### Gradient Boosting Classifier
The core predicting engine utilizes a Scikit-Learn `GradientBoostingClassifier`. After reducing noise and multi-collinearity through PCA, the model targets predicting $T+1$ positive returns.
- **Regularization:** The model enforces strict hyperparameters (shallow `max_depth`, limited `n_estimators`, and stochastic `subsample`) to actively combat overfitting on historical data, ensuring out-of-sample robustness.
- **Confidence Barrier:** A signal is only executed as a Long (1) if the ensemble predicts a probability of an upward move strictly greater than **0.60**. Any prediction at or below 0.60 defaults to Flat (0).

### Risk Management (Bracket Orders & Sizing)
The system algorithmically enforces risk limits rather than relying on human intervention:
1. **Capital Allocation:** The system enforces a strict Fixed Fractional sizing model, allocating exactly **5%** of total account equity per trade. It gracefully skips trades if the required cash is already tied up in existing positions.
2. **Take-Profit (+10%):** A Take-Profit limit order is simultaneously attached to the entry, automatically closing the position in profit if the asset appreciates by 10%.
3. **Stop-Loss (-5%):** A Stop-Loss order is attached to strictly cap downside exposure, automatically exiting the position if the asset depreciates by 5%.

---

## Installation & Environment Configuration

### 1. Install Dependencies
Ensure you have Python 3.9+ installed, then install the required dependencies using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
The system requires an Alpaca Trading API account (Paper). In the root directory of the project, create a file named `.env` and populate it with your keys:

```env
# .env
APCA_API_KEY_ID=your_alpaca_paper_key_here
APCA_API_SECRET_KEY=your_alpaca_paper_secret_here
```
*(Alternatively, keys can be securely entered directly via the Streamlit sidebar upon launch).*

---

## Workflow & Operational Manual

Launch the application terminal by running:
```bash
streamlit run main_3.py
```

### Phase 1: Research & Deploy
1. Open the **Navigation** sidebar and select **Research & Deploy**.
2. Input a target ticker (e.g., `AMZN`) and click **Run Pipeline**.
3. Review the Model Diagnostics (PCA variance capture) and the Backtest Results (Log-scaled Equity Curves and the 7 Core Metrics).
4. If the Out-Of-Sample performance is satisfactory, click **Deploy ML Strategy** to permanently save the ticker to your live portfolio state. (You may deploy an unlimited number of tickers).

### Phase 2: Live Portfolio Tracking
1. Switch the sidebar navigation to **Live Portfolio Tracking**.
2. Select any deployed ticker from the dropdown to instantly view its live Alpaca position metrics (Shares, Avg Entry, Unrealized P&L), today's actionable ML probability, and historical trade execution logs.
3. To liquidate the position and remove it from the ledger, click **Stop Tracking**.

### Phase 3: The Autopilot Engine & Execution Board
The application features a non-blocking, dedicated background daemon engine.
1. Turn the **Enable Auto-Pilot Trading** toggle to True to activate the background daemon.
2. The Autopilot engine executes strictly once per day at 3:55 PM EST. This synchronizes the live execution environment with the daily closing price assumptions used during the model's training and backtesting phases, eliminating intraday data leakage and whipsaw risks.
3. Navigate to the **Auto-Pilot** tab to monitor the live Execution Status Board. This Pandas DataFrame natively visualizes the background thread's progress, displaying row-by-row updates of the Alpaca API responses.


### Manual Testing & Override (For Video Demos)
If you need to guarantee an API execution for a demonstration (or if you wish to override the ML prediction):
1. Navigate to the **Live Portfolio Tracking** tab.
2. Open the **Manual Trade Controls (Troubleshooting)** expander.
3. Use the **Force Market Buy** or **Force Market Sell** buttons to bypass the ML logic and explicitly force an order through the Alpaca API.
