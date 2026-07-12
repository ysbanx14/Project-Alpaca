import os
import traceback
import datetime
from dotenv import load_dotenv
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

import streamlit as st
import time
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load environment variables
load_dotenv()

from data_pipeline import DataPipeline
from ml_model import MLModelPipeline
from backtester import Backtester
from paper_trader import PaperTrader
from portfolio_manager import PortfolioManager
from live_data_stream import AlpacaDataConnector

from alpaca.trading.requests import GetPortfolioHistoryRequest

pm = PortfolioManager()

# Page Configuration
st.set_page_config(page_title="ML Trading System", layout="wide", page_icon="🤖")

if 'execution_status' not in st.session_state:
    st.session_state['execution_status'] = {}

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1CB5E0 0%, #000851 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        border: 1px solid #1CB5E0;
    }
    .disclaimer {
        background-color: rgba(255, 75, 75, 0.1);
        border-left: 5px solid #ff4b4b;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        margin-bottom: 20px;
        font-weight: bold;
        color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Quantitative ML Trading System")

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
st.sidebar.header("Configuration")
env_api_key = os.getenv("APCA_API_KEY_ID")
env_secret_key = os.getenv("APCA_API_SECRET_KEY")

if env_api_key and env_secret_key:
    st.sidebar.success("✅ API Keys loaded securely from .env")
    api_key = env_api_key
    secret_key = env_secret_key
else:
    api_key = st.sidebar.text_input("Alpaca API Key ID", type="password")
    secret_key = st.sidebar.text_input("Alpaca Secret Key", type="password")

# -------------------------------------------------------------
# AUTO-PILOT LOGIC (Background Thread Target)
# -------------------------------------------------------------
def run_portfolio_execution(board_placeholder=None):
    if not api_key or not secret_key:
        return
    tracked = pm.get_tracked_tickers()
    active_models = len(tracked)
    if not tracked:
        return
        
    dp = DataPipeline(api_key, secret_key)
    ml = MLModelPipeline()
    pt = PaperTrader(api_key, secret_key)
    
    tracked_sorted = sorted(tracked)
    
    for t in tracked_sorted:
        if t not in st.session_state['execution_status']:
            st.session_state['execution_status'][t] = {"indicator": "🔴", "message": "-"}
        else:
            st.session_state['execution_status'][t]["indicator"] = "🔴"
            
    if board_placeholder is not None:
        status_df = pd.DataFrame.from_dict(st.session_state['execution_status'], orient='index').reset_index()
        status_df.columns = ["Ticker", "Status", "Latest Action"]
        board_placeholder.dataframe(status_df, use_container_width=True, hide_index=True)
            
    for t in tracked_sorted:
        try:
            time.sleep(0.5)  # Rate Limit Protection
            df = dp.fetch_data(t, years=5)
            if df.empty:
                st.session_state['execution_status'][t]["indicator"] = "🟢"
                st.session_state['execution_status'][t]["message"] = "ERROR - No data"
                if board_placeholder is not None:
                    status_df = pd.DataFrame.from_dict(st.session_state['execution_status'], orient='index').reset_index()
                    status_df.columns = ["Ticker", "Status", "Latest Action"]
                    board_placeholder.dataframe(status_df, use_container_width=True, hide_index=True)
                continue
            df_features = dp.engineer_features(df)
            ml.train(df_features)
            
            latest_df = dp.get_latest_data_for_prediction(t)
            if not latest_df.empty:
                today_signal, _ = ml.predict_today_signal(latest_df)
                res = pt.execute_trade(t, today_signal, active_models=active_models)
                st.session_state['execution_status'][t]["indicator"] = "🟢"
                st.session_state['execution_status'][t]["message"] = res
        except Exception as e:
            st.session_state['execution_status'][t]["indicator"] = "🟢"
            st.session_state['execution_status'][t]["message"] = f"ERROR - Failed to trade: {e}"
            
        if board_placeholder is not None:
            status_df = pd.DataFrame.from_dict(st.session_state['execution_status'], orient='index').reset_index()
            status_df.columns = ["Ticker", "Status", "Latest Action"]
            board_placeholder.dataframe(status_df, use_container_width=True, hide_index=True)


def get_plotly_layout():
    return dict(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )


@st.fragment(run_every=30)
def render_live_market_data(ticker, api_key, secret_key):
    if 'live_connector' not in st.session_state:
        st.session_state['live_connector'] = AlpacaDataConnector(api_key, secret_key)
        
    connector = st.session_state['live_connector']
    connector.update_symbol(ticker)
    
    live_data = connector.get_realtime_data()
    
    l_col1, l_col2, l_col3 = st.columns(3)
    l_col1.metric("Bid Price", f"${live_data['bid']:.2f}")
    l_col2.metric("Ask Price", f"${live_data['ask']:.2f}")
    l_col3.metric("Last Price", f"${live_data['last']:.2f}")

# -------------------------------------------------------------
# MAIN TABS LAYOUT
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Home Dashboard", 
    "Research & Deploy", 
    "Live Portfolio Tracking", 
    "Auto-Pilot", 
    "Models",
    "Troubleshooting"
])

board_placeholder = None
auto_pilot = False
auto_pilot_interval = 5

# ==========================================
# TAB 1: HOME DASHBOARD
# ==========================================
with tab1:
    st.header("🏠 Home Dashboard")
    if not api_key or not secret_key:
        st.warning("Please configure API keys in the sidebar to view account details.")
    else:
        try:
            pt = PaperTrader(api_key, secret_key)
            account = pt.trading_client.get_account()
            portfolio_data = pt.get_portfolio_capital()
            
            total_equity = portfolio_data["total_equity"]
            allocated = portfolio_data["allocated_capital"]
            buying_power = float(account.buying_power)
            
            last_equity = float(account.last_equity)
            daily_pl = total_equity - last_equity
            daily_pl_pct = (daily_pl / last_equity) * 100 if last_equity > 0 else 0.0

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Portfolio Value", f"${total_equity:,.2f}", f"${daily_pl:,.2f} ({daily_pl_pct:.2f}%)")
            col2.metric("Allocated Capital", f"${allocated:,.2f}")
            col3.metric("Buying Power", f"${buying_power:,.2f}")
            
            st.markdown("---")
            st.subheader("Portfolio Equity Curve (1 Month)")
            
            try:
                req = GetPortfolioHistoryRequest(period="1M", timeframe="1D")
                history = pt.trading_client.get_portfolio_history(req)
                hist_df = pd.DataFrame({
                    'timestamp': [pd.to_datetime(t, unit='s') for t in history.timestamp],
                    'equity': history.equity
                })
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(x=hist_df['timestamp'], y=hist_df['equity'], mode='lines', name='Equity', line=dict(color='#00ff99', width=2)))
                fig_hist.update_layout(title="Daily Equity Curve", xaxis_title="Date", yaxis_title="Equity ($)", **get_plotly_layout())
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception as e:
                st.info(f"Could not load portfolio history: {e}")
            
            st.markdown("---")
            st.subheader("Open Positions")
            positions = pt.trading_client.get_all_positions()
            if positions:
                pos_data = []
                for p in positions:
                    pos_data.append({
                        "Ticker": p.symbol,
                        "Qty": float(p.qty),
                        "Market Value": float(p.market_value),
                        "Avg Entry": float(p.avg_entry_price),
                        "Current Price": float(p.current_price),
                        "Unrealized P&L": float(p.unrealized_pl),
                        "Unrealized P&L %": float(p.unrealized_plpc) * 100
                    })
                pos_df = pd.DataFrame(pos_data)
                st.dataframe(pos_df.style.format({
                    'Market Value': '${:,.2f}',
                    'Avg Entry': '${:,.2f}',
                    'Current Price': '${:,.2f}',
                    'Unrealized P&L': '${:,.2f}',
                    'Unrealized P&L %': '{:.2f}%'
                }), use_container_width=True, hide_index=True)
            else:
                st.info("No open positions.")
        except Exception as e:
            st.error(f"Failed to fetch account data: {e}")

# ==========================================
# TAB 2: RESEARCH & DEPLOY
# ==========================================
with tab2:
    st.header("🔬 Research & Deploy")
    col_t, col_b = st.columns([3, 1])
    with col_t:
        ticker = st.text_input("Ticker Symbol", value="AAPL").upper().strip()
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Run Pipeline"):
            st.session_state['run_pipeline_for'] = ticker
            st.session_state['pipeline_results'] = None
            
    if api_key and secret_key and ticker:
        st.markdown("---")
        st.header("📡 Live Market Data")
        render_live_market_data(ticker, api_key, secret_key)

    if st.session_state.get('run_pipeline_for') == ticker:
        if not api_key or not secret_key:
            st.error("Please enter both Alpaca API Key and Secret Key in the sidebar.")
        else:
            try:
                # 1. Fetch or Retrieve Pipeline Results
                if not st.session_state.get('pipeline_results') or st.session_state.get('pipeline_results_ticker') != ticker:
                    with st.spinner("Fetching Data and Training Model..."):
                        dp = DataPipeline(api_key, secret_key)
                        raw_df = dp.fetch_data(ticker, years=5)
                        if raw_df.empty:
                            st.error(f"No data returned for {ticker}.")
                            st.stop()
                            
                        df_features = dp.engineer_features(raw_df)
                        split_idx = int(len(df_features) * 0.8)
                        train_df = df_features.iloc[:split_idx]
                        test_df = df_features.iloc[split_idx:]
                        
                        ml = MLModelPipeline()
                        ml.train(train_df)
                        
                        signals = ml.generate_signals(df_features)
                        test_signals = ml.generate_signals(test_df)
                        
                        bt = Backtester(initial_capital=100000.0)
                        _, _, bt_df = bt.run(df_features, signals)
                        oos_strat_metrics, oos_bh_metrics, _ = bt.run(test_df, test_signals)
                        
                        st.session_state['pipeline_results'] = (ml, dp, df_features, test_df, bt_df, oos_strat_metrics, oos_bh_metrics)
                        st.session_state['pipeline_results_ticker'] = ticker
                else:
                    ml, dp, df_features, test_df, bt_df, oos_strat_metrics, oos_bh_metrics = st.session_state['pipeline_results']
                    
                # 2. Backtest Results
                st.markdown("---")
                st.header("📊 Backtest Results")
                st.subheader("Out-Of-Sample Performance (Last 20% of data)")
                metrics_df = pd.DataFrame([oos_strat_metrics, oos_bh_metrics], index=["ML Strategy (OOS)", "Buy & Hold (OOS)"]).T
                st.dataframe(metrics_df.style.highlight_max(axis=1, color='#2e4034'), use_container_width=True)
                
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(x=bt_df.index, y=bt_df['strategy_equity'], mode='lines', name='ML Strategy', line=dict(color='#1CB5E0', width=2)))
                fig_equity.add_trace(go.Scatter(x=bt_df.index, y=bt_df['buy_hold_equity'], mode='lines', name='Buy & Hold', line=dict(color='#ff9900', width=2, dash='dot')))
                fig_equity.update_layout(title=f"Full Equity Curve Comparison ({ticker})", xaxis_title="Date", yaxis_title="Portfolio Value ($)", legend=dict(x=0.01, y=0.99), **get_plotly_layout())
                fig_equity.update_yaxes(type="log", title_text="Portfolio Value (Log Scale, $)")
                
                split_date = test_df.index[0]
                fig_equity.add_vline(x=split_date, line_width=2, line_dash="dash", line_color="red")
                fig_equity.add_annotation(x=split_date, y=1.05, yref="paper", text="← In-Sample | Out-Of-Sample →", showarrow=False, xanchor="left", font=dict(color="red"))
                st.plotly_chart(fig_equity, use_container_width=True)
                
                st.subheader("Out-Of-Sample Normalized Comparison")
                oos_df = bt_df.loc[split_date:].copy()
                start_strat = oos_df['strategy_equity'].iloc[0]
                start_bh = oos_df['buy_hold_equity'].iloc[0]
                oos_strat_rebased = (oos_df['strategy_equity'] / start_strat) * 100000.0
                oos_bh_rebased = (oos_df['buy_hold_equity'] / start_bh) * 100000.0
                fig_oos = go.Figure()
                fig_oos.add_trace(go.Scatter(x=oos_df.index, y=oos_strat_rebased, mode='lines', name='ML Strategy (OOS)', line=dict(color='#1CB5E0', width=2)))
                fig_oos.add_trace(go.Scatter(x=oos_df.index, y=oos_bh_rebased, mode='lines', name='Buy & Hold (OOS)', line=dict(color='#ff9900', width=2, dash='dot')))
                fig_oos.update_layout(title=f"Out-Of-Sample Equity Curve ({ticker})", xaxis_title="Date", yaxis_title="Portfolio Value (Linear Scale, $)", legend=dict(x=0.01, y=0.99), **get_plotly_layout())
                st.plotly_chart(fig_oos, use_container_width=True)
                        
                # Live Paper Trading -> Deploy Strategy
                st.markdown("---")
                st.header("⚡ Deploy Strategy")
                        
                latest_df = dp.get_latest_data_for_prediction(ticker)
                if not latest_df.empty:
                    today_signal, today_prob = ml.predict_today_signal(latest_df)
                    signal_text = "🟢 LONG (Buy)" if today_signal == 1 else "⚪ FLAT (Sell/Cash)"
                    prob_pct = today_prob * 100
                    st.markdown(f"### Current Signal for {ticker}: **{signal_text}**")
                    st.progress(today_prob, text=f"Probability of Upward Move: {prob_pct:.2f}%")
                    st.markdown('<div class="disclaimer">Deploying this model will save it to the tracked tickers list for live portfolio execution.</div>', unsafe_allow_html=True)
                    
                    if st.button(f"🟢 Deploy ML Strategy for {ticker}"):
                        if pm.add_ticker(ticker):
                            st.success(f"{ticker} has been successfully deployed to the Live Portfolio!")
                        else:
                            st.info(f"{ticker} is already deployed.")
                else:
                    st.warning("Could not fetch latest data for real-time prediction.")
            except Exception as e:
                st.error(f"An error occurred during pipeline execution: {str(e)}")
                st.code(traceback.format_exc(), language="text")

# ==========================================
# TAB 3: LIVE PORTFOLIO TRACKING
# ==========================================
with tab3:
    st.header("📈 Live Portfolio Tracking")
    tracked = pm.get_tracked_tickers()
    
    if not tracked:
        st.warning("You have not deployed any ML strategies yet. Go to 'Research & Deploy' to add tickers.")
    elif not api_key or not secret_key:
        st.warning("API Keys missing.")
    else:
        selected_ticker = st.selectbox("Select Deployed Ticker", tracked)
        
        if selected_ticker:
            pt = PaperTrader(api_key, secret_key)
            dp = DataPipeline(api_key, secret_key)
            ml = MLModelPipeline()
            
            with st.spinner(f"Loading live data for {selected_ticker}..."):
                st.subheader(f"Current Position: {selected_ticker}")
                pos = pt.get_position(selected_ticker)
                if pos:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Shares Held", pos['shares'])
                    col2.metric("Avg Entry Price", f"${pos['avg_entry_price']:.2f}")
                    col3.metric("Market Value", f"${pos['market_value']:.2f}")
                    col4.metric("Unrealized P&L", f"${pos['unrealized_pl']:.2f}", f"{pos['unrealized_plpc']*100:.2f}%")
                else:
                    st.info(f"No active position held for {selected_ticker}.")
                
                st.markdown("---")
                
                st.subheader("Today's ML Signal")
                try:
                    df = dp.fetch_data(selected_ticker, years=5)
                    if not df.empty:
                        df_features = dp.engineer_features(df)
                        split_idx = int(len(df_features) * 0.8)
                        ml.train(df_features.iloc[:split_idx])
                        latest_df = dp.get_latest_data_for_prediction(selected_ticker)
                        today_signal, today_prob = ml.predict_today_signal(latest_df)
                        signal_text = "🟢 LONG (Buy)" if today_signal == 1 else "⚪ FLAT (Sell/Cash)"
                        st.write(f"**Action:** {signal_text} (Probability: {today_prob*100:.2f}%)")
                except Exception as e:
                    st.error(f"Error calculating signal: {e}")
                
                st.markdown("---")
                
                st.subheader("Recent Alpaca Trade Logs")
                logs = pt.get_trade_logs(selected_ticker)
                if logs:
                    log_df = pd.DataFrame(logs)
                    st.dataframe(log_df, use_container_width=True)
                else:
                    st.info("No historical trades found for this ticker.")

# ==========================================
# TAB 4: AUTO-PILOT
# ==========================================
with tab4:
    st.header("🤖 Auto-Pilot Configuration & Logs")
    tracked = pm.get_tracked_tickers()
    
    st.subheader("Deployed Models")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        new_ticker = st.text_input("Manually Add Ticker").upper().strip()
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add Ticker"):
            if new_ticker:
                if pm.add_ticker(new_ticker):
                    st.success(f"{new_ticker} added to autopilot!")
                    st.rerun()
                else:
                    st.info(f"{new_ticker} is already deployed.")
            else:
                st.warning("Please enter a ticker symbol.")
                
    if tracked:
        st.write("**Currently Tracked:** " + ", ".join(tracked))
        tickers_to_remove = st.multiselect("Select Tickers to Remove", tracked)
        if st.button("🗑️ Remove Selected Tickers"):
            if tickers_to_remove:
                for t in tickers_to_remove:
                    pm.remove_ticker(t)
                st.success(f"Removed {len(tickers_to_remove)} tickers.")
                st.rerun()
            else:
                st.warning("Please select at least one ticker to remove.")
    else:
        st.write("No models deployed yet.")
    
    st.markdown("---")
    st.subheader("Configuration")
    auto_pilot = st.toggle("🤖 Enable Auto-Pilot Trading", value=False)
    
    if auto_pilot:
        st.warning("Auto-Pilot is ACTIVE: Scanning and trading every 5 minutes.")
        
    st.markdown("---")
    st.subheader("🖨️ Auto-Pilot Execution Board")
    
    if not st.session_state.get('execution_status'):
        st.info("No executions recorded yet. Deploy models to begin tracking.")
    else:
        board_placeholder = st.empty()
        status_df = pd.DataFrame.from_dict(st.session_state['execution_status'], orient='index')
        status_df.reset_index(inplace=True)
        status_df.columns = ["Ticker", "Status", "Latest Action"]
        board_placeholder.dataframe(status_df, use_container_width=True, hide_index=True)

# ==========================================
# TAB 5: MODELS
# ==========================================
with tab5:
    st.header("🧠 Model Diagnostics (PCA & Features)")
    st.write("This section provides a static overview of the base ML model configuration and PCA feature reduction. It uses `SPY` as a baseline reference dataset.")
    
    if api_key and secret_key:
        if 'models_diagnostics_rendered' not in st.session_state:
            with st.spinner("Generating PCA baseline..."):
                try:
                    dp_models = DataPipeline(api_key, secret_key)
                    raw_spy = dp_models.fetch_data("SPY", years=1)
                    if not raw_spy.empty:
                        spy_features = dp_models.engineer_features(raw_spy)
                        ml_models = MLModelPipeline()
                        ml_models.train(spy_features)
                        st.session_state['models_diagnostics_rendered'] = (ml_models.n_components_kept, len(ml_models._get_feature_cols(spy_features)), ml_models.cumulative_variance_)
                except Exception as e:
                    st.error(f"Failed to fetch SPY for PCA baseline: {e}")
        
        if 'models_diagnostics_rendered' in st.session_state:
            comp_kept, total_feat, cum_var = st.session_state['models_diagnostics_rendered']
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.info("**Gradient Boosting Classifier**")
                st.metric("PCA Components Kept (>= 80% Var)", comp_kept)
                st.metric("Total Original Features", total_feat)
            with col2:
                fig_pca = go.Figure()
                fig_pca.add_trace(go.Scatter(y=cum_var, mode='lines+markers', name='Cumulative Variance', line=dict(color='#00ff99', width=2)))
                fig_pca.add_hline(y=0.80, line_dash="dash", line_color="red", annotation_text="80% Threshold")
                fig_pca.update_layout(title="PCA Cumulative Explained Variance", xaxis_title="Number of Components", yaxis_title="Cumulative Variance Explained", **get_plotly_layout())
                st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.warning("Provide API keys in the sidebar to generate the diagnostics overview.")

# ==========================================
# TAB 6: TROUBLESHOOTING
# ==========================================
with tab6:
    st.header("🔧 Troubleshooting & Manual Override")
    
    st.subheader("Mass Deploy")
    st.write("Deploy strategy to top liquid assets instantly:")
    if st.button("🚀 Deploy to Top 50 Liquid Stocks"):
        top_50 = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "META", "GOOGL", "AMZN", "NFLX", "SPY", "QQQ", "BA", "DIS", "JPM", "V", "MA", "WMT", "JNJ", "PG", "HD", "UNH", "XOM", "CVX", "ABBV", "PFE", "KO", "PEP", "MRK", "TMO", "COST", "CSCO", "MCD", "NKE", "CRM", "ADBE", "TXN", "AVGO", "QCOM", "INTC", "IBM", "HON", "AMGN", "CAT", "GE", "MMM", "GS", "MS", "BLK", "UBER"]
        added_count = 0
        for t_mass in top_50:
            if pm.add_ticker(t_mass):
                added_count += 1
        st.success(f"Successfully deployed {added_count} new tickers to the Live Portfolio!")
        
    st.markdown("---")
    st.warning("Use these controls to bypass the ML logic or clear system state.")
    
    st.subheader("Manual Trade Controls")
    tracked = pm.get_tracked_tickers()
    
    if not api_key or not secret_key:
        st.error("API Keys missing.")
    elif not tracked:
        st.info("No models deployed to troubleshoot.")
    else:
        ts_ticker = st.selectbox("Select Ticker to Override", tracked, key="ts_ticker")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🟢 Force Market Buy"):
                with st.spinner("Forcing API Trade Execution..."):
                    pt = PaperTrader(api_key, secret_key)
                    res = pt.execute_trade(ts_ticker, 1, active_models=len(tracked))
                    if "SUCCESS" in res:
                        st.success(res)
                    else:
                        st.error(res)
        with col2:
            if st.button("🔴 Force Market Sell"):
                with st.spinner("Forcing API Trade Execution..."):
                    pt = PaperTrader(api_key, secret_key)
                    res = pt.execute_trade(ts_ticker, 0, active_models=len(tracked))
                    if "SUCCESS" in res:
                        st.success(res)
                    else:
                        st.error(res)
        with col3:
            if st.button(f"🛑 Stop Tracking {ts_ticker}"):
                pt = PaperTrader(api_key, secret_key)
                res = pt.execute_trade(ts_ticker, 0, active_models=len(tracked))
                pm.remove_ticker(ts_ticker)
                st.success(f"{ts_ticker} removed. Liquidating position: {res}")
                
    st.markdown("---")
    st.subheader("System Actions")
    if st.button("🗑️ Clear Auto-Pilot Board"):
        st.session_state['execution_status'] = {}
        st.success("Board cleared. Refresh to update.")

# ==========================================
# AUTO-PILOT THREAD SPAWNER
# ==========================================
def autopilot_daemon_loop(board_ph):
    import pytz
    import datetime
    import time
    est = pytz.timezone('US/Eastern')
    while True:
        now = datetime.datetime.now(est)
        # Execute exactly at 3:55 PM EST, Monday - Friday
        if now.hour == 15 and now.minute == 55 and now.weekday() < 5:
            if st.session_state.get('last_autopilot_run_date') != now.date():
                run_portfolio_execution(board_ph)
                st.session_state['last_autopilot_run_date'] = now.date()
        time.sleep(30)

if auto_pilot:
    if 'autopilot_daemon_running' not in st.session_state:
        board_ph = board_placeholder if 'board_placeholder' in locals() else None
        t = threading.Thread(target=autopilot_daemon_loop, args=(board_ph,), daemon=True)
        add_script_run_ctx(t)
        t.start()
        st.session_state['autopilot_daemon_running'] = True
