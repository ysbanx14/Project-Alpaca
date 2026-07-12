import threading
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.live.stock import StockDataStream
from alpaca.data.enums import DataFeed



class AlpacaDataConnector:
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        
        # Initialize historical data client
        self.historical_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        
        # State for live stream
        self.stream = None
        self.stream_thread = None
        self.loop = None
        
        self.realtime_data = {
            "bid": 0.0,
            "ask": 0.0,
            "last": 0.0
        }
        self.current_symbol = None

    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Fetches historical 5-minute bars for the given symbol over the past `days` days.
        """
        import streamlit as st
        from datetime import timezone
        
        try:
            # Ensure datetimes are timezone-aware (Alpaca requires this)
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days)
            
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start_time,
                end=end_time,
                feed=DataFeed.IEX  # Explicitly specify IEX data feed
            )
            
            bars = self.historical_client.get_stock_bars(request_params)
            df = bars.df
            
            if df.empty:
                return pd.DataFrame()
                
            # Drop the 'symbol' level from multi-index if it exists
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level='symbol')
                
            return df
        except Exception as e:
            # Pass the error string back to the UI explicitly
            st.error(f"API Error: {e}")
            return pd.DataFrame()

    async def _quote_handler(self, quote):
        """Callback to handle incoming real-time quotes."""
        if quote.symbol == self.current_symbol:
            self.realtime_data["bid"] = quote.bid_price
            self.realtime_data["ask"] = quote.ask_price

    async def _trade_handler(self, trade):
        """Callback to handle incoming real-time trades."""
        if trade.symbol == self.current_symbol:
            self.realtime_data["last"] = trade.price

    def _run_stream(self):
        """Runs the WebSocket stream in a dedicated asyncio event loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        self.stream = StockDataStream(self.api_key, self.secret_key)
        
        if self.current_symbol:
            self.stream.subscribe_quotes(self._quote_handler, self.current_symbol)
            self.stream.subscribe_trades(self._trade_handler, self.current_symbol)
            
        try:
            self.stream.run()
        except Exception as e:
            print(f"Stream error: {e}")

    def update_symbol(self, symbol: str):
        """
        Updates the active symbol for real-time streaming.
        Unsubscribes from the old symbol and subscribes to the new one to avoid connection limits.
        """
        if not symbol:
            return
            
        symbol = symbol.upper()
        if symbol == self.current_symbol:
            return
            
        # Reset realtime data for new symbol
        self.realtime_data = {"bid": 0.0, "ask": 0.0, "last": 0.0}
        
        # If stream isn't running, start it
        if self.stream is None:
            self.current_symbol = symbol
            self.stream_thread = threading.Thread(target=self._run_stream, daemon=True)
            self.stream_thread.start()
        else:
            # Unsubscribe from old symbol
            if self.current_symbol:
                try:
                    self.stream.unsubscribe_quotes(self.current_symbol)
                    self.stream.unsubscribe_trades(self.current_symbol)
                except Exception as e:
                    print(f"Error unsubscribing from {self.current_symbol}: {e}")
            
            self.current_symbol = symbol
            
            # Subscribe to new symbol
            try:
                self.stream.subscribe_quotes(self._quote_handler, self.current_symbol)
                self.stream.subscribe_trades(self._trade_handler, self.current_symbol)
            except Exception as e:
                print(f"Error subscribing to {self.current_symbol}: {e}")

    def get_realtime_data(self) -> dict:
        """Returns the most recently received quote and trade prices."""
        return self.realtime_data
