import os
import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

logging.basicConfig(
    filename='trading_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PaperTrader")

class PaperTrader:
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        
        # Initialize the trading client securely pointing to the paper environment
        self.trading_client = TradingClient(
            api_key=self.api_key, 
            secret_key=self.secret_key, 
            paper=True
        )

    def execute_trade(self, symbol: str, signal: int, active_models: int = 1):
        """
        Executes a paper trade based on the given signal.
        signal == 1: Long (Market Buy)
        signal == 0: Flat (Liquidate position)
        active_models: The number of active ticker models deployed in the portfolio.
        """
        try:
            # Check current position
            positions = self.trading_client.get_all_positions()
            current_position = next((p for p in positions if p.symbol == symbol), None)
            
            if signal == 1:
                # We want to be Long
                if current_position is None:
                    # Check for pending open buy orders
                    open_orders = self.trading_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
                    if open_orders:
                        return f"SKIPPED: Pending buy order already exists for {symbol}."
                        
                    # Not in position, so we buy
                    account = self.trading_client.get_account()
                    
                    # Calculate fixed 5% allocation
                    trade_amount = float(account.equity) * 0.05
                    
                    # Buying Power Risk Check
                    available_cash = float(account.buying_power)
                    if trade_amount > available_cash:
                        msg = f"SKIPPED: Insufficient buying power to allocate 5% to {symbol}. Waiting for existing positions to close."
                        logger.warning(msg)
                        return msg

                    if trade_amount >= 10:  # Minimum threshold for a trade
                        try:
                            from alpaca.data.historical import StockHistoricalDataClient
                            from alpaca.data.requests import StockLatestTradeRequest
                            data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
                            trade = data_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=[symbol]))
                            current_price = float(trade[symbol].price)
                            qty = int(trade_amount // current_price)
                            
                            if qty <= 0:
                                msg = f"ERROR: Insufficient funds to buy even 1 share of {symbol} at ${current_price:.2f}."
                                logger.error(msg)
                                return msg
                                
                            take_profit_price = round(current_price * 1.10, 2)
                            stop_loss_price = round(current_price * 0.95, 2)
                            
                            take_profit = TakeProfitRequest(limit_price=take_profit_price)
                            stop_loss = StopLossRequest(stop_price=stop_loss_price)
                            
                            market_order_data = MarketOrderRequest(
                                symbol=symbol,
                                qty=qty,
                                side=OrderSide.BUY,
                                time_in_force=TimeInForce.DAY,
                                take_profit=take_profit,
                                stop_loss=stop_loss
                            )
                            order = self.trading_client.submit_order(order_data=market_order_data)
                            msg = f"SUCCESS: Submitted BUY order for {qty} shares of {symbol} with Bracket (TP: ${take_profit_price:.2f}, SL: ${stop_loss_price:.2f}). Order ID: {order.id}"
                            logger.info(msg)
                            return msg
                        except Exception as error:
                            msg = f"ERROR executing trade for {symbol}: {str(error)}"
                            logger.error(msg)
                            return msg
                    else:
                        return f"SKIPPED: 5% of equity (${trade_amount:.2f}) is below the $10 minimum trade size."
                else:
                    return f"SKIPPED: Already holding a Long position for {symbol}."
            
            elif signal == 0:
                # We want to be Flat
                open_orders = self.trading_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
                
                if current_position is not None and float(current_position.qty) > 0:
                    try:
                        # We hold a position, liquidate it
                        close_info = self.trading_client.close_position(symbol)
                        msg = f"SUCCESS: Liquidated position for {symbol}. Order ID: {close_info.id}"
                        logger.info(msg)
                        return msg
                    except Exception as e:
                        # Alpaca might queue it if market is closed, or reject it.
                        msg = f"API RESPONSE (Sell): {str(e)}"
                        logger.error(msg)
                        return msg
                else:
                    if open_orders:
                        for o in open_orders:
                            self.trading_client.cancel_order_by_id(o.id)
                        msg = f"SUCCESS: Cancelled pending orders for {symbol} to remain flat."
                        logger.info(msg)
                        return msg
                    return f"SKIPPED: Already flat on {symbol}. No position to close."
            
            else:
                return f"ERROR: Invalid signal {signal}. Must be 1 or 0."
                
        except Exception as e:
            return f"ERROR executing trade: {str(e)}"

    def get_position(self, symbol: str) -> dict:
        """Returns current held shares, average entry price, and unrealized P&L."""
        try:
            positions = self.trading_client.get_all_positions()
            pos = next((p for p in positions if p.symbol == symbol), None)
            if pos:
                return {
                    'shares': float(pos.qty),
                    'market_value': float(pos.market_value),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc)
                }
            return None
        except Exception as e:
            return None

    def get_trade_logs(self, symbol: str) -> list:
        """Fetches the closed orders/activities for this specific ticker."""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=[symbol],
                limit=100
            )
            orders = self.trading_client.get_orders(filter=req)
            
            logs = []
            for o in orders:
                logs.append({
                    'created_at': o.created_at.strftime('%Y-%m-%d %H:%M:%S') if o.created_at else 'Unknown',
                    'side': str(o.side.value).upper() if o.side else 'UNKNOWN',
                    'qty': float(o.filled_qty) if o.filled_qty else 0.0,
                    'filled_avg_price': float(o.filled_avg_price) if o.filled_avg_price else 0.0,
                    'status': str(o.status.value).upper() if o.status else 'UNKNOWN'
                })
            return logs
        except Exception as e:
            return []

    def get_portfolio_capital(self) -> dict:
        """
        Dynamically calculates the true total equity and currently allocated capital 
        across all open positions directly from the Alpaca API.
        """
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()
            
            total_equity = float(account.equity)
            allocated_capital = sum(float(pos.market_value) for pos in positions)
            
            return {
                "total_equity": total_equity,
                "allocated_capital": allocated_capital
            }
        except Exception as e:
            return {"total_equity": 100000.0, "allocated_capital": 0.0}
