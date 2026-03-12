import numpy as np
import pandas as pd
from utils.helpers import is_jp_ticker, jp_lot_round


class PositionManager:
    def __init__(self, total_capital: float, max_position_pct: float = 0.10,
                 stop_loss_pct: float = 0.08):
        self.total_capital = total_capital
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct

    def calculate_position_size(self, ticker: str, current_price: float,
                                 risk_per_trade_pct: float = 0.02) -> dict:
        risk_amount = self.total_capital * risk_per_trade_pct
        max_position_value = self.total_capital * self.max_position_pct

        shares_by_risk = risk_amount / (current_price * self.stop_loss_pct)
        position_value = shares_by_risk * current_price

        if position_value > max_position_value:
            shares_by_risk = max_position_value / current_price

        if is_jp_ticker(ticker):
            shares = jp_lot_round(shares_by_risk)
        else:
            shares = int(shares_by_risk)

        actual_value = shares * current_price
        stop_loss_price = current_price * (1 - self.stop_loss_pct)

        return {
            "ticker": ticker,
            "shares": shares,
            "position_value": actual_value,
            "position_pct": actual_value / self.total_capital,
            "stop_loss_price": stop_loss_price,
            "risk_amount": shares * current_price * self.stop_loss_pct,
        }

    def atr_stop_loss(self, ohlcv: pd.DataFrame, multiplier: float = 2.0,
                      period: int = 14) -> float:
        high = ohlcv["High"]
        low = ohlcv["Low"]
        close = ohlcv["Close"]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean().iloc[-1]
        current_price = close.iloc[-1]
        return current_price - multiplier * atr

    def trailing_stop(self, prices: pd.Series, pct: float = 0.08) -> pd.Series:
        high_watermark = prices.cummax()
        return high_watermark * (1 - pct)

    def portfolio_allocation(self, weights: dict, prices: dict) -> dict:
        allocations = {}
        total_allocated = 0

        for ticker, weight in weights.items():
            if ticker not in prices or weight < 0.001:
                continue

            target_value = self.total_capital * weight
            price = prices[ticker]
            raw_shares = target_value / price

            if is_jp_ticker(ticker):
                shares = jp_lot_round(raw_shares)
            else:
                shares = int(raw_shares)

            actual_value = shares * price
            total_allocated += actual_value

            allocations[ticker] = {
                "target_weight": weight,
                "shares": shares,
                "actual_value": actual_value,
                "actual_weight": actual_value / self.total_capital,
                "deviation": abs(actual_value / self.total_capital - weight),
            }

        allocations["_summary"] = {
            "total_allocated": total_allocated,
            "cash_remaining": self.total_capital - total_allocated,
            "utilization": total_allocated / self.total_capital,
        }

        return allocations
