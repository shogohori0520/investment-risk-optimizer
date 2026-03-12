import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from core.portfolio import PortfolioOptimizer
from core.risk import RiskAnalyzer
from utils.helpers import setup_logging

logger = setup_logging()


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    trades: List[Dict] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    weights_history: Optional[pd.DataFrame] = None


class BacktestEngine:
    def __init__(self, returns: pd.DataFrame, weights: Dict[str, float],
                 rebalance_freq: str = "quarterly",
                 initial_capital: float = 10_000_000,
                 risk_free_rate: float = 0.005):
        self.returns = returns
        self.weights = weights
        self.rebalance_freq = rebalance_freq
        self.initial_capital = initial_capital
        self.rf = risk_free_rate

    def _get_rebalance_dates(self) -> List[pd.Timestamp]:
        freq_map = {
            "monthly": "MS",
            "quarterly": "QS",
            "annually": "YS",
        }
        freq = freq_map.get(self.rebalance_freq, "QS")
        dates = pd.date_range(
            self.returns.index[0], self.returns.index[-1], freq=freq
        )
        return [d for d in dates if d in self.returns.index or
                self.returns.index.searchsorted(d) < len(self.returns.index)]

    def run(self) -> BacktestResult:
        tickers = [t for t in self.weights if t in self.returns.columns]
        if not tickers:
            raise ValueError("No matching tickers between weights and returns")

        w = np.array([self.weights.get(t, 0) for t in tickers])
        w = w / w.sum()  # normalize

        ret = self.returns[tickers]
        rebalance_dates = self._get_rebalance_dates()

        portfolio_values = [self.initial_capital]
        current_weights = w.copy()
        trades = []

        for i in range(len(ret)):
            date = ret.index[i]
            daily_ret = ret.iloc[i].values

            # Apply daily returns to current weights
            asset_values = current_weights * portfolio_values[-1]
            asset_values = asset_values * (1 + daily_ret)
            new_total = asset_values.sum()
            portfolio_values.append(new_total)

            # Update weights based on drift
            if new_total > 0:
                current_weights = asset_values / new_total
            else:
                current_weights = w.copy()

            # Rebalance check
            if date in rebalance_dates or (
                len(rebalance_dates) > 0 and
                any(abs((date - rd).days) <= 1 for rd in rebalance_dates)
            ):
                trades.append({
                    "date": date,
                    "type": "rebalance",
                    "weights_before": dict(zip(tickers, current_weights)),
                    "weights_after": dict(zip(tickers, w)),
                    "portfolio_value": new_total,
                })
                current_weights = w.copy()

        equity = pd.Series(portfolio_values[1:], index=ret.index, name="portfolio")
        daily_returns = equity.pct_change().dropna()

        analyzer = RiskAnalyzer(daily_returns, self.rf)
        metrics = analyzer.summary(daily_returns)
        metrics["total_return"] = (equity.iloc[-1] / self.initial_capital) - 1
        metrics["cagr"] = (equity.iloc[-1] / self.initial_capital) ** (
            252 / len(equity)) - 1

        return BacktestResult(
            equity_curve=equity,
            daily_returns=daily_returns,
            trades=trades,
            metrics=metrics,
        )

    def run_with_optimization(self, method: str = "risk_parity",
                               lookback: int = 252,
                               max_weight: float = 0.10) -> BacktestResult:
        tickers = [t for t in self.weights if t in self.returns.columns]
        ret = self.returns[tickers]
        rebalance_dates = self._get_rebalance_dates()

        portfolio_values = [self.initial_capital]
        current_weights = np.ones(len(tickers)) / len(tickers)
        trades = []
        weights_records = []

        for i in range(len(ret)):
            date = ret.index[i]
            daily_ret = ret.iloc[i].values

            asset_values = current_weights * portfolio_values[-1]
            asset_values = asset_values * (1 + daily_ret)
            new_total = asset_values.sum()
            portfolio_values.append(new_total)

            if new_total > 0:
                current_weights = asset_values / new_total

            # Walk-forward rebalance with re-optimization
            is_rebalance = any(abs((date - rd).days) <= 1 for rd in rebalance_dates)
            if is_rebalance and i >= lookback:
                window = ret.iloc[i - lookback:i]
                try:
                    optimizer = PortfolioOptimizer(window, self.rf)
                    if method == "risk_parity":
                        result = optimizer.risk_parity()
                    elif method == "min_variance":
                        result = optimizer.minimum_variance(max_weight)
                    else:
                        result = optimizer.mean_variance_optimize(max_weight=max_weight)

                    if result["success"]:
                        new_w = np.array([result["weights"].get(t, 0) for t in tickers])
                        new_w = new_w / new_w.sum()

                        trades.append({
                            "date": date,
                            "type": f"rebalance_{method}",
                            "weights_before": dict(zip(tickers, current_weights)),
                            "weights_after": dict(zip(tickers, new_w)),
                            "portfolio_value": new_total,
                        })
                        current_weights = new_w
                except Exception as e:
                    logger.warning(f"Optimization failed at {date}: {e}")

            weights_records.append(dict(zip(tickers, current_weights)))

        equity = pd.Series(portfolio_values[1:], index=ret.index, name="portfolio")
        daily_returns = equity.pct_change().dropna()

        analyzer = RiskAnalyzer(daily_returns, self.rf)
        metrics = analyzer.summary(daily_returns)
        metrics["total_return"] = (equity.iloc[-1] / self.initial_capital) - 1
        metrics["cagr"] = (equity.iloc[-1] / self.initial_capital) ** (
            252 / len(equity)) - 1

        return BacktestResult(
            equity_curve=equity,
            daily_returns=daily_returns,
            trades=trades,
            metrics=metrics,
            weights_history=pd.DataFrame(weights_records, index=ret.index),
        )

    def compare_strategies(self) -> Dict[str, BacktestResult]:
        results = {}
        # Equal weight (static)
        n = len([t for t in self.weights if t in self.returns.columns])
        eq_weights = {t: 1/n for t in self.weights if t in self.returns.columns}
        self.weights = eq_weights
        results["equal_weight"] = self.run()

        # Walk-forward risk parity
        results["risk_parity"] = self.run_with_optimization("risk_parity")
        # Walk-forward min variance
        results["min_variance"] = self.run_with_optimization("min_variance")

        return results
