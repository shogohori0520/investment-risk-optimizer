import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from utils.helpers import annualize_return, annualize_volatility


class PortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.005):
        self.returns = returns
        self.rf = risk_free_rate
        self.n_assets = len(returns.columns)
        self.tickers = list(returns.columns)
        self.mu = returns.mean() * 252  # annualized expected returns
        self.cov = self._shrunk_covariance()

    def _shrunk_covariance(self) -> pd.DataFrame:
        lw = LedoitWolf().fit(self.returns.dropna())
        return pd.DataFrame(
            lw.covariance_ * 252,
            index=self.tickers, columns=self.tickers
        )

    def mean_variance_optimize(self, target_return: float = None,
                                max_weight: float = 0.10) -> dict:
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        if target_return is not None:
            constraints.append({
                "type": "eq",
                "fun": lambda w: w @ self.mu.values - target_return
            })

        bounds = [(0, max_weight) for _ in range(self.n_assets)]
        w0 = np.ones(self.n_assets) / self.n_assets

        if target_return is None:
            # Maximize Sharpe ratio (minimize negative Sharpe)
            def neg_sharpe(w):
                ret = w @ self.mu.values
                vol = np.sqrt(w @ self.cov.values @ w)
                if vol == 0:
                    return 0
                return -(ret - self.rf) / vol

            result = minimize(neg_sharpe, w0, method="SLSQP",
                            bounds=bounds, constraints=constraints)
        else:
            # Minimize variance for given target return
            def variance(w):
                return w @ self.cov.values @ w

            result = minimize(variance, w0, method="SLSQP",
                            bounds=bounds, constraints=constraints)

        weights = dict(zip(self.tickers, result.x))
        port_ret = result.x @ self.mu.values
        port_vol = np.sqrt(result.x @ self.cov.values @ result.x)

        return {
            "weights": weights,
            "expected_return": port_ret,
            "volatility": port_vol,
            "sharpe": (port_ret - self.rf) / port_vol if port_vol > 0 else 0,
            "success": result.success,
        }

    def minimum_variance(self, max_weight: float = 0.10) -> dict:
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, max_weight) for _ in range(self.n_assets)]
        w0 = np.ones(self.n_assets) / self.n_assets

        def variance(w):
            return w @ self.cov.values @ w

        result = minimize(variance, w0, method="SLSQP",
                         bounds=bounds, constraints=constraints)

        weights = dict(zip(self.tickers, result.x))
        port_ret = result.x @ self.mu.values
        port_vol = np.sqrt(result.x @ self.cov.values @ result.x)

        return {
            "weights": weights,
            "expected_return": port_ret,
            "volatility": port_vol,
            "sharpe": (port_ret - self.rf) / port_vol if port_vol > 0 else 0,
            "success": result.success,
        }

    def risk_parity(self) -> dict:
        n = self.n_assets
        cov = self.cov.values

        def risk_budget_objective(w):
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol == 0:
                return 0
            marginal_contrib = cov @ w
            risk_contrib = w * marginal_contrib / port_vol
            target_rc = port_vol / n
            return np.sum((risk_contrib - target_rc) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.01, 1.0) for _ in range(n)]
        w0 = np.ones(n) / n

        result = minimize(risk_budget_objective, w0, method="SLSQP",
                         bounds=bounds, constraints=constraints)

        weights = dict(zip(self.tickers, result.x))
        port_ret = result.x @ self.mu.values
        port_vol = np.sqrt(result.x @ cov @ result.x)

        return {
            "weights": weights,
            "expected_return": port_ret,
            "volatility": port_vol,
            "sharpe": (port_ret - self.rf) / port_vol if port_vol > 0 else 0,
            "success": result.success,
        }

    def efficient_frontier(self, n_points: int = 50,
                           max_weight: float = 0.10) -> pd.DataFrame:
        min_ret = self.mu.min()
        max_ret = self.mu.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)

        results = []
        for target in target_returns:
            try:
                opt = self.mean_variance_optimize(target_return=target,
                                                   max_weight=max_weight)
                if opt["success"]:
                    results.append({
                        "return": opt["expected_return"],
                        "volatility": opt["volatility"],
                        "sharpe": opt["sharpe"],
                    })
            except Exception:
                continue

        return pd.DataFrame(results)

    def compare_strategies(self) -> dict:
        return {
            "max_sharpe": self.mean_variance_optimize(),
            "min_variance": self.minimum_variance(),
            "risk_parity": self.risk_parity(),
            "equal_weight": self._equal_weight(),
        }

    def _equal_weight(self) -> dict:
        w = np.ones(self.n_assets) / self.n_assets
        port_ret = w @ self.mu.values
        port_vol = np.sqrt(w @ self.cov.values @ w)
        return {
            "weights": dict(zip(self.tickers, w)),
            "expected_return": port_ret,
            "volatility": port_vol,
            "sharpe": (port_ret - self.rf) / port_vol if port_vol > 0 else 0,
            "success": True,
        }
