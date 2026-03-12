import numpy as np
import pandas as pd
from scipy import stats
from utils.helpers import annualize_return, annualize_volatility


class RiskAnalyzer:
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.005):
        self.returns = returns
        self.rf = risk_free_rate
        self.rf_daily = (1 + risk_free_rate) ** (1/252) - 1

    def sharpe_ratio(self, series: pd.Series = None) -> float:
        s = series if series is not None else self._portfolio_returns()
        ann_ret = annualize_return(s)
        ann_vol = annualize_volatility(s)
        if ann_vol == 0:
            return 0.0
        return (ann_ret - self.rf) / ann_vol

    def sortino_ratio(self, series: pd.Series = None) -> float:
        s = series if series is not None else self._portfolio_returns()
        ann_ret = annualize_return(s)
        downside = s[s < 0]
        if len(downside) == 0:
            return float('inf')
        downside_vol = downside.std() * np.sqrt(252)
        if downside_vol == 0:
            return 0.0
        return (ann_ret - self.rf) / downside_vol

    def max_drawdown(self, series: pd.Series = None) -> float:
        s = series if series is not None else self._portfolio_returns()
        cum = (1 + s).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd.min()

    def drawdown_series(self, series: pd.Series = None) -> pd.Series:
        s = series if series is not None else self._portfolio_returns()
        cum = (1 + s).cumprod()
        peak = cum.cummax()
        return (cum - peak) / peak

    def var(self, confidence: float = 0.95, method: str = "historical",
            series: pd.Series = None) -> float:
        s = series if series is not None else self._portfolio_returns()
        if method == "historical":
            return np.percentile(s, (1 - confidence) * 100)
        elif method == "parametric":
            z = stats.norm.ppf(1 - confidence)
            return s.mean() + z * s.std()
        elif method == "cornish_fisher":
            z = stats.norm.ppf(1 - confidence)
            sk = stats.skew(s)
            ku = stats.kurtosis(s)
            z_cf = (z + (z**2 - 1) * sk / 6
                    + (z**3 - 3*z) * ku / 24
                    - (2*z**3 - 5*z) * sk**2 / 36)
            return s.mean() + z_cf * s.std()
        else:
            raise ValueError(f"Unknown method: {method}")

    def cvar(self, confidence: float = 0.95, series: pd.Series = None) -> float:
        s = series if series is not None else self._portfolio_returns()
        var_level = self.var(confidence, "historical", s)
        return s[s <= var_level].mean()

    def calmar_ratio(self, series: pd.Series = None) -> float:
        s = series if series is not None else self._portfolio_returns()
        ann_ret = annualize_return(s)
        mdd = abs(self.max_drawdown(s))
        if mdd == 0:
            return 0.0
        return ann_ret / mdd

    def correlation_matrix(self) -> pd.DataFrame:
        return self.returns.corr()

    def rolling_metrics(self, window: int = 252) -> pd.DataFrame:
        port = self._portfolio_returns()
        rolling_ret = port.rolling(window).apply(annualize_return, raw=False)
        rolling_vol = port.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_ret - self.rf) / rolling_vol

        cum = (1 + port).cumprod()
        rolling_dd = cum.rolling(window).apply(
            lambda x: (x[-1] - x.max()) / x.max() if x.max() != 0 else 0, raw=True
        )

        return pd.DataFrame({
            "return": rolling_ret,
            "volatility": rolling_vol,
            "sharpe": rolling_sharpe,
            "drawdown": rolling_dd,
        })

    def summary(self, series: pd.Series = None) -> dict:
        s = series if series is not None else self._portfolio_returns()
        return {
            "annualized_return": annualize_return(s),
            "annualized_volatility": annualize_volatility(s),
            "sharpe_ratio": self.sharpe_ratio(s),
            "sortino_ratio": self.sortino_ratio(s),
            "max_drawdown": self.max_drawdown(s),
            "calmar_ratio": self.calmar_ratio(s),
            "var_95_historical": self.var(0.95, "historical", s),
            "var_95_cornish_fisher": self.var(0.95, "cornish_fisher", s),
            "cvar_95": self.cvar(0.95, s),
            "skewness": float(stats.skew(s)),
            "kurtosis": float(stats.kurtosis(s)),
        }

    def _portfolio_returns(self) -> pd.Series:
        if isinstance(self.returns, pd.Series):
            return self.returns
        return self.returns.mean(axis=1)
