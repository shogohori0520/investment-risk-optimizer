import os
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List
from utils.helpers import setup_logging, is_jp_ticker

logger = setup_logging()


class DataFetcher:
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, ticker: str) -> str:
        safe_name = ticker.replace("^", "IDX_").replace("=", "_")
        return os.path.join(self.cache_dir, f"{safe_name}.csv")

    def fetch(self, tickers: List[str], start: str, end: str,
              interval: str = "1d") -> Dict[str, pd.DataFrame]:
        result = {}
        for ticker in tickers:
            cache = self._cache_path(ticker)
            if os.path.exists(cache):
                df = pd.read_csv(cache, index_col=0, parse_dates=True)
                if df.index.min() <= pd.Timestamp(start) and df.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=5):
                    result[ticker] = df.loc[start:end]
                    logger.debug(f"Cache hit: {ticker}")
                    continue

            logger.info(f"Downloading: {ticker}")
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            if df.empty:
                logger.warning(f"No data for {ticker}")
                continue
            # yfinance may return MultiIndex columns; flatten
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(cache)
            result[ticker] = df
        return result

    def get_returns(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        data = self.fetch(tickers, start, end)
        closes = pd.DataFrame({t: d["Close"] for t, d in data.items() if "Close" in d.columns})
        returns = np.log(closes / closes.shift(1)).dropna()
        return returns

    def get_fx_rate(self, start: str, end: str) -> pd.Series:
        data = self.fetch(["USDJPY=X"], start, end)
        if "USDJPY=X" not in data:
            logger.warning("FX data unavailable, using 1.0")
            return pd.Series(dtype=float)
        return data["USDJPY=X"]["Close"]

    def unify_currency(self, returns: pd.DataFrame, tickers_us: List[str],
                       start: str, end: str) -> pd.DataFrame:
        fx_data = self.fetch(["USDJPY=X"], start, end)
        if "USDJPY=X" not in fx_data:
            logger.warning("Cannot unify currency, FX data missing")
            return returns

        fx_close = fx_data["USDJPY=X"]["Close"]
        fx_returns = np.log(fx_close / fx_close.shift(1)).dropna()

        unified = returns.copy()
        for ticker in tickers_us:
            if ticker in unified.columns:
                common_idx = unified[ticker].index.intersection(fx_returns.index)
                unified.loc[common_idx, ticker] = (
                    unified.loc[common_idx, ticker] + fx_returns.loc[common_idx]
                )
        return unified
