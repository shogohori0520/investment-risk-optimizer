import logging
import yaml
import numpy as np
import pandas as pd


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("investment")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    return logger


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def annualize_return(daily_returns: pd.Series) -> float:
    total = (1 + daily_returns).prod()
    n_years = len(daily_returns) / 252
    if n_years <= 0:
        return 0.0
    return total ** (1 / n_years) - 1


def annualize_volatility(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(252)


def is_jp_ticker(ticker: str) -> bool:
    return ticker.endswith(".T")


def jp_lot_round(shares: float) -> int:
    return int(shares // 100) * 100
