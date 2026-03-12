import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

import click
import pandas as pd
from utils.helpers import load_config, setup_logging

logger = setup_logging()


@click.group()
@click.option("--config", default="config.yaml", help="設定ファイルのパス")
@click.pass_context
def cli(ctx, config):
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


@cli.command()
@click.option("--method", default="risk_parity",
              type=click.Choice(["risk_parity", "max_sharpe", "min_variance"]),
              help="最適化手法")
@click.pass_context
def optimize(ctx, method):
    """ポートフォリオ最適化を実行"""
    cfg = ctx.obj["config"]
    tickers = cfg["portfolio"]["tickers_jp"] + cfg["portfolio"]["tickers_us"]
    rf = cfg["risk"]["risk_free_rate"]
    max_w = cfg["position"]["max_position_pct"]

    from core.data import DataFetcher
    from core.portfolio import PortfolioOptimizer
    from core.risk import RiskAnalyzer
    from core.position import PositionManager
    from viz import charts

    fetcher = DataFetcher()
    start = cfg["backtest"]["start_date"]
    end = cfg["backtest"]["end_date"]

    logger.info("データを取得中...")
    returns = fetcher.get_returns(tickers, start, end)
    returns = fetcher.unify_currency(returns, cfg["portfolio"]["tickers_us"], start, end)

    logger.info(f"最適化手法: {method}")
    optimizer = PortfolioOptimizer(returns, rf)

    if method == "risk_parity":
        result = optimizer.risk_parity()
    elif method == "max_sharpe":
        result = optimizer.mean_variance_optimize(max_weight=max_w)
    else:
        result = optimizer.minimum_variance(max_weight=max_w)

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo(f"  最適化結果 ({method})")
    click.echo("=" * 60)
    click.echo(f"  期待リターン（年率）: {result['expected_return']*100:.2f}%")
    click.echo(f"  ボラティリティ（年率）: {result['volatility']*100:.2f}%")
    click.echo(f"  シャープレシオ: {result['sharpe']:.3f}")
    click.echo("\n  配分:")
    for ticker, weight in sorted(result["weights"].items(), key=lambda x: -x[1]):
        if weight > 0.001:
            click.echo(f"    {ticker:>10s}: {weight*100:6.2f}%")

    # Position sizing
    click.echo("\n  具体的な投資額:")
    capital = cfg["position"]["total_capital"]
    pm = PositionManager(capital, max_w)

    # Get latest prices (use last available data from returns period)
    all_data = fetcher.fetch(tickers, start, end)
    prices = {}
    for t, df in all_data.items():
        if not df.empty and "Close" in df.columns:
            prices[t] = df["Close"].iloc[-1]

    alloc = pm.portfolio_allocation(result["weights"], prices)
    for ticker, info in alloc.items():
        if ticker == "_summary":
            continue
        click.echo(f"    {ticker:>10s}: {info['shares']:>6}株  "
                    f"¥{info['actual_value']:>12,.0f}  "
                    f"({info['actual_weight']*100:.1f}%)")

    summary = alloc.get("_summary", {})
    click.echo(f"\n  投資総額: ¥{summary.get('total_allocated', 0):,.0f}")
    click.echo(f"  残り現金: ¥{summary.get('cash_remaining', 0):,.0f}")

    # Compare all strategies
    click.echo("\n" + "=" * 60)
    click.echo("  戦略比較")
    click.echo("=" * 60)
    comparison = optimizer.compare_strategies()
    click.echo(f"  {'戦略':<16s} {'リターン':>8s} {'リスク':>8s} {'Sharpe':>8s}")
    click.echo("  " + "-" * 44)
    for name, res in comparison.items():
        click.echo(f"  {name:<16s} {res['expected_return']*100:>7.2f}% "
                    f"{res['volatility']*100:>7.2f}% {res['sharpe']:>7.3f}")

    # Generate charts
    logger.info("チャートを生成中...")
    frontier = optimizer.efficient_frontier(max_weight=max_w)
    if not frontier.empty:
        charts.plot_efficient_frontier(frontier, result)

    risk_analyzer = RiskAnalyzer(returns, rf)
    charts.plot_correlation_heatmap(risk_analyzer.correlation_matrix())
    charts.plot_weights(result["weights"])

    click.echo(f"\n  チャートを reports/ に保存しました")


@cli.command()
@click.option("--method", default="risk_parity",
              type=click.Choice(["risk_parity", "max_sharpe", "min_variance", "equal_weight"]),
              help="バックテスト戦略")
@click.pass_context
def backtest(ctx, method):
    """バックテストを実行"""
    cfg = ctx.obj["config"]
    tickers = cfg["portfolio"]["tickers_jp"] + cfg["portfolio"]["tickers_us"]
    rf = cfg["risk"]["risk_free_rate"]
    start = cfg["backtest"]["start_date"]
    end = cfg["backtest"]["end_date"]
    freq = cfg["backtest"]["rebalance_frequency"]
    capital = cfg["position"]["total_capital"]

    from core.data import DataFetcher
    from core.backtest import BacktestEngine
    from core.risk import RiskAnalyzer
    from viz import charts

    fetcher = DataFetcher()
    logger.info("データを取得中...")
    returns = fetcher.get_returns(tickers, start, end)
    returns = fetcher.unify_currency(returns, cfg["portfolio"]["tickers_us"], start, end)

    # Initial equal weights
    available = [t for t in tickers if t in returns.columns]
    weights = {t: 1/len(available) for t in available}

    engine = BacktestEngine(returns, weights, freq, capital, rf)

    if method == "equal_weight":
        logger.info("等ウェイトバックテスト実行中...")
        result = engine.run()
    else:
        logger.info(f"ウォークフォワード {method} バックテスト実行中...")
        result = engine.run_with_optimization(
            method=method,
            lookback=cfg["backtest"].get("lookback_days", 252),
        )

    # Display results
    click.echo("\n" + "=" * 60)
    click.echo(f"  バックテスト結果 ({method})")
    click.echo(f"  期間: {start} ~ {end}")
    click.echo("=" * 60)

    m = result.metrics
    click.echo(f"  総リターン: {m.get('total_return', 0)*100:.2f}%")
    click.echo(f"  CAGR: {m.get('cagr', 0)*100:.2f}%")
    click.echo(f"  年率ボラティリティ: {m['annualized_volatility']*100:.2f}%")
    click.echo(f"  シャープレシオ: {m['sharpe_ratio']:.3f}")
    click.echo(f"  ソルティノレシオ: {m['sortino_ratio']:.3f}")
    click.echo(f"  最大ドローダウン: {m['max_drawdown']*100:.2f}%")
    click.echo(f"  カルマーレシオ: {m['calmar_ratio']:.3f}")
    click.echo(f"  VaR (95%): {m['var_95_historical']*100:.2f}%")
    click.echo(f"  CVaR (95%): {m['cvar_95']*100:.2f}%")
    click.echo(f"  リバランス回数: {len(result.trades)}")

    final_value = result.equity_curve.iloc[-1]
    click.echo(f"\n  初期資本: ¥{capital:,.0f}")
    click.echo(f"  最終資産: ¥{final_value:,.0f}")
    click.echo(f"  損益: ¥{final_value - capital:,.0f}")

    # Generate charts
    logger.info("チャートを生成中...")

    # Try to get benchmark
    benchmark_equity = None
    try:
        bench_ticker = cfg["portfolio"]["benchmark"]
        bench_ret = fetcher.get_returns([bench_ticker], start, end)
        if bench_ticker in bench_ret.columns:
            bench_cum = (1 + bench_ret[bench_ticker]).cumprod() * capital
            # Align indices
            common = result.equity_curve.index.intersection(bench_cum.index)
            if len(common) > 0:
                benchmark_equity = bench_cum.loc[common]
    except Exception:
        pass

    charts.plot_equity_curve(result.equity_curve, benchmark_equity)

    analyzer = RiskAnalyzer(result.daily_returns, rf)
    charts.plot_drawdown(analyzer.drawdown_series(result.daily_returns))

    rolling = analyzer.rolling_metrics()
    charts.plot_rolling_metrics(rolling)
    charts.plot_risk_summary(m)

    click.echo(f"\n  チャートを reports/ に保存しました")


@cli.command(name="risk-report")
@click.pass_context
def risk_report(ctx):
    """現在のリスクレポートを生成"""
    cfg = ctx.obj["config"]
    tickers = cfg["portfolio"]["tickers_jp"] + cfg["portfolio"]["tickers_us"]
    rf = cfg["risk"]["risk_free_rate"]

    from core.data import DataFetcher
    from core.risk import RiskAnalyzer
    from viz import charts

    fetcher = DataFetcher()
    start = cfg["backtest"]["start_date"]
    end = cfg["backtest"]["end_date"]

    logger.info("データを取得中...")
    returns = fetcher.get_returns(tickers, start, end)
    returns = fetcher.unify_currency(returns, cfg["portfolio"]["tickers_us"], start, end)

    analyzer = RiskAnalyzer(returns, rf)

    click.echo("\n" + "=" * 60)
    click.echo("  リスクレポート")
    click.echo("=" * 60)

    # Per-asset metrics
    click.echo(f"\n  {'銘柄':>10s} {'リターン':>8s} {'リスク':>8s} {'Sharpe':>8s} {'最大DD':>8s}")
    click.echo("  " + "-" * 46)
    for ticker in returns.columns:
        s = analyzer.summary(returns[ticker])
        click.echo(f"  {ticker:>10s} {s['annualized_return']*100:>7.2f}% "
                    f"{s['annualized_volatility']*100:>7.2f}% "
                    f"{s['sharpe_ratio']:>7.3f} "
                    f"{s['max_drawdown']*100:>7.2f}%")

    # Portfolio (equal weight)
    port_ret = returns.mean(axis=1)
    port_summary = analyzer.summary(port_ret)
    click.echo(f"\n  ポートフォリオ（等ウェイト）:")
    click.echo(f"    年率リターン: {port_summary['annualized_return']*100:.2f}%")
    click.echo(f"    年率リスク: {port_summary['annualized_volatility']*100:.2f}%")
    click.echo(f"    シャープレシオ: {port_summary['sharpe_ratio']:.3f}")
    click.echo(f"    最大ドローダウン: {port_summary['max_drawdown']*100:.2f}%")
    click.echo(f"    VaR (95%): {port_summary['var_95_historical']*100:.2f}%")
    click.echo(f"    CVaR (95%): {port_summary['cvar_95']*100:.2f}%")

    # Correlation
    corr = analyzer.correlation_matrix()
    click.echo(f"\n  相関マトリクス（低相関 = 分散効果大）:")
    import numpy as _np
    avg_corr = corr.values[~_np.eye(len(corr), dtype=bool)].mean() if len(corr) > 1 else 0
    click.echo(f"    平均相関係数: {avg_corr:.3f}")

    # Generate charts
    charts.plot_correlation_heatmap(corr)
    charts.plot_risk_summary(port_summary)

    rolling = analyzer.rolling_metrics()
    charts.plot_rolling_metrics(rolling)

    click.echo(f"\n  チャートを reports/ に保存しました")


@cli.command()
@click.option("--ticker", required=True, help="銘柄コード")
@click.option("--budget", type=float, default=None, help="投資予算（省略時はconfig参照）")
@click.option("--risk-pct", type=float, default=0.02, help="1トレードのリスク率")
@click.pass_context
def position(ctx, ticker, budget, risk_pct):
    """ポジションサイズを計算"""
    cfg = ctx.obj["config"]
    capital = budget or cfg["position"]["total_capital"]
    stop_loss = cfg["position"]["stop_loss_pct"]
    max_pos = cfg["position"]["max_position_pct"]

    from core.data import DataFetcher
    from core.position import PositionManager

    fetcher = DataFetcher()
    end = cfg["backtest"]["end_date"]
    start = pd.Timestamp(end) - pd.Timedelta(days=30)

    data = fetcher.fetch([ticker], str(start.date()), end)
    if ticker not in data or data[ticker].empty:
        click.echo(f"  エラー: {ticker} のデータを取得できませんでした")
        return

    ohlcv = data[ticker]
    current_price = ohlcv["Close"].iloc[-1]

    pm = PositionManager(capital, max_pos, stop_loss)

    # Fixed stop-loss
    result = pm.calculate_position_size(ticker, current_price, risk_pct)

    click.echo("\n" + "=" * 60)
    click.echo(f"  ポジションサイズ計算: {ticker}")
    click.echo("=" * 60)
    click.echo(f"  現在価格: ¥{current_price:,.0f}")
    click.echo(f"  総資本: ¥{capital:,.0f}")
    click.echo(f"  リスク率: {risk_pct*100:.1f}%")
    click.echo(f"\n  推奨株数: {result['shares']}株")
    click.echo(f"  投資額: ¥{result['position_value']:,.0f} "
                f"({result['position_pct']*100:.1f}%)")
    click.echo(f"  損切り価格（固定 {stop_loss*100:.0f}%）: "
                f"¥{result['stop_loss_price']:,.0f}")
    click.echo(f"  最大損失額: ¥{result['risk_amount']:,.0f}")

    # ATR-based stop
    atr_stop = pm.atr_stop_loss(ohlcv)
    click.echo(f"  損切り価格（ATR）: ¥{atr_stop:,.0f}")

    click.echo()


if __name__ == "__main__":
    cli()
