import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "reports"


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_efficient_frontier(frontier_df: pd.DataFrame, optimal: dict,
                            filename: str = "efficient_frontier.png"):
    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frontier_df["volatility"] * 100, frontier_df["return"] * 100,
            "b-", linewidth=2, label="効率的フロンティア")
    ax.scatter(optimal["volatility"] * 100, optimal["expected_return"] * 100,
               c="red", s=200, marker="*", zorder=5, label="最適ポートフォリオ")
    ax.set_xlabel("リスク（年率ボラティリティ %）")
    ax.set_ylabel("期待リターン（年率 %）")
    ax.set_title("効率的フロンティア")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame,
                             filename: str = "correlation.png"):
    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("資産間相関マトリクス")
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_equity_curve(equity: pd.Series, benchmark: pd.Series = None,
                      filename: str = "equity_curve.png"):
    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(12, 6))

    # Normalize to 100
    normalized = equity / equity.iloc[0] * 100
    ax.plot(normalized.index, normalized.values, "b-", linewidth=1.5,
            label="ポートフォリオ")

    if benchmark is not None:
        bench_norm = benchmark / benchmark.iloc[0] * 100
        ax.plot(bench_norm.index, bench_norm.values, "r--", linewidth=1,
                alpha=0.7, label="ベンチマーク")

    ax.set_xlabel("日付")
    ax.set_ylabel("パフォーマンス（100基準）")
    ax.set_title("エクイティカーブ")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown(drawdown: pd.Series, filename: str = "drawdown.png"):
    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown.values * 100, 0,
                     color="red", alpha=0.3)
    ax.plot(drawdown.index, drawdown.values * 100, "r-", linewidth=0.8)
    ax.set_xlabel("日付")
    ax.set_ylabel("ドローダウン (%)")
    ax.set_title("最大ドローダウン推移")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_weights(weights: dict, filename: str = "weights.png"):
    _ensure_output_dir()
    filtered = {k: v for k, v in weights.items() if v > 0.001}
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(filtered.values(), labels=filtered.keys(), autopct="%1.1f%%",
           startangle=90)
    ax.set_title("ポートフォリオ配分")
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_metrics(rolling_df: pd.DataFrame,
                         filename: str = "rolling_metrics.png"):
    _ensure_output_dir()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(rolling_df.index, rolling_df["sharpe"], "b-", linewidth=0.8)
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].set_ylabel("シャープレシオ")
    axes[0].set_title("ローリング指標（252日）")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rolling_df.index, rolling_df["volatility"] * 100, "orange",
                 linewidth=0.8)
    axes[1].set_ylabel("ボラティリティ (%)")
    axes[1].grid(True, alpha=0.3)

    axes[2].fill_between(rolling_df.index, rolling_df["drawdown"] * 100, 0,
                          color="red", alpha=0.3)
    axes[2].set_ylabel("ドローダウン (%)")
    axes[2].set_xlabel("日付")
    axes[2].grid(True, alpha=0.3)

    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_risk_summary(metrics: dict, filename: str = "risk_summary.png"):
    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    labels = {
        "annualized_return": "年率リターン",
        "annualized_volatility": "年率ボラティリティ",
        "sharpe_ratio": "シャープレシオ",
        "sortino_ratio": "ソルティノレシオ",
        "max_drawdown": "最大ドローダウン",
        "calmar_ratio": "カルマーレシオ",
        "var_95_historical": "VaR (95%, ヒストリカル)",
        "cvar_95": "CVaR (95%)",
    }

    rows = []
    for key, label in labels.items():
        if key in metrics:
            val = metrics[key]
            if "ratio" in key or key == "calmar_ratio":
                rows.append([label, f"{val:.3f}"])
            else:
                rows.append([label, f"{val*100:.2f}%"])

    table = ax.table(cellText=rows, colLabels=["指標", "値"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    ax.set_title("リスク指標サマリー", fontsize=14, pad=20)

    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
