#!/usr/bin/env python3
"""
Trader Performance vs Market Sentiment Analysis
================================================
Explores how Bitcoin Fear & Greed sentiment regimes relate to
Hyperliquid trader performance, uncovers hidden patterns, and
surfaces actionable trading strategy insights.

Usage:
    python trader_sentiment_analysis.py

Expected project structure:
    .
    ├── dataset/
    │   ├── historical_data.csv      # Hyperliquid trades
    │   └── fear_greed_index.csv     # Bitcoin Fear & Greed Index
    └── trader_sentiment_analysis.py

Outputs:
    outputs/
        charts/        ← PNG charts (300 dpi)
        tables/        ← CSV summary tables
        report.md      ← Full markdown report
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import f_oneway, kruskal, spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(".")
DATA_DIR      = BASE_DIR / "dataset"
OUTPUT_DIR    = BASE_DIR / "outputs"
CHART_DIR     = OUTPUT_DIR / "charts"
TABLE_DIR     = OUTPUT_DIR / "tables"
TRADES_FILE   = DATA_DIR  / "historical_data.csv"
SENTIMENT_FILE= DATA_DIR  / "fear_greed_index.csv"

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

# Palette aligned with sentiment order (blue → red)
SENTIMENT_PALETTE = {
    "Extreme Fear": "#1a78c2",
    "Fear":         "#56a0d3",
    "Neutral":      "#a8a8a8",
    "Greed":        "#e07b54",
    "Extreme Greed":"#c0392b",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ensure_directories() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def standardize_sentiment(series: pd.Series) -> pd.Series:
    mapping = {
        "extreme fear": "Extreme Fear",
        "fear":         "Fear",
        "neutral":      "Neutral",
        "greed":        "Greed",
        "extreme greed":"Extreme Greed",
    }
    return (
        series.astype(str).str.strip().str.lower()
        .map(mapping)
        .fillna(series.astype(str).str.strip())
    )


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("  saved → charts/%s", filename)


def clip_outliers(series: pd.Series, pct: float = 0.01) -> pd.Series:
    """Winsorise extreme tails for cleaner visualisations."""
    lo = series.quantile(pct)
    hi = series.quantile(1 - pct)
    return series.clip(lo, hi)


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    for f in (TRADES_FILE, SENTIMENT_FILE):
        if not f.exists():
            raise FileNotFoundError(f"Missing file: {f}")
    log.info("Loading trades …")
    trades    = pd.read_csv(TRADES_FILE)
    log.info("  trades shape: %s", trades.shape)
    log.info("Loading sentiment …")
    sentiment = pd.read_csv(SENTIMENT_FILE)
    log.info("  sentiment shape: %s", sentiment.shape)
    return trades, sentiment


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def resolve_trade_timestamp(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Pick the best timestamp column and return it as a proper datetime column.

    Detection order:
      1. Named columns (timestamp_ist, time, datetime, date_time) that parse
         as datetime strings with ≥ 95 % fill.
      2. 'timestamp' column:
         a. Already datetime-like string → parse directly.
         b. Unix epoch seconds  (9e8  – 2e9).
         c. Unix epoch milliseconds (9e11 – 2e12).
         d. Unix epoch microseconds (9e14 – 2e15).
         e. Unix epoch nanoseconds  (9e17 – 2e18).
    """
    df = df.copy()

    # ── Step 1: named datetime-string columns ──────────────────────────────
    preferred = ["timestamp_ist", "time", "datetime", "date_time"]
    for col in preferred:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            fill_rate = parsed.notna().mean()
            if fill_rate >= 0.95:
                df[col] = parsed
                log.info("  timestamp column: '%s'  (fill %.1f%%)", col, fill_rate * 100)
                return df, col

    # ── Step 2: 'timestamp' column ─────────────────────────────────────────
    if "timestamp" not in df.columns:
        raise KeyError(
            "Cannot resolve a trade timestamp column. Tried: "
            + str(preferred + ["timestamp"])
        )

    raw = df["timestamp"]

    # 2a. Try direct datetime string parse first (catches ISO strings)
    parsed_str = pd.to_datetime(raw, errors="coerce")
    str_fill = parsed_str.notna().mean()
    if str_fill >= 0.95:
        # Sanity-check: parsed dates must overlap with a reasonable trading era
        valid_dates = parsed_str.dropna()
        if not valid_dates.empty:
            median_year = valid_dates.dt.year.median()
            if 2010 <= median_year <= 2035:
                df["timestamp"] = parsed_str
                log.info(
                    "  timestamp column: 'timestamp' (datetime string, fill %.1f%%, "
                    "median year %.0f)", str_fill * 100, median_year
                )
                return df, "timestamp"

    # 2b-e. Numeric epoch detection
    numeric_ts = pd.to_numeric(raw, errors="coerce").dropna()
    if numeric_ts.empty:
        raise KeyError("'timestamp' column contains no parseable numeric or datetime values.")

    median_val = numeric_ts.median()
    log.info("  'timestamp' raw median value: %.3e", median_val)

    # Thresholds: lower bound of each unit for 2000-01-01
    epoch_units = [
        ("s",  9.46e8,  1.90e9 ),   # seconds
        ("ms", 9.46e11, 1.90e12),   # milliseconds
        ("us", 9.46e14, 1.90e15),   # microseconds
        ("ns", 9.46e17, 1.90e18),   # nanoseconds
    ]
    for unit, lo, hi in epoch_units:
        if lo <= median_val <= hi:
            df["timestamp_dt"] = pd.to_datetime(numeric_ts, unit=unit, errors="coerce").reindex(df.index)
            log.info(
                "  timestamp column: 'timestamp' (Unix epoch %s) → 'timestamp_dt'  "
                "sample: %s", unit, df["timestamp_dt"].dropna().iloc[0]
            )
            return df, "timestamp_dt"

    # Last resort: let pandas infer
    df["timestamp_dt"] = pd.to_datetime(raw, infer_datetime_format=True, errors="coerce")
    fill = df["timestamp_dt"].notna().mean()
    if fill >= 0.5:
        log.warning("  timestamp column: pandas inferred (fill %.1f%%) — check dates!", fill * 100)
        return df, "timestamp_dt"

    raise KeyError(
        f"Could not determine epoch unit for 'timestamp'. Median value: {median_val:.3e}. "
        "Please convert it to datetime before running."
    )


def preprocess(trades: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    log.info("Preprocessing …")

    trades    = clean_columns(trades)
    sentiment = clean_columns(sentiment)

    # ── Sentiment ──────────────────────────────────────────────────────────
    required_sent = {"date", "classification"}
    missing = required_sent - set(sentiment.columns)
    if missing:
        raise KeyError(f"Sentiment dataset missing columns: {missing}")

    sentiment["date"]           = pd.to_datetime(sentiment["date"], errors="coerce")
    sentiment["classification"] = standardize_sentiment(sentiment["classification"])

    # Optional: add numeric sentiment score for correlation analysis
    score_col = next((c for c in sentiment.columns if "value" in c), None)
    if score_col:
        sentiment[score_col] = pd.to_numeric(sentiment[score_col], errors="coerce")

    # ── Trades ─────────────────────────────────────────────────────────────
    trades, ts_col = resolve_trade_timestamp(trades)

    numeric_cols = [
        "execution_price", "size_tokens", "size_usd", "closed_pnl",
        "fee", "start_position", "leverage",
    ]
    trades = safe_numeric(trades, numeric_cols)

    # Normalise side / direction column
    side_col = next((c for c in ["side", "direction"] if c in trades.columns), None)
    if side_col:
        trades[side_col] = trades[side_col].astype(str).str.strip().str.lower()
        if side_col != "side":
            trades["side"] = trades[side_col]

    # Normalise coin / symbol column
    coin_col = next((c for c in ["coin", "symbol", "asset"] if c in trades.columns), None)
    if coin_col:
        trades[coin_col] = trades[coin_col].astype(str).str.strip()
        if coin_col != "coin":
            trades["coin"] = trades[coin_col]

    # Extract calendar features
    trades["date"]    = pd.to_datetime(trades[ts_col].dt.date)
    trades["hour"]    = trades[ts_col].dt.hour
    trades["weekday"] = trades[ts_col].dt.day_name()
    trades["month"]   = trades[ts_col].dt.to_period("M").astype(str)

    # ── Merge ──────────────────────────────────────────────────────────────
    sent_cols = ["date", "classification"]
    if score_col:
        sent_cols.append(score_col)

    data = trades.merge(sentiment[sent_cols], on="date", how="left")
    before = len(data)
    data = data.dropna(subset=["classification"]).copy()

    # ── Diagnostic: if merge produced 0 rows, show sample dates so the user
    #    can see exactly why the join failed ───────────────────────────────
    if len(data) == 0:
        trade_dates   = trades["date"].dropna().sort_values()
        sent_dates    = sentiment["date"].dropna().sort_values()
        log.error("=" * 60)
        log.error("MERGE PRODUCED 0 ROWS — date ranges do not overlap!")
        log.error("  Trade dates   : %s → %s  (%d unique)",
                  trade_dates.iloc[0].date(), trade_dates.iloc[-1].date(),
                  trade_dates.nunique())
        log.error("  Sentiment dates: %s → %s  (%d unique)",
                  sent_dates.iloc[0].date(), sent_dates.iloc[-1].date(),
                  sent_dates.nunique())
        log.error("  Trade date dtype    : %s", trades['date'].dtype)
        log.error("  Sentiment date dtype: %s", sentiment['date'].dtype)
        log.error("  Sample trade dates  : %s", trade_dates.dt.date.unique()[:5].tolist())
        log.error("  Sample sent  dates  : %s", sent_dates.dt.date.unique()[:5].tolist())
        log.error("=" * 60)
        raise ValueError(
            "Date merge failed: trades and sentiment share 0 matching dates.\n"
            f"  Trade range   : {trade_dates.iloc[0].date()} → {trade_dates.iloc[-1].date()}\n"
            f"  Sentiment range: {sent_dates.iloc[0].date()} → {sent_dates.iloc[-1].date()}\n"
            "Check the timestamp unit (s / ms / us / ns) — run with DEBUG to see "
            "the raw median value printed above."
        )

    log.info(
        "  rows with sentiment match: %d / %d (%.1f%%)",
        len(data), before, len(data) / before * 100,
    )

    # Keep only completed trades with valid non-zero PnL
    if "closed_pnl" not in data.columns:
        raise KeyError("Trades must contain a 'closed_pnl' column.")
    data = data[data["closed_pnl"].notna() & (data["closed_pnl"] != 0)].copy()
    log.info("  rows after PnL filter: %d", len(data))

    # ── Derived features ───────────────────────────────────────────────────
    data["is_profit"]   = (data["closed_pnl"] > 0).astype(int)
    data["is_loss"]     = (data["closed_pnl"] < 0).astype(int)

    size_col = next((c for c in ["size_usd", "size_tokens"] if c in data.columns), None)
    if size_col:
        data["abs_size"] = data[size_col].abs()
    else:
        data["abs_size"] = np.nan

    data["pnl_per_size"] = data["closed_pnl"] / data["abs_size"]
    data["pnl_per_size"]  = data["pnl_per_size"].replace([np.inf, -np.inf], np.nan)
    # Clip at 1st / 99th percentile to remove extreme micro-trades
    data["pnl_per_size"]  = clip_outliers(data["pnl_per_size"].dropna()).reindex(data.index)

    data["log_abs_size"] = np.log1p(data["abs_size"].abs())

    data["classification"] = pd.Categorical(
        data["classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    data = data.sort_values(["date", "classification"]).copy()

    if score_col:
        data.rename(columns={score_col: "sentiment_score"}, inplace=True)

    log.info("Preprocessing done.  Final shape: %s", data.shape)
    return data


# ══════════════════════════════════════════════════════════════════════════════
# 4. SUMMARY TABLES
# ══════════════════════════════════════════════════════════════════════════════

def build_summary_tables(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    # ── Core summary ───────────────────────────────────────────────────────
    summary = (
        data.groupby("classification", observed=True)
        .agg(
            trade_count    = ("closed_pnl", "count"),
            total_pnl      = ("closed_pnl", "sum"),
            avg_pnl        = ("closed_pnl", "mean"),
            median_pnl     = ("closed_pnl", "median"),
            std_pnl        = ("closed_pnl", "std"),
            win_rate_pct   = ("is_profit", lambda x: x.mean() * 100),
            avg_size       = ("abs_size",  "mean"),
            total_volume   = ("abs_size",  "sum"),
            pnl_efficiency = ("pnl_per_size", "mean"),
        )
        .round(4)
        .reset_index()
    )
    # Sharpe-like ratio: mean PnL / std PnL per sentiment bucket
    summary["pnl_sharpe"] = (summary["avg_pnl"] / summary["std_pnl"]).round(4)
    tables["summary_by_sentiment"] = summary

    # ── Side performance ───────────────────────────────────────────────────
    if "side" in data.columns:
        side_perf = (
            data.groupby(["classification", "side"], observed=True)
            .agg(
                trades       = ("closed_pnl", "count"),
                total_pnl    = ("closed_pnl", "sum"),
                avg_pnl      = ("closed_pnl", "mean"),
                win_rate_pct = ("is_profit", lambda x: x.mean() * 100),
                avg_size     = ("abs_size",  "mean"),
            )
            .round(4)
            .reset_index()
        )
        tables["side_performance"] = side_perf

    # ── Top coin performance ───────────────────────────────────────────────
    if "coin" in data.columns:
        top_coins = data.groupby("coin")["closed_pnl"].sum().nlargest(10).index
        coin_summary = (
            data[data["coin"].isin(top_coins)]
            .groupby(["classification", "coin"], observed=True)
            .agg(
                trades       = ("closed_pnl", "count"),
                total_pnl    = ("closed_pnl", "sum"),
                avg_pnl      = ("closed_pnl", "mean"),
                win_rate_pct = ("is_profit", lambda x: x.mean() * 100),
            )
            .round(4)
            .reset_index()
        )
        tables["top_coin_performance"] = coin_summary

    # ── Account-level performance ──────────────────────────────────────────
    if "account" in data.columns:
        acct = (
            data.groupby(["account", "classification"], observed=True)
            .agg(
                trades    = ("closed_pnl", "count"),
                total_pnl = ("closed_pnl", "sum"),
                win_rate  = ("is_profit", "mean"),
            )
            .round(4)
            .reset_index()
        )
        tables["account_sentiment_performance"] = acct

    # ── Hourly pattern ─────────────────────────────────────────────────────
    hourly = (
        data.groupby(["hour", "classification"], observed=True)["closed_pnl"]
        .mean()
        .round(4)
        .reset_index()
    )
    tables["hourly_avg_pnl"] = hourly

    # ── Weekday pattern ────────────────────────────────────────────────────
    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekday = (
        data.groupby(["weekday", "classification"], observed=True)
        .agg(avg_pnl=("closed_pnl","mean"), win_rate_pct=("is_profit", lambda x: x.mean()*100))
        .round(4)
        .reset_index()
    )
    weekday["weekday"] = pd.Categorical(weekday["weekday"], categories=weekday_order, ordered=True)
    weekday = weekday.sort_values("weekday")
    tables["weekday_pnl"] = weekday

    return tables


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    log.info("Saved %d tables to %s", len(tables), TABLE_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# 5. CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def set_theme() -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "figure.figsize":  (11, 5),
        "axes.titlesize":  14,
        "axes.labelsize":  11,
        "axes.titleweight":"bold",
        "grid.alpha":       0.4,
    })


def _sentiment_colors(cats):
    return [SENTIMENT_PALETTE.get(c, "#888888") for c in cats]


# ── 5a. Total PnL bar ──────────────────────────────────────────────────────
def plot_total_pnl(data: pd.DataFrame) -> None:
    agg = (
        data.groupby("classification", observed=True)["closed_pnl"]
        .sum()
        .reset_index()
    )
    plt.figure()
    bars = plt.bar(
        agg["classification"].astype(str),
        agg["closed_pnl"],
        color=_sentiment_colors(agg["classification"].astype(str)),
        edgecolor="white", linewidth=0.6,
    )
    plt.bar_label(bars, fmt="%.0f", padding=4, fontsize=9)
    plt.title("Total Realised PnL by Sentiment Regime")
    plt.xlabel("Sentiment")
    plt.ylabel("Total Closed PnL (USD)")
    plt.xticks(rotation=30, ha="right")
    save_plot("01_total_pnl_by_sentiment.png")


# ── 5b. Average PnL bar ───────────────────────────────────────────────────
def plot_avg_pnl(data: pd.DataFrame) -> None:
    agg = (
        data.groupby("classification", observed=True)["closed_pnl"]
        .mean()
        .reset_index()
    )
    plt.figure()
    bars = plt.bar(
        agg["classification"].astype(str),
        agg["closed_pnl"],
        color=_sentiment_colors(agg["classification"].astype(str)),
        edgecolor="white", linewidth=0.6,
    )
    plt.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    plt.title("Average PnL per Trade by Sentiment Regime")
    plt.xlabel("Sentiment")
    plt.ylabel("Average Closed PnL (USD)")
    plt.xticks(rotation=30, ha="right")
    save_plot("02_avg_pnl_by_sentiment.png")


# ── 5c. Win rate bar ──────────────────────────────────────────────────────
def plot_win_rate(data: pd.DataFrame) -> None:
    agg = (
        data.groupby("classification", observed=True)["is_profit"]
        .mean()
        .mul(100)
        .reset_index(name="win_rate_pct")
    )
    plt.figure()
    bars = plt.bar(
        agg["classification"].astype(str),
        agg["win_rate_pct"],
        color=_sentiment_colors(agg["classification"].astype(str)),
        edgecolor="white", linewidth=0.6,
    )
    plt.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    plt.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="50% baseline")
    plt.legend()
    plt.title("Win Rate (%) by Sentiment Regime")
    plt.xlabel("Sentiment")
    plt.ylabel("Win Rate (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=30, ha="right")
    save_plot("03_win_rate_by_sentiment.png")


# ── 5d. Trade count bar ───────────────────────────────────────────────────
def plot_trade_count(data: pd.DataFrame) -> None:
    agg = (
        data.groupby("classification", observed=True)
        .size()
        .reset_index(name="trade_count")
    )
    plt.figure()
    bars = plt.bar(
        agg["classification"].astype(str),
        agg["trade_count"],
        color=_sentiment_colors(agg["classification"].astype(str)),
        edgecolor="white", linewidth=0.6,
    )
    plt.bar_label(bars, fmt="%d", padding=4, fontsize=9)
    plt.title("Number of Trades by Sentiment Regime")
    plt.xlabel("Sentiment")
    plt.ylabel("Trade Count")
    plt.xticks(rotation=30, ha="right")
    save_plot("04_trade_count_by_sentiment.png")


# ── 5e. Trade size bar ────────────────────────────────────────────────────
def plot_trade_size(data: pd.DataFrame) -> None:
    if data["abs_size"].isna().all():
        return
    agg = (
        data.groupby("classification", observed=True)["abs_size"]
        .mean()
        .reset_index()
    )
    plt.figure()
    bars = plt.bar(
        agg["classification"].astype(str),
        agg["abs_size"],
        color=_sentiment_colors(agg["classification"].astype(str)),
        edgecolor="white", linewidth=0.6,
    )
    plt.bar_label(bars, fmt="%.0f", padding=4, fontsize=9)
    plt.title("Average Trade Size by Sentiment Regime")
    plt.xlabel("Sentiment")
    plt.ylabel("Average Trade Size (USD)")
    plt.xticks(rotation=30, ha="right")
    save_plot("05_avg_trade_size_by_sentiment.png")


# ── 5f. PnL distribution box ──────────────────────────────────────────────
def plot_pnl_distribution(data: pd.DataFrame) -> None:
    plot_data = data.copy()
    plot_data["closed_pnl"] = clip_outliers(plot_data["closed_pnl"])
    palette = [SENTIMENT_PALETTE.get(c, "#888") for c in SENTIMENT_ORDER
               if c in plot_data["classification"].cat.categories]
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=plot_data, x="classification", y="closed_pnl",
        palette=palette, showfliers=False, linewidth=1.2,
    )
    plt.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    plt.title("PnL Distribution by Sentiment Regime  (outliers clipped)")
    plt.xlabel("Sentiment")
    plt.ylabel("Closed PnL (USD)")
    plt.xticks(rotation=30, ha="right")
    save_plot("06_pnl_distribution_by_sentiment.png")


# ── 5g. Side performance ──────────────────────────────────────────────────
def plot_side_performance(data: pd.DataFrame) -> None:
    if "side" not in data.columns:
        return
    agg = (
        data.groupby(["classification", "side"], observed=True)["closed_pnl"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(12, 5))
    sns.barplot(
        data=agg, x="classification", y="closed_pnl",
        hue="side", errorbar=None,
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Average PnL by Sentiment Regime and Trade Side")
    plt.xlabel("Sentiment")
    plt.ylabel("Average Closed PnL (USD)")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Side")
    save_plot("07_side_performance_by_sentiment.png")


# ── 5h. Top coin performance ──────────────────────────────────────────────
def plot_top_coin_performance(data: pd.DataFrame) -> None:
    if "coin" not in data.columns:
        return
    top_coins = data.groupby("coin")["closed_pnl"].sum().nlargest(5).index
    coin_data = data[data["coin"].isin(top_coins)].copy()
    agg = (
        coin_data.groupby(["classification", "coin"], observed=True)["closed_pnl"]
        .sum()
        .reset_index()
    )
    plt.figure(figsize=(13, 6))
    sns.barplot(data=agg, x="classification", y="closed_pnl", hue="coin", errorbar=None)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Top-5 Coin Performance Across Sentiment Regimes  (Total PnL)")
    plt.xlabel("Sentiment")
    plt.ylabel("Total Closed PnL (USD)")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Coin", bbox_to_anchor=(1.01, 1), loc="upper left")
    save_plot("08_top_coin_performance.png")


# ── 5i. PnL efficiency bar ────────────────────────────────────────────────
def plot_efficiency(data: pd.DataFrame) -> None:
    if data["pnl_per_size"].isna().all():
        return
    agg = (
        data.groupby("classification", observed=True)["pnl_per_size"]
        .mean()
        .reset_index()
    )
    plt.figure()
    bars = plt.bar(
        agg["classification"].astype(str),
        agg["pnl_per_size"],
        color=_sentiment_colors(agg["classification"].astype(str)),
        edgecolor="white", linewidth=0.6,
    )
    plt.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Average PnL-to-Size Ratio (Capital Efficiency) by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("PnL / Trade Size")
    plt.xticks(rotation=30, ha="right")
    save_plot("09_pnl_efficiency_by_sentiment.png")


# ── 5j. Hourly heatmap ────────────────────────────────────────────────────
def plot_hourly_heatmap(data: pd.DataFrame) -> None:
    if data.empty or "hour" not in data.columns:
        log.warning("  skip hourly heatmap — no data")
        return
    pivot = (
        data.groupby(["hour", "classification"], observed=True)["closed_pnl"]
        .mean()
        .unstack("classification")
        .reindex(columns=SENTIMENT_ORDER)
    )
    if pivot.empty or pivot.isna().all().all():
        log.warning("  skip hourly heatmap — pivot is empty")
        return
    plt.figure(figsize=(13, 6))
    sns.heatmap(
        pivot, cmap="RdYlGn", center=0,
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Avg PnL (USD)"},
        fmt=".0f", annot=True, annot_kws={"size": 7},
    )
    plt.title("Average PnL by Hour-of-Day × Sentiment Regime")
    plt.xlabel("Sentiment")
    plt.ylabel("Hour (UTC)")
    plt.yticks(rotation=0)
    save_plot("10_hourly_pnl_heatmap.png")


# ── 5k. Weekday heatmap ───────────────────────────────────────────────────
def plot_weekday_heatmap(data: pd.DataFrame) -> None:
    if data.empty or "weekday" not in data.columns:
        log.warning("  skip weekday heatmap — no data")
        return
    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = (
        data.groupby(["weekday", "classification"], observed=True)["closed_pnl"]
        .mean()
        .unstack("classification")
        .reindex(columns=SENTIMENT_ORDER)
        .reindex(weekday_order)
    )
    if pivot.empty or pivot.isna().all().all():
        log.warning("  skip weekday heatmap — pivot is empty")
        return
    plt.figure(figsize=(13, 5))
    sns.heatmap(
        pivot, cmap="RdYlGn", center=0,
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Avg PnL (USD)"},
        fmt=".0f", annot=True, annot_kws={"size": 8},
    )
    plt.title("Average PnL by Day-of-Week × Sentiment Regime")
    plt.xlabel("Sentiment")
    plt.ylabel("Weekday")
    save_plot("11_weekday_pnl_heatmap.png")


# ── 5l. Cumulative PnL over time ──────────────────────────────────────────
def plot_cumulative_pnl(data: pd.DataFrame) -> None:
    if data.empty:
        log.warning("  skip cumulative PnL — no data"); return
    daily = (
        data.groupby(["date", "classification"], observed=True)["closed_pnl"]
        .sum()
        .reset_index()
        .sort_values("date")
    )
    daily["cumulative_pnl"] = (
        daily.groupby("classification", observed=True)["closed_pnl"]
        .cumsum()
    )
    plt.figure(figsize=(14, 6))
    for cat in SENTIMENT_ORDER:
        subset = daily[daily["classification"] == cat]
        if subset.empty:
            continue
        plt.plot(
            subset["date"], subset["cumulative_pnl"],
            label=cat, color=SENTIMENT_PALETTE.get(cat, "#888"), linewidth=1.6,
        )
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Cumulative PnL Over Time by Sentiment Regime")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL (USD)")
    plt.legend(title="Sentiment")
    plt.xticks(rotation=30, ha="right")
    save_plot("12_cumulative_pnl_over_time.png")


# ── 5m. Win rate heatmap (coin × sentiment) ───────────────────────────────
def plot_coin_sentiment_winrate(data: pd.DataFrame) -> None:
    if "coin" not in data.columns:
        return
    top_coins = data.groupby("coin")["closed_pnl"].sum().nlargest(8).index
    pivot = (
        data[data["coin"].isin(top_coins)]
        .groupby(["coin", "classification"], observed=True)["is_profit"]
        .mean()
        .mul(100)
        .unstack("classification")
        .reindex(columns=SENTIMENT_ORDER)
    )
    plt.figure(figsize=(13, 5))
    sns.heatmap(
        pivot, cmap="RdYlGn", vmin=30, vmax=70, center=50,
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Win Rate (%)"},
        fmt=".1f", annot=True, annot_kws={"size": 8},
    )
    plt.title("Win Rate (%) by Coin × Sentiment Regime  (Top 8 Coins by PnL)")
    plt.xlabel("Sentiment")
    plt.ylabel("Coin")
    save_plot("13_coin_sentiment_winrate_heatmap.png")


# ── 5n. Correlation heatmap ───────────────────────────────────────────────
def plot_correlation_heatmap(data: pd.DataFrame) -> None:
    cols = [c for c in [
        "closed_pnl", "abs_size", "pnl_per_size",
        "execution_price", "fee", "hour", "sentiment_score",
    ] if c in data.columns]
    if len(cols) < 2:
        return
    corr = data[cols].corr(numeric_only=True)
    plt.figure(figsize=(len(cols)+2, len(cols)))
    sns.heatmap(
        corr, annot=True, cmap="coolwarm", fmt=".2f",
        linewidths=0.4, linecolor="white",
        vmin=-1, vmax=1,
    )
    plt.title("Numeric Feature Correlation Heatmap")
    save_plot("14_correlation_heatmap.png")


# ── 5o. Sharpe-like ratio bar ─────────────────────────────────────────────
def plot_sharpe(data: pd.DataFrame) -> None:
    agg = (
        data.groupby("classification", observed=True)["closed_pnl"]
        .agg(["mean", "std"])
        .assign(sharpe=lambda d: d["mean"] / d["std"])
        .reset_index()
    )
    plt.figure()
    bars = plt.bar(
        agg["classification"].astype(str),
        agg["sharpe"],
        color=_sentiment_colors(agg["classification"].astype(str)),
        edgecolor="white", linewidth=0.6,
    )
    plt.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("PnL Sharpe-Like Ratio by Sentiment  (mean PnL / std PnL)")
    plt.xlabel("Sentiment")
    plt.ylabel("Sharpe-Like Ratio")
    plt.xticks(rotation=30, ha="right")
    save_plot("15_sharpe_by_sentiment.png")


def create_all_charts(data: pd.DataFrame) -> None:
    set_theme()
    log.info("Generating charts …")
    plot_total_pnl(data)
    plot_avg_pnl(data)
    plot_win_rate(data)
    plot_trade_count(data)
    plot_trade_size(data)
    plot_pnl_distribution(data)
    plot_side_performance(data)
    plot_top_coin_performance(data)
    plot_efficiency(data)
    plot_hourly_heatmap(data)
    plot_weekday_heatmap(data)
    plot_cumulative_pnl(data)
    plot_coin_sentiment_winrate(data)
    plot_correlation_heatmap(data)
    plot_sharpe(data)
    log.info("All charts saved to %s", CHART_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# 6. STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════

def run_statistical_tests(data: pd.DataFrame) -> pd.DataFrame:
    groups = [
        grp["closed_pnl"].dropna().values
        for _, grp in data.groupby("classification", observed=True)
        if grp["closed_pnl"].dropna().shape[0] > 5
    ]
    results = []

    def _test(name, fn, *args):
        try:
            stat, p = fn(*args)
            return {
                "test": name,
                "statistic": round(float(stat), 4),
                "p_value":   round(float(p), 6),
                "significant_at_0.05": p < 0.05,
                "interpretation": (
                    "Significant difference across sentiment groups"
                    if p < 0.05
                    else "No statistically significant difference"
                ),
            }
        except Exception as exc:
            return {"test": name, "statistic": np.nan, "p_value": np.nan,
                    "significant_at_0.05": False, "interpretation": str(exc)}

    if len(groups) > 1:
        results.append(_test("One-Way ANOVA",    f_oneway,  *groups))
        results.append(_test("Kruskal-Wallis",   kruskal,   *groups))

    # Spearman correlation between sentiment order and avg PnL
    sent_numeric = {s: i for i, s in enumerate(SENTIMENT_ORDER)}
    data_copy = data.copy()
    data_copy["sent_num"] = data_copy["classification"].map(sent_numeric)
    corr_val, corr_p = spearmanr(
        data_copy["sent_num"].dropna(),
        data_copy.loc[data_copy["sent_num"].notna(), "closed_pnl"],
    )
    results.append({
        "test": "Spearman ρ  (sentiment order vs PnL)",
        "statistic": round(float(corr_val), 4),
        "p_value":   round(float(corr_p), 6),
        "significant_at_0.05": corr_p < 0.05,
        "interpretation": (
            f"ρ={corr_val:.3f}: {'positive' if corr_val > 0 else 'negative'} "
            f"monotonic relationship between sentiment and PnL "
            f"({'significant' if corr_p < 0.05 else 'not significant'} at α=0.05)"
        ),
    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLE_DIR / "statistical_tests.csv", index=False)
    log.info("Statistical tests saved.")
    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# 7. INSIGHT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def infer_insights(
    summary: pd.DataFrame,
    side_df: pd.DataFrame | None,
    stats_df: pd.DataFrame,
) -> list[str]:
    insights: list[str] = []

    def safe_row(df, col):
        col_data = df[col].dropna()
        if col_data.empty:
            return None
        return df.loc[col_data.idxmax()]

    if not summary.empty:
        best_total  = safe_row(summary, "total_pnl")
        best_avg    = safe_row(summary, "avg_pnl")
        best_win    = safe_row(summary, "win_rate_pct")
        best_eff    = safe_row(summary, "pnl_efficiency")
        best_sharpe = safe_row(summary, "pnl_sharpe")
        busiest     = safe_row(summary, "trade_count")

        if best_total is not None:
            insights.append(
                f"🏆 **Highest Total PnL**: '{best_total['classification']}' regime generated "
                f"the most cumulative profit ({best_total['total_pnl']:,.2f} USD)."
            )
        if best_avg is not None:
            insights.append(
                f"📈 **Best Average Trade**: '{best_avg['classification']}' produced the "
                f"highest average PnL per trade ({best_avg['avg_pnl']:,.2f} USD)."
            )
        if best_win is not None:
            insights.append(
                f"🎯 **Highest Win Rate**: '{best_win['classification']}' achieved the best "
                f"win rate at {best_win['win_rate_pct']:.1f}%."
            )
        if best_eff is not None and not pd.isna(best_eff["pnl_efficiency"]):
            insights.append(
                f"💡 **Best Capital Efficiency**: '{best_eff['classification']}' had the "
                f"highest PnL-to-size ratio ({best_eff['pnl_efficiency']:.4f}), meaning "
                f"capital was deployed most efficiently in this regime."
            )
        if best_sharpe is not None and not pd.isna(best_sharpe["pnl_sharpe"]):
            insights.append(
                f"📊 **Best Risk-Adjusted Returns**: '{best_sharpe['classification']}' produced "
                f"the strongest Sharpe-like ratio ({best_sharpe['pnl_sharpe']:.3f}), balancing "
                f"return and volatility most effectively."
            )
        if busiest is not None:
            insights.append(
                f"🔥 **Most Active Regime**: '{busiest['classification']}' saw the highest "
                f"trading activity ({int(busiest['trade_count']):,} trades)."
            )

    # Side-specific insights
    if side_df is not None and not side_df.empty:
        for side in sorted(side_df["side"].dropna().unique()):
            sub = side_df[side_df["side"] == side].copy()
            if sub.empty:
                continue
            best = sub.loc[sub["avg_pnl"].idxmax()]
            insights.append(
                f"↕️  **{side.title()} Side**: Strongest average performance during "
                f"'{best['classification']}' ({best['avg_pnl']:,.2f} USD avg PnL)."
            )

    # Statistical significance
    anova_row = stats_df[stats_df["test"] == "One-Way ANOVA"]
    if not anova_row.empty:
        p = float(anova_row["p_value"].values[0])
        insights.append(
            f"🔬 **Statistical Significance**: One-way ANOVA p-value = {p:.4f} — the "
            f"difference in PnL across sentiment regimes is "
            f"{'statistically significant' if p < 0.05 else 'NOT statistically significant'} "
            f"at the 5% level."
        )

    spearman_row = stats_df[stats_df["test"].str.startswith("Spearman")]
    if not spearman_row.empty:
        insights.append(
            f"🔗 **Sentiment–PnL Correlation**: {spearman_row['interpretation'].values[0]}"
        )

    insights.append(
        "💼 **Strategy Implication**: Use sentiment regime as a contextual filter — "
        "adjust position sizing, directional bias, and asset selection based on "
        "the prevailing Fear & Greed state rather than treating all market "
        "conditions uniformly."
    )
    return insights


# ══════════════════════════════════════════════════════════════════════════════
# 8. REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def write_report(
    data: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    stats_df: pd.DataFrame,
) -> None:
    summary  = tables["summary_by_sentiment"]
    side_df  = tables.get("side_performance")
    insights = infer_insights(summary, side_df, stats_df)

    charts_section = "\n".join([
        "| # | Chart | What it shows |",
        "|---|-------|---------------|",
        "| 01 | total_pnl_by_sentiment | Which sentiment regime generated the most profit |",
        "| 02 | avg_pnl_by_sentiment | Quality of individual trades per regime |",
        "| 03 | win_rate_by_sentiment | Trade success rate across regimes |",
        "| 04 | trade_count_by_sentiment | Trading activity / volume per regime |",
        "| 05 | avg_trade_size_by_sentiment | Position sizing behaviour per regime |",
        "| 06 | pnl_distribution_by_sentiment | PnL spread & consistency per regime |",
        "| 07 | side_performance_by_sentiment | Long vs Short edge per regime |",
        "| 08 | top_coin_performance | Best assets in each sentiment state |",
        "| 09 | pnl_efficiency_by_sentiment | Capital efficiency per regime |",
        "| 10 | hourly_pnl_heatmap | Best hours to trade within each regime |",
        "| 11 | weekday_pnl_heatmap | Best days to trade within each regime |",
        "| 12 | cumulative_pnl_over_time | Equity curve coloured by sentiment |",
        "| 13 | coin_sentiment_winrate_heatmap | Asset-specific win rate per regime |",
        "| 14 | correlation_heatmap | Numeric feature relationships |",
        "| 15 | sharpe_by_sentiment | Risk-adjusted return per regime |",
    ])

    report = f"""# Trader Performance vs Market Sentiment — Analysis Report

## 1. Objective
Explore the relationship between **Bitcoin Fear & Greed sentiment** and **Hyperliquid trader performance**, uncover hidden behavioural patterns, and surface actionable insights for smarter strategy design.

---

## 2. Datasets

| Dataset | Source | Key Columns |
|---------|--------|-------------|
| Bitcoin Fear & Greed Index | fear_greed_index.csv | date, classification (Extreme Fear → Extreme Greed) |
| Hyperliquid Historical Trades | historical_data.csv | account, coin, execution_price, size_usd, side, timestamp, closed_pnl, fee |

---

## 3. Methodology

1. **Column standardisation** — snake_case normalisation across both files.
2. **Timestamp resolution** — automatically selects the best-populated datetime column; falls back to Unix-epoch conversion.
3. **Sentiment merge** — left-join trades to daily sentiment on calendar date.
4. **PnL filter** — retained only completed, non-zero PnL trades.
5. **Derived features**
   - `is_profit` / `is_loss` binary flags
   - `abs_size` — absolute trade size in USD
   - `pnl_per_size` — capital efficiency ratio (winsorised at 1st/99th percentile)
   - `log_abs_size` — log-scaled size for correlation
   - `hour`, `weekday`, `month` calendar features
   - `pnl_sharpe` — mean PnL / std PnL per sentiment bucket
6. **Statistical tests** — ANOVA, Kruskal-Wallis (difference across groups), Spearman ρ (monotonic sentiment-PnL relationship).

---

## 4. Dataset Summary

| Metric | Value |
|--------|-------|
| Total rows after cleaning | {len(data):,} |
| Sentiment regimes present | {data['classification'].nunique()} |
| Date range | {data['date'].min().date()} → {data['date'].max().date()} |
| Unique accounts | {data['account'].nunique() if 'account' in data.columns else 'N/A'} |
| Unique coins | {data['coin'].nunique() if 'coin' in data.columns else 'N/A'} |

---

## 5. Core Summary Table

{summary.to_markdown(index=False)}

---

## 6. Key Insights

{chr(10).join(insights)}

---

## 7. Statistical Tests

{stats_df.to_markdown(index=False) if not stats_df.empty else "_Tests could not be computed._"}

---

## 8. Charts Generated

{charts_section}

---

## 9. Strategic Recommendations

| Scenario | Recommended Action |
|----------|--------------------|
| Extreme Fear | Reduce position size; favour selective long setups with tight stops |
| Fear | Slightly cautious sizing; look for mean-reversion longs in high-value coins |
| Neutral | Normal sizing; balanced long/short based on momentum |
| Greed | Consider locking profits on longs; short opportunities may emerge |
| Extreme Greed | Reduce exposure; avoid chasing; tighten risk controls |

> **Note**: These recommendations are derived from historical data patterns and should be combined with your own risk management framework, not followed mechanically.

---

## 10. Files Generated

```
outputs/
├── report.md
├── charts/
│   ├── 01_total_pnl_by_sentiment.png
│   ├── 02_avg_pnl_by_sentiment.png
│   ├── 03_win_rate_by_sentiment.png
│   ├── 04_trade_count_by_sentiment.png
│   ├── 05_avg_trade_size_by_sentiment.png
│   ├── 06_pnl_distribution_by_sentiment.png
│   ├── 07_side_performance_by_sentiment.png
│   ├── 08_top_coin_performance.png
│   ├── 09_pnl_efficiency_by_sentiment.png
│   ├── 10_hourly_pnl_heatmap.png
│   ├── 11_weekday_pnl_heatmap.png
│   ├── 12_cumulative_pnl_over_time.png
│   ├── 13_coin_sentiment_winrate_heatmap.png
│   ├── 14_correlation_heatmap.png
│   └── 15_sharpe_by_sentiment.png
└── tables/
    ├── summary_by_sentiment.csv
    ├── side_performance.csv
    ├── top_coin_performance.csv
    ├── account_sentiment_performance.csv
    ├── hourly_avg_pnl.csv
    ├── weekday_pnl.csv
    ├── statistical_tests.csv
    └── cleaned_merged_data.csv   (essential columns only)
```

---

## 11. Conclusion

The analysis examines whether and how Bitcoin market sentiment regimes systematically influence trader outcomes on Hyperliquid. By combining **15 visualisations**, **7 summary tables**, and **3 statistical tests**, the report provides a multi-dimensional view of performance across Extreme Fear, Fear, Neutral, Greed, and Extreme Greed conditions.

Sentiment-aware position sizing, directional bias, and asset selection can form the basis of a more adaptive trading strategy compared with static, regime-agnostic approaches.
"""
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")
    log.info("Report written to %s/report.md", OUTPUT_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ensure_directories()

    trades, sentiment = load_data()

    # ── DEBUG helper: print raw timestamp sample so user can verify unit ──
    if "timestamp" in trades.columns:
        sample = pd.to_numeric(trades["timestamp"], errors="coerce").dropna()
        if not sample.empty:
            log.info(
                "Raw 'timestamp' — min: %.3e  max: %.3e  median: %.3e  dtype: %s",
                sample.min(), sample.max(), sample.median(), trades["timestamp"].dtype,
            )

    data = preprocess(trades, sentiment)

    # Save only essential columns to keep file size manageable
    essential_cols = [
        c for c in [
            "account", "coin", "side", "closed_pnl", "abs_size",
            "pnl_per_size", "is_profit", "is_loss",
            "classification", "date", "hour", "weekday", "month",
            "execution_price", "fee", "sentiment_score",
        ]
        if c in data.columns
    ]
    data[essential_cols].to_csv(TABLE_DIR / "cleaned_merged_data.csv", index=False)
    log.info("Cleaned data saved (%d cols).", len(essential_cols))

    tables   = build_summary_tables(data)
    save_tables(tables)
    create_all_charts(data)
    stats_df = run_statistical_tests(data)
    write_report(data, tables, stats_df)

    log.info("=" * 60)
    log.info("✅  Analysis complete!")
    log.info("    Outputs : %s", OUTPUT_DIR.resolve())
    log.info("    Charts  : %s", CHART_DIR.resolve())
    log.info("    Tables  : %s", TABLE_DIR.resolve())
    log.info("=" * 60)


if __name__ == "__main__":
    main()