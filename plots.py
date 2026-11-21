"""
plots.py - Matplotlib / Seaborn figures used by the Streamlit app.
Every function returns a matplotlib Figure (the app shows it with st.pyplot).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # no display needed (works on Streamlit Cloud)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

NAVY = "#17365D"
BLUE = "#1F4E9C"
LIGHT_BLUE = "#CFE0F7"
ACCENT = "#F59E0B"
_PALETTE = ["#1F4E9C", "#2FA4E7", "#F59E0B", "#10B981", "#6366F1", "#EF4444", "#14B8A6", "#8B5CF6"]

sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": "#C9D3E3", "grid.color": "#E6ECF5",
    "axes.titleweight": "bold", "axes.titlecolor": NAVY,
    "axes.labelcolor": NAVY, "text.color": NAVY,
    "xtick.color": "#3A4A63", "ytick.color": "#3A4A63",
})


def college_colors(colleges) -> dict:
    return {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(sorted(colleges))}


def _clean_axes(ax):
    sns.despine(ax=ax)
    ax.grid(axis="x", visible=False)


def hist_figure(values: pd.Series, title: str) -> plt.Figure:
    s = pd.Series(values).dropna().astype(float)
    fig, ax = plt.subplots(figsize=(8, 4))
    counts, _, _ = ax.hist(s, bins=np.arange(0, 101, 10), color=BLUE, edgecolor="white", alpha=0.9)
    ax.set_ylim(0, (counts.max() if len(counts) else 1) * 1.25 + 0.5)
    if len(s):
        ax.axvline(s.mean(), color=ACCENT, lw=2, ls="--", label=f"Mean = {s.mean():.1f}%")
        ax.axvline(s.median(), color="#10B981", lw=2, ls=":", label=f"Median = {s.median():.1f}%")
        ax.legend(frameon=False)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xlabel("Placement percentage (class width = 10)")
    ax.set_ylabel("Number of records")
    ax.set_title(title)
    ax.yaxis.get_major_locator().set_params(integer=True)
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def _boxes(ax, groups: list, labels: list, colors_: list) -> None:
    """Vertical box plots with the individual records drawn on top."""
    bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                    flierprops={"marker": "o", "markersize": 4, "markerfacecolor": NAVY})
    rng = np.random.default_rng(0)  # fixed jitter so the picture is reproducible
    for i, (box, med, g, col) in enumerate(zip(bp["boxes"], bp["medians"], groups, colors_), start=1):
        box.set(facecolor=col, alpha=0.75, edgecolor=NAVY, linewidth=1.2)
        med.set(color=NAVY, linewidth=2)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(g)), g, s=14, color=NAVY, alpha=0.55, zorder=3)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylim(-3, 103)
    sns.despine(ax=ax)
    ax.grid(axis="x", visible=False)


def box_figure(df: pd.DataFrame, colors: dict, title: str) -> plt.Figure:
    d = df.dropna(subset=["placement_percentage"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 2.2]})
    _boxes(ax1, [d["placement_percentage"].to_numpy()], ["All records"], [LIGHT_BLUE])
    ax1.set_title("All records")
    ax1.set_ylabel("Placement percentage")

    order = sorted(d["college_name"].unique())
    groups = [d.loc[d["college_name"] == c, "placement_percentage"].to_numpy() for c in order]
    _boxes(ax2, groups, order, [colors.get(c, BLUE) for c in order])
    ax2.set_title("By college")
    ax2.tick_params(axis="x", rotation=15)
    fig.suptitle(title, fontweight="bold", color=NAVY)
    fig.tight_layout()
    return fig


def grouped_bar_figure(overall: pd.DataFrame, colleges: list, colors: dict) -> plt.Figure:
    d = overall[overall["college_name"].isin(colleges)].dropna(subset=["placement_percentage"])
    years = sorted(d["graduation_year"].unique())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / max(len(colleges), 1)
    for i, college in enumerate(colleges):
        vals = []
        for y in years:
            m = d[(d["college_name"] == college) & (d["graduation_year"] == y)]["placement_percentage"]
            vals.append(m.iloc[0] if len(m) else np.nan)
        x = np.arange(len(years)) + i * width - 0.4 + width / 2
        bars = ax.bar(x, vals, width * 0.92, label=college, color=colors.get(college, BLUE))
        for b, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels(years)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Placement percentage")
    ax.set_xlabel("Graduation year")
    ax.set_title("Placement percentage by year")
    ax.legend(frameon=False, ncol=min(len(colleges), 4), loc="upper center", bbox_to_anchor=(0.5, -0.15))
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def avg_bar_figure(avg: pd.Series, colors: dict, title: str) -> plt.Figure:
    avg = avg.sort_values()
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.7 * len(avg) + 1)))
    bars = ax.barh(avg.index, avg.values, color=[colors.get(c, BLUE) for c in avg.index])
    for b, v in zip(bars, avg.values):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Average placement percentage")
    ax.set_title(title)
    sns.despine(ax=ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def branch_bar_figure(branch_df: pd.DataFrame, title: str) -> plt.Figure:
    d = branch_df.dropna(subset=["placement_percentage"]).sort_values("placement_percentage")
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.45 * len(d) + 1)))
    bars = ax.barh(d["Branch"], d["placement_percentage"], color=BLUE)
    for b, v in zip(bars, d["placement_percentage"]):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Placement percentage")
    ax.set_title(title)
    sns.despine(ax=ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def prediction_figure(fit: dict, college: str, target_year: int, target_label: str) -> plt.Figure:
    d = fit["data"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(d["year_number"], d["placement_percentage"], "o-", color=BLUE, lw=2.2, ms=8, label="Actual (historical)")
    xs = np.array([d["year_number"].min(), target_year])
    ys = fit["slope"] * xs + fit["intercept"]
    ax.plot(xs, ys, "--", color=ACCENT, lw=1.8, label="Linear regression line")
    ax.scatter([target_year], [fit["prediction"]], marker="*", s=320, color=ACCENT, zorder=5,
               edgecolor=NAVY, label=f"Estimate {target_label}")
    for xv, yv in zip(d["year_number"], d["placement_percentage"]):
        ax.annotate(f"{yv:.1f}%", (xv, yv), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.annotate(f"{fit['prediction']:.1f}%", (target_year, fit["prediction"]), textcoords="offset points",
                xytext=(0, 14), ha="center", fontsize=10, fontweight="bold")
    ticks = list(d["year_number"]) + [target_year]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}-{str(t + 1)[-2:]}" for t in ticks])
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Graduation year")
    ax.set_ylabel("Placement percentage")
    ax.set_title(f"{college}: historical placement and estimate")
    ax.legend(frameon=False, loc="best")
    _clean_axes(ax)
    fig.tight_layout()
    return fig


def evaluation_figure(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(table))
    w = 0.38
    ax.bar(x - w / 2, table["Actual (%)"], w, label="Actual", color=BLUE)
    ax.bar(x + w / 2, table["Predicted (%)"], w, label="Predicted", color=ACCENT)
    for xi, a, p in zip(x, table["Actual (%)"], table["Predicted (%)"]):
        ax.text(xi - w / 2, a + 1, f"{a:.1f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, p + 1, f"{p:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(table["College"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Placement percentage")
    ax.set_title("Test year: actual vs predicted")
    ax.legend(frameon=False)
    _clean_axes(ax)
    fig.tight_layout()
    return fig
