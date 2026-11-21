"""
analysis.py - data loading, cleaning, statistics and the prediction model.

This module has NO Streamlit code, so every function can be tested on its own.
The methodology is the one used in the original maths_pbl.py:
  * only rows with Branch == "Overall" are used for college-level analysis
  * the year is the first 4-digit number of "graduation_year" (2023-24 -> 2023)
  * prediction = one Linear Regression per college (year -> placement %)
  * evaluation = for each college, train on all years except the last one,
    test on the last year (needs >= 3 yearly records), then pool all test points
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "placement_data_filled.csv"

REQUIRED_COLUMNS = [
    "college_name", "Branch", "graduation_year",
    "total_student", "placed_students", "placement_percentage",
]
OPTIONAL_COLUMNS = ["average_package", "highest_package", "total companies"]
NUMERIC_COLUMNS = [
    "total_student", "placed_students", "placement_percentage",
    "average_package", "highest_package",
]

PCT_RECOMPUTED = "recomputed"   # placement % = placed / total x 100
PCT_FROM_CSV = "csv"            # placement % exactly as written in the CSV


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw_csv(source) -> pd.DataFrame:
    """Read a CSV from a path, bytes, or file-like object. Column names are stripped."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        df = pd.read_csv(source, encoding="utf-8-sig")
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, encoding="latin-1")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def missing_required_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def _normalise_branch(value: str) -> str:
    value = str(value).strip()
    return "Overall" if value.lower() == "overall" else value.upper()


def clean_data(raw: pd.DataFrame, pct_mode: str = PCT_RECOMPUTED):
    """
    Clean the raw dataset.

    Returns (clean_df, issues_df) where issues_df lists every data problem that
    was detected and what was done about it. Nothing is silently changed and
    missing values are never replaced by zero.
    """
    df = raw.copy()
    issues: list[dict] = []

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # --- text columns -------------------------------------------------------
    df["college_name"] = df["college_name"].astype(str).str.strip()
    df["graduation_year"] = df["graduation_year"].astype(str).str.strip()
    old_branch = df["Branch"].astype(str).str.strip()
    df["Branch"] = old_branch.map(_normalise_branch)
    changed = df.loc[old_branch != df["Branch"], ["college_name", "Branch"]]
    renamed = sorted({f"{o} -> {n}" for o, n in zip(old_branch[old_branch != df["Branch"]],
                                                    df.loc[old_branch != df["Branch"], "Branch"])})
    if renamed:
        issues.append({
            "Issue": "Inconsistent branch spelling / letter case",
            "Details": ", ".join(renamed) + f" ({len(changed)} rows)",
            "Action taken": "Branch names made consistent (upper case, 'Overall' kept as is).",
        })

    # --- numeric columns ----------------------------------------------------
    for col in NUMERIC_COLUMNS:
        before_na = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        newly_na = df[col].isna().sum() - before_na
        if newly_na > 0:
            issues.append({
                "Issue": f"Non-numeric values in '{col}'",
                "Details": f"{newly_na} value(s) could not be read as numbers",
                "Action taken": "Treated as missing (not as zero).",
            })

    # --- year ---------------------------------------------------------------
    df["year_number"] = pd.to_numeric(
        df["graduation_year"].str.extract(r"(\d{4})")[0], errors="coerce"
    )
    bad_year = df["year_number"].isna()
    if bad_year.any():
        issues.append({
            "Issue": "Unreadable graduation_year",
            "Details": f"{int(bad_year.sum())} row(s)",
            "Action taken": "Rows removed (year is needed for every analysis).",
        })
        df = df.loc[~bad_year].copy()
    df["year_number"] = df["year_number"].astype(int)

    # --- duplicates ---------------------------------------------------------
    key = ["college_name", "Branch", "graduation_year"]
    dup = df.duplicated(subset=key, keep="first")
    if dup.any():
        issues.append({
            "Issue": "Duplicate college + branch + year records",
            "Details": f"{int(dup.sum())} duplicate row(s)",
            "Action taken": "First record kept, duplicates removed.",
        })
        df = df.loc[~dup].copy()

    # --- placement percentage ----------------------------------------------
    df["placement_percentage_csv"] = df["placement_percentage"]
    valid_total = df["total_student"] > 0
    recomputed = pd.Series(np.nan, index=df.index, dtype=float)
    recomputed.loc[valid_total] = (
        df.loc[valid_total, "placed_students"] / df.loc[valid_total, "total_student"] * 100
    )
    df["placement_percentage_recomputed"] = recomputed

    mismatch = (
        df["placement_percentage_csv"].notna()
        & recomputed.notna()
        & ((df["placement_percentage_csv"] - recomputed).abs() > 0.01)
    )
    for idx in df.index[mismatch]:
        r = df.loc[idx]
        issues.append({
            "Issue": "Placement % in CSV does not match placed / total",
            "Details": (
                f"{r['college_name']} | {r['Branch']} | {r['graduation_year']}: "
                f"CSV says {r['placement_percentage_csv']:.2f}%, but "
                f"{int(r['placed_students'])} / {int(r['total_student'])} = "
                f"{r['placement_percentage_recomputed']:.2f}%"
            ),
            "Action taken": (
                "Using the recomputed value." if pct_mode == PCT_RECOMPUTED
                else "Using the CSV value (as selected in the sidebar)."
            ),
        })

    if pct_mode == PCT_RECOMPUTED:
        df["placement_percentage"] = recomputed.where(recomputed.notna(), df["placement_percentage_csv"])
    else:
        df["placement_percentage"] = df["placement_percentage_csv"]

    # --- missing totals -----------------------------------------------------
    no_total = df["total_student"].isna()
    if no_total.any():
        rows = df.loc[no_total]
        where = "; ".join(
            f"{r.college_name} | {r.Branch} | {r.graduation_year}" for r in rows.itertuples()
        )
        issues.append({
            "Issue": "total_student is missing",
            "Details": f"{int(no_total.sum())} row(s): {where}",
            "Action taken": (
                "Placement % left blank for these rows (it cannot be computed). "
                "Placed-student counts are kept."
            ),
        })

    # --- impossible values --------------------------------------------------
    too_many = df["placed_students"] > df["total_student"]
    if too_many.any():
        issues.append({
            "Issue": "placed_students is larger than total_student",
            "Details": f"{int(too_many.sum())} row(s)",
            "Action taken": "Rows kept but their % is treated as invalid and left blank.",
        })
        df.loc[too_many, "placement_percentage"] = np.nan
    out_of_range = df["placement_percentage"].notna() & ~df["placement_percentage"].between(0, 100)
    if out_of_range.any():
        df.loc[out_of_range, "placement_percentage"] = np.nan

    df = df.sort_values(["college_name", "year_number", "Branch"]).reset_index(drop=True)
    issues_df = pd.DataFrame(issues, columns=["Issue", "Details", "Action taken"])
    return df, issues_df


def split_levels(clean: pd.DataFrame):
    """Return (overall_df, branch_df). They are never mixed together."""
    overall = clean[clean["Branch"] == "Overall"].copy()
    branches = clean[clean["Branch"] != "Overall"].copy()
    return overall, branches


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that have a valid placement percentage."""
    return df.dropna(subset=["placement_percentage"]).copy()


def coverage_report(clean: pd.DataFrame) -> pd.DataFrame:
    """Compare branch rows with the Overall row for each college + year."""
    overall, branches = split_levels(clean)
    rows = []
    for (college, year), o in overall.groupby(["college_name", "graduation_year"]):
        b = branches[(branches["college_name"] == college) & (branches["graduation_year"] == year)]
        o_total = o["total_student"].iloc[0]
        o_placed = o["placed_students"].iloc[0]
        b_total = b["total_student"].sum()
        b_placed = b["placed_students"].sum()
        rows.append({
            "College": college, "Year": year,
            "Overall: total": o_total, "Branches: total": b_total,
            "Overall: placed": o_placed, "Branches: placed": b_placed,
            "Branches add up to Overall": bool(
                pd.notna(o_total) and o_total == b_total and o_placed == b_placed
            ),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------
def modes_of(values: pd.Series, decimals: int = 2) -> list[float]:
    """Mode(s) of the values (rounded). Empty list when no value repeats."""
    s = pd.Series(values).dropna().round(decimals)
    if s.empty:
        return []
    counts = s.value_counts()
    if counts.max() < 2:
        return []
    return sorted(counts[counts == counts.max()].index.tolist())


def summary_stats(values: pd.Series) -> dict:
    """Mean, median, mode, variance, std. dev., min, max ... (sample variance, ddof=1)."""
    s = pd.Series(values).dropna().astype(float)
    n = int(s.size)
    if n == 0:
        return {"count": 0}
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else np.nan
    return {
        "count": n,
        "mean": mean,
        "median": float(s.median()),
        "modes": modes_of(s),
        "variance": float(s.var(ddof=1)) if n > 1 else np.nan,
        "std": std,
        "min": float(s.min()),
        "q1": float(q1),
        "q3": float(q3),
        "max": float(s.max()),
        "range": float(s.max() - s.min()),
        "iqr": float(q3 - q1),
        "cv": float(std / mean * 100) if n > 1 and mean != 0 else np.nan,
    }


def format_modes(modes: list[float]) -> str:
    if not modes:
        return "No mode (no value repeats)"
    shown = ", ".join(f"{m:.2f}" for m in modes[:3])
    return shown + (" ..." if len(modes) > 3 else "")


def summary_table(stats: dict) -> pd.DataFrame:
    if stats.get("count", 0) == 0:
        return pd.DataFrame(columns=["Statistic", "Value"])
    f = lambda x: "n/a" if pd.isna(x) else round(float(x), 2)
    rows = [
        ("Number of records (n)", stats["count"]),
        ("Mean", f(stats["mean"])),
        ("Median", f(stats["median"])),
        ("Mode", format_modes(stats["modes"])),
        ("Variance (sample)", f(stats["variance"])),
        ("Standard deviation (sample)", f(stats["std"])),
        ("Minimum", f(stats["min"])),
        ("First quartile (Q1)", f(stats["q1"])),
        ("Third quartile (Q3)", f(stats["q3"])),
        ("Maximum", f(stats["max"])),
        ("Range (max - min)", f(stats["range"])),
        ("Inter-quartile range (IQR)", f(stats["iqr"])),
        ("Coefficient of variation (%)", f(stats["cv"])),
    ]
    return pd.DataFrame(rows, columns=["Statistic", "Value"]).astype({"Value": str})


def grouped_stats(df: pd.DataFrame, group_col: str = "college_name") -> pd.DataFrame:
    rows = []
    for name, g in df.groupby(group_col):
        st_ = summary_stats(g["placement_percentage"])
        if st_["count"] == 0:
            continue
        rows.append({
            group_col: name, "Records": st_["count"],
            "Mean": st_["mean"], "Median": st_["median"],
            "Mode": format_modes(st_["modes"]),
            "Variance": st_["variance"], "Std. deviation": st_["std"],
            "Minimum": st_["min"], "Maximum": st_["max"],
        })
    out = pd.DataFrame(rows)
    num = out.select_dtypes("number").columns.drop("Records", errors="ignore")
    out[num] = out[num].round(2)
    return out


def frequency_table(values: pd.Series, width: int = 10) -> pd.DataFrame:
    """Grouped frequency distribution with class width `width` (0-10, 10-20, ... 90-100)."""
    s = pd.Series(values).dropna().astype(float)
    edges = np.arange(0, 100 + width, width)
    counts, _ = np.histogram(s, bins=edges)   # last class includes 100
    labels = [f"{int(a)} - {int(b)}" for a, b in zip(edges[:-1], edges[1:])]
    total = counts.sum()
    out = pd.DataFrame({"Class interval (%)": labels, "Frequency": counts})
    out["Relative frequency (%)"] = (counts / total * 100).round(2) if total else 0.0
    out["Cumulative frequency"] = counts.cumsum()
    return out


def explain_stats(stats: dict, label: str) -> list[str]:
    """Plain-language explanation of each statistic."""
    if stats.get("count", 0) == 0:
        return ["There are no valid records to describe."]
    n, mean, med = stats["count"], stats["mean"], stats["median"]
    lines = [
        f"**Mean = {mean:.2f}%** - if we add all {n} {label} placement percentages and divide by {n}, "
        f"we get {mean:.2f}%. It is the 'average' placement.",
        f"**Median = {med:.2f}%** - the middle value when all percentages are arranged in order. "
        "Half of the records are below it and half are above it.",
    ]
    if abs(mean - med) < 2:
        lines.append("The mean and median are close, so the data is fairly balanced around the centre.")
    elif mean > med:
        lines.append("The mean is higher than the median, so a few high placement values are pulling the average up.")
    else:
        lines.append("The mean is lower than the median, so a few low placement values are pulling the average down.")
    if stats["modes"]:
        lines.append(f"**Mode = {format_modes(stats['modes'])}%** - the value(s) that occur most often.")
    else:
        lines.append(
            "**Mode:** no placement percentage repeats, so there is no mode. "
            "(For measured percentages with many decimals, a mode is rarely meaningful.)"
        )
    if pd.notna(stats["std"]):
        lines.append(
            f"**Standard deviation = {stats['std']:.2f}** (variance = {stats['variance']:.2f}) - on average the "
            f"records differ from the mean by about {stats['std']:.1f} percentage points. "
            + ("This is large, so placements differ a lot between records."
               if stats["std"] > 15 else
               "This is moderate, so placements are reasonably similar across records.")
        )
    lines.append(
        f"**Minimum = {stats['min']:.2f}%, Maximum = {stats['max']:.2f}%** - the lowest and highest records; "
        f"the range is {stats['range']:.2f} percentage points."
    )
    lines.append(
        f"**Quartiles:** 25% of records are at or below {stats['q1']:.2f}% and 75% are at or below "
        f"{stats['q3']:.2f}%. The middle 50% of the data lies within an IQR of {stats['iqr']:.2f} points."
    )
    return lines


# --------------------------------------------------------------------------
# Prediction (Linear Regression, same method as the original project)
# --------------------------------------------------------------------------
def next_year(overall: pd.DataFrame) -> int:
    """Year being predicted: the year after the latest record (2025-26 -> 2026-27)."""
    return int(overall["year_number"].max()) + 1


def year_label(year_number: int) -> str:
    return f"{year_number}-{str(year_number + 1)[-2:]}"


def fit_predict(college_df: pd.DataFrame, target_year: int) -> dict:
    """Fit placement% ~ year for one college and predict `target_year`."""
    d = usable(college_df).sort_values("year_number")
    n = len(d)
    if n < 2 or d["year_number"].nunique() < 2:
        return {"ok": False, "n": n,
                "message": "At least 2 yearly records (in different years) are needed for a prediction."}
    model = LinearRegression()
    model.fit(d[["year_number"]], d["placement_percentage"])
    raw_pred = float(model.predict(pd.DataFrame({"year_number": [target_year]}))[0])
    return {
        "ok": True, "n": n, "model": model,
        "slope": float(model.coef_[0]), "intercept": float(model.intercept_),
        "raw_prediction": raw_pred,
        "prediction": float(np.clip(raw_pred, 0, 100)),
        "clipped": not (0 <= raw_pred <= 100),
        "train_r2": float(model.score(d[["year_number"]], d["placement_percentage"])) if n > 2 else np.nan,
        "data": d,
    }


def predict_all(overall: pd.DataFrame, target_year: int) -> pd.DataFrame:
    rows = []
    for college, g in overall.groupby("college_name"):
        r = fit_predict(g, target_year)
        d = usable(g).sort_values("year_number")
        rows.append({
            "College": college,
            "Records used": r["n"],
            "Latest actual (%)": round(float(d["placement_percentage"].iloc[-1]), 2) if len(d) else np.nan,
            "Trend (points / year)": round(r["slope"], 2) if r["ok"] else np.nan,
            f"Estimate {year_label(target_year)} (%)": round(r["prediction"], 2) if r["ok"] else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Model evaluation (same method as the original project)
# --------------------------------------------------------------------------
def evaluate_model(overall: pd.DataFrame) -> dict:
    """
    For every college with >= 3 yearly records: train on all years except the
    last, test on the last year. Metrics are computed on the pooled test points.
    """
    rows = []
    skipped = []
    for college, g in overall.groupby("college_name"):
        d = usable(g).sort_values("year_number")
        if len(d) < 3:
            skipped.append(college)
            continue
        train, test = d.iloc[:-1], d.iloc[-1:]
        model = LinearRegression().fit(train[["year_number"]], train["placement_percentage"])
        pred = float(model.predict(test[["year_number"]])[0])
        actual = float(test["placement_percentage"].iloc[0])
        rows.append({
            "College": college,
            "Training years": ", ".join(train["graduation_year"]),
            "Test year": test["graduation_year"].iloc[0],
            "Actual (%)": actual, "Predicted (%)": pred,
            "Error (pred - actual)": pred - actual,
            "Absolute error": abs(pred - actual),
        })
    table = pd.DataFrame(rows)
    result = {"table": table, "skipped": skipped, "n_test": len(table)}
    if table.empty:
        return result
    a, p = table["Actual (%)"].to_numpy(), table["Predicted (%)"].to_numpy()
    mse = float(mean_squared_error(a, p))
    result.update({
        "mae": float(mean_absolute_error(a, p)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(a, p)) if len(a) >= 2 and np.var(a) > 0 else None,
    })
    return result
