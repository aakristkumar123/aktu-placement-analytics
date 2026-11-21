"""
AKTU PLACEMENT ANALYTICS
Statistical Analysis and Prediction of Placement Records of AKTU Affiliated Colleges

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import html

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import analysis as an
import plots as pl

# ---------------------------------------------------------------------------
# Page setup and style (blue and white theme)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AKTU Placement Analytics", page_icon="🎓", layout="wide")

st.markdown(
    """
<style>
.hero{background:linear-gradient(120deg,#17365D 0%,#1F4E9C 60%,#2FA4E7 100%);
      padding:1.3rem 1.6rem;border-radius:14px;margin-bottom:1.1rem;}
.hero-title{color:#FFFFFF;font-size:1.9rem;font-weight:800;letter-spacing:.06em;line-height:1.2;}
.hero-sub{color:#E3EEFF;font-size:1rem;margin-top:.35rem;}
.kpi{background:#FFFFFF;border:1px solid #DCE5F3;border-top:4px solid #1F4E9C;border-radius:12px;
     padding:.85rem 1rem;box-shadow:0 1px 3px rgba(23,54,93,.08);min-height:108px;margin-bottom:.6rem;}
.kpi-label{color:#5B6B84;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;}
.kpi-value{color:#17365D;font-size:1.65rem;font-weight:700;line-height:1.3;}
.kpi-note{color:#5B6B84;font-size:.8rem;}
h2,h3{color:#17365D;}
</style>
""",
    unsafe_allow_html=True,
)


def kpi(container, label: str, value: str, note: str = "") -> None:
    container.markdown(
        f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f'<div class="kpi-note">{html.escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def show_fig(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def pct(x) -> str:
    return "Not reported" if pd.isna(x) else f"{x:.2f}%"


def num(x, decimals: int = 0) -> str:
    return "Not reported" if pd.isna(x) else f"{x:,.{decimals}f}"


# ---------------------------------------------------------------------------
# Sidebar: navigation + data settings
# ---------------------------------------------------------------------------
PAGES = [
    "Home Dashboard",
    "College-wise Analysis",
    "Statistical Analysis",
    "College Comparison",
    "Placement Prediction",
    "Model Evaluation",
    "Dataset Explorer",
]

st.sidebar.markdown("## 🎓 AKTU Placement Analytics")
st.sidebar.caption("Statistical Analysis & Prediction")
page = st.sidebar.radio("Navigate", PAGES)

with st.sidebar.expander("Data settings", expanded=False):
    uploaded_file = st.file_uploader("Use your own CSV (optional)", type=["csv"])
    pct_choice = st.radio(
        "Placement % is taken from",
        ["Recomputed: placed ÷ total × 100 (recommended)", "Exactly as written in the CSV"],
        help="One record in the CSV has a percentage that does not match placed ÷ total. "
             "See Dataset Explorer → Data quality notes.",
    )
PCT_MODE = an.PCT_RECOMPUTED if pct_choice.startswith("Recomputed") else an.PCT_FROM_CSV

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_default(path: str, modified_time: float) -> pd.DataFrame:
    return an.load_raw_csv(path)


@st.cache_data(show_spinner=False)
def load_uploaded(content: bytes) -> pd.DataFrame:
    return an.load_raw_csv(content)


@st.cache_data(show_spinner=False)
def prepare(raw: pd.DataFrame, pct_mode: str):
    return an.clean_data(raw, pct_mode)


try:
    if uploaded_file is not None:
        raw = load_uploaded(uploaded_file.getvalue())
    else:
        raw = load_default(str(an.DATA_PATH), an.DATA_PATH.stat().st_mtime)
except FileNotFoundError:
    st.error("The dataset file 'placement_data_filled.csv' was not found. Keep it in the same "
             "folder as app.py, or upload a CSV from the sidebar (Data settings).")
    st.stop()
except Exception as exc:  # unreadable / empty file
    st.error(f"The dataset could not be read: {exc}")
    st.stop()

missing = an.missing_required_columns(raw)
if missing:
    st.error("The CSV is missing required column(s): " + ", ".join(missing))
    st.write("Columns found in the CSV:", list(raw.columns))
    st.stop()

clean, issues = prepare(raw, PCT_MODE)
overall_all, branches_all = an.split_levels(clean)
overall = an.usable(overall_all)       # Overall rows with a valid placement %
branch_ok = an.usable(branches_all)    # Branch rows with a valid placement %

if overall.empty:
    st.error("No usable 'Overall' placement records were found after cleaning. "
             "Check the Branch column (it must contain 'Overall' rows) and the numbers.")
    st.stop()

colleges = sorted(overall["college_name"].unique().tolist())
colors = pl.college_colors(colleges)
TARGET_YEAR = an.next_year(overall)
TARGET_LABEL = an.year_label(TARGET_YEAR)
n_pct_mismatch = int(issues["Issue"].str.startswith("Placement % in CSV").sum()) if len(issues) else 0

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><div class="hero-title">AKTU PLACEMENT ANALYTICS</div>'
    '<div class="hero-sub">Statistical Analysis and Prediction of Placement Records of '
    "AKTU Affiliated Colleges</div></div>",
    unsafe_allow_html=True,
)

# ===========================================================================
# PAGE 1 - HOME
# ===========================================================================
if page == "Home Dashboard":
    st.markdown(
        "This dashboard studies the placement records of AKTU affiliated colleges: it summarises past "
        "results, compares colleges, applies descriptive statistics, and uses **Linear Regression** to "
        f"estimate the placement percentage for **{TARGET_LABEL}**."
    )
    st.caption("All college-level numbers use only the **Overall** record of each college and year. "
               "Branch records are analysed separately so nothing is counted twice.")

    first_year, last_year = overall["graduation_year"].min(), overall["graduation_year"].max()
    top = overall.loc[overall["placement_percentage"].idxmax()]
    placed_sum = overall["placed_students"].sum()
    total_sum = overall["total_student"].sum()

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Colleges", f"{len(colleges)}", ", ".join(colleges))
    kpi(c2, "Placement records", f"{len(overall)}", f"Overall records · plus {len(branches_all)} branch-level rows")
    kpi(c3, "Average placement", f"{overall['placement_percentage'].mean():.2f}%",
        "Mean of all Overall records")
    kpi(c4, "Highest placement", f"{top['placement_percentage']:.2f}%",
        f"{top['college_name']} · {top['graduation_year']}")

    c5, c6, c7, c8 = st.columns(4)
    kpi(c5, "Total students", f"{total_sum:,.0f}", "Sum over all Overall records")
    kpi(c6, "Placed students", f"{placed_sum:,.0f}", "Sum over all Overall records")
    kpi(c7, "Weighted placement rate", f"{placed_sum / total_sum * 100:.2f}%" if total_sum else "n/a",
        "Total placed ÷ total students")
    low = overall.loc[overall["placement_percentage"].idxmin()]
    kpi(c8, "Lowest placement", f"{low['placement_percentage']:.2f}%",
        f"{low['college_name']} · {low['graduation_year']}")

    if n_pct_mismatch and PCT_MODE == an.PCT_RECOMPUTED:
        st.info(f"Data note: {n_pct_mismatch} record(s) had a placement % in the CSV that does not equal "
                "placed ÷ total. The recomputed value is used. Details: Dataset Explorer → Data quality notes.")

    st.divider()
    st.subheader("Year-wise placement trend")
    trend = overall.pivot_table(index="graduation_year", columns="college_name",
                                values="placement_percentage", aggfunc="mean").sort_index()
    trend["Average of colleges"] = trend.mean(axis=1)
    left, right = st.columns([3, 2])
    with left:
        st.line_chart(trend, height=340)
        st.caption("Placement percentage of each college by graduation year (hover for values).")
    with right:
        st.markdown("**Year-wise average across colleges**")
        yearly = trend[["Average of colleges"]].round(2).reset_index()
        yearly.columns = ["Graduation year", "Average placement (%)"]
        st.dataframe(yearly, hide_index=True)
        change = yearly["Average placement (%)"].iloc[-1] - yearly["Average placement (%)"].iloc[0]
        if len(yearly) > 1:
            word = "fell" if change < 0 else "rose"
            st.write(f"From {first_year} to {last_year} the average placement {word} by "
                     f"**{abs(change):.2f} percentage points**.")

    st.subheader("Overall placement statistics by college")
    stats = []
    for college, g in overall.groupby("college_name"):
        g = g.sort_values("year_number")
        stats.append({
            "College": college, "Records": len(g),
            "Average (%)": g["placement_percentage"].mean(),
            "Highest (%)": g["placement_percentage"].max(),
            "Lowest (%)": g["placement_percentage"].min(),
            f"Latest ({g['graduation_year'].iloc[-1]}) (%)": g["placement_percentage"].iloc[-1],
            "Change first → latest (points)": g["placement_percentage"].iloc[-1] - g["placement_percentage"].iloc[0],
        })
    stats_df = pd.DataFrame(stats)
    num_cols = stats_df.select_dtypes("number").columns.drop("Records")
    stats_df[num_cols] = stats_df[num_cols].round(2)
    st.dataframe(stats_df, hide_index=True)

    st.subheader("Average placement by college")
    show_fig(pl.avg_bar_figure(overall.groupby("college_name")["placement_percentage"].mean(),
                               colors, "Average placement percentage (all years)"))

# ===========================================================================
# PAGE 2 - COLLEGE-WISE
# ===========================================================================
elif page == "College-wise Analysis":
    st.header("College-wise Placement Analysis")
    college = st.selectbox("Select a college", colleges)
    d = overall[overall["college_name"] == college].sort_values("year_number")
    years_desc = d["graduation_year"].tolist()[::-1]
    year = st.selectbox("Graduation year", years_desc, index=0)
    rec = d[d["graduation_year"] == year].iloc[0]

    c1, c2, c3 = st.columns(3)
    kpi(c1, "Graduation year", str(rec["graduation_year"]), college)
    kpi(c2, "Total students", num(rec["total_student"]))
    kpi(c3, "Placed students", num(rec["placed_students"]))
    c4, c5, c6 = st.columns(3)
    kpi(c4, "Placement percentage", pct(rec["placement_percentage"]),
        "Placed ÷ total" if PCT_MODE == an.PCT_RECOMPUTED else "As written in the CSV")
    has_pkg = d[["average_package", "highest_package"]].notna().any().any()
    if has_pkg:
        kpi(c5, "Average package", num(rec["average_package"], 2))
        kpi(c6, "Highest package", num(rec["highest_package"], 2))
        st.caption("Package values are shown exactly as written in the dataset (the CSV does not state the "
                   "unit; placement packages in India are usually quoted in LPA). 'Not reported' means the "
                   "value is missing - it is never treated as zero.")
    else:
        kpi(c5, "Average package", "Not available", "Not reported for this college")
        kpi(c6, "Highest package", "Not available", "Not reported for this college")

    st.subheader("Placement records (all years)")
    table = d[["graduation_year", "total_student", "placed_students", "placement_percentage",
               "average_package", "highest_package", "total companies"]].copy()
    table["total_student"] = table["total_student"].astype("Int64")
    table["placed_students"] = table["placed_students"].astype("Int64")
    table["placement_percentage"] = table["placement_percentage"].round(2)
    table = table.rename(columns={
        "graduation_year": "Graduation year", "total_student": "Total students",
        "placed_students": "Placed students", "placement_percentage": "Placement (%)",
        "average_package": "Average package", "highest_package": "Highest package",
        "total companies": "Companies visited",
    })
    drop_cols = [c for c in ["Average package", "Highest package", "Companies visited"] if table[c].isna().all()]
    st.dataframe(table.drop(columns=drop_cols), hide_index=True)

    st.subheader("Placement trend")
    left, right = st.columns(2)
    with left:
        st.markdown("**Placement percentage by year**")
        st.line_chart(d.set_index("graduation_year")[["placement_percentage"]]
                      .rename(columns={"placement_percentage": "Placement (%)"}), height=300)
    with right:
        st.markdown("**Total vs placed students**")
        st.line_chart(d.set_index("graduation_year")[["total_student", "placed_students"]]
                      .rename(columns={"total_student": "Total students", "placed_students": "Placed students"}),
                      height=300)

    st.subheader(f"Branch-wise records · {year}")
    b = branches_all[(branches_all["college_name"] == college) & (branches_all["graduation_year"] == year)]
    if b.empty:
        st.info("No branch-level records are available for this college and year.")
    else:
        bt = b[["Branch", "total_student", "placed_students", "placement_percentage"]].copy()
        bt["total_student"] = bt["total_student"].astype("Int64")
        bt["placed_students"] = bt["placed_students"].astype("Int64")
        bt["placement_percentage"] = bt["placement_percentage"].round(2)
        bt.columns = ["Branch", "Total students", "Placed students", "Placement (%)"]
        lc, rc = st.columns([2, 3])
        with lc:
            st.dataframe(bt, hide_index=True)
        with rc:
            if b["placement_percentage"].notna().any():
                show_fig(pl.branch_bar_figure(b, f"{college} · {year}: placement % by branch"))
        if b["total_student"].isna().any():
            st.caption("Branches with a blank percentage have no 'total students' figure in the dataset, "
                       "so their percentage cannot be calculated.")
        cov = an.coverage_report(clean)
        row = cov[(cov["College"] == college) & (cov["Year"] == year)]
        if not row.empty and not bool(row["Branches add up to Overall"].iloc[0]):
            r = row.iloc[0]
            st.warning(
                f"The branch rows below account for {r['Branches: total']:,.0f} students and "
                f"{r['Branches: placed']:,.0f} placed students, while the Overall record says "
                f"{r['Overall: total']:,.0f} students and {r['Overall: placed']:,.0f} placed. Branch data is "
                "incomplete for this year, so use the Overall figures for college-level conclusions.")

# ===========================================================================
# PAGE 3 - STATISTICS
# ===========================================================================
elif page == "Statistical Analysis":
    st.header("Statistical Analysis")
    level = st.radio("Which records should be analysed?",
                     ["College level (Overall records)", "Branch level (individual branches)"],
                     horizontal=True)
    college_pick = st.selectbox("College", ["All colleges"] + colleges)
    base = overall if level.startswith("College") else branch_ok
    if college_pick != "All colleges":
        base = base[base["college_name"] == college_pick]
    label = "college-level" if level.startswith("College") else "branch-level"
    st.caption("Overall and branch records are never mixed. "
               + ("Branch records without a total-students figure are left out because their percentage "
                  "cannot be calculated." if label == "branch-level" else
                  "Each Overall record is one college in one year."))

    values = base["placement_percentage"]
    s = an.summary_stats(values)
    if s["count"] == 0:
        st.warning("There are no valid records for this selection.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Records (n)", f"{s['count']}")
    kpi(c2, "Mean", f"{s['mean']:.2f}%")
    kpi(c3, "Median", f"{s['median']:.2f}%")
    kpi(c4, "Mode", "No mode" if not s["modes"] else an.format_modes(s["modes"]),
        "No repeated value" if not s["modes"] else "Most frequent value")
    c5, c6, c7, c8 = st.columns(4)
    kpi(c5, "Variance", "n/a" if pd.isna(s["variance"]) else f"{s['variance']:.2f}", "Sample variance (n − 1)")
    kpi(c6, "Standard deviation", "n/a" if pd.isna(s["std"]) else f"{s['std']:.2f}", "Sample std. deviation")
    kpi(c7, "Minimum", f"{s['min']:.2f}%")
    kpi(c8, "Maximum", f"{s['max']:.2f}%")

    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["Summary table", "Frequency distribution", "Histogram", "Box plot", "College-wise table", "Explanation"])
    with t1:
        st.dataframe(an.summary_table(s), hide_index=True)
        st.caption("Variance is in 'percentage points squared'. Sample formulas (divide by n − 1) are used, "
                   "the same as pandas' default in the original project.")
    with t2:
        freq = an.frequency_table(values)
        st.dataframe(freq, hide_index=True)
        st.caption("Records are grouped into classes of width 10 percentage points. The last class includes 100%.")
    with t3:
        show_fig(pl.hist_figure(values, f"Distribution of {label} placement percentage"))
    with t4:
        show_fig(pl.box_figure(base, colors, f"Box plot of {label} placement percentage"))
        st.caption("The box covers the middle 50% of records (Q1 to Q3), the line inside is the median, "
                   "whiskers show the spread, dots are individual records.")
    with t5:
        gs = an.grouped_stats(base)
        st.dataframe(gs, hide_index=True)
        st.caption("Colleges with a single record have no variance or standard deviation (shown blank).")
    with t6:
        st.markdown(f"### What do these numbers mean? ({label} records)")
        for line in an.explain_stats(s, label):
            st.markdown("- " + line)
        if s["count"] < 10:
            st.warning(f"Only {s['count']} records are used here, so these statistics are rough descriptions "
                       "and not strong evidence about all AKTU colleges.")

# ===========================================================================
# PAGE 4 - COMPARISON
# ===========================================================================
elif page == "College Comparison":
    st.header("College Comparison")
    picked = st.multiselect("Choose two or more colleges", colleges, default=colleges)
    if len(picked) < 2:
        st.info("Select at least two colleges to see the comparison.")
    else:
        sub = overall[overall["college_name"].isin(picked)]
        rows = []
        for college, g in sub.groupby("college_name"):
            g = g.sort_values("year_number")
            rows.append({
                "College": college, "Records": len(g),
                "Average (%)": g["placement_percentage"].mean(),
                "Highest (%)": g["placement_percentage"].max(),
                "Lowest (%)": g["placement_percentage"].min(),
                "Latest year": g["graduation_year"].iloc[-1],
                "Latest (%)": g["placement_percentage"].iloc[-1],
                "Change first → latest (points)": g["placement_percentage"].iloc[-1] - g["placement_percentage"].iloc[0],
            })
        comp = pd.DataFrame(rows).sort_values("Average (%)", ascending=False)
        numc = comp.select_dtypes("number").columns.drop("Records")
        comp[numc] = comp[numc].round(2)

        best = comp.iloc[0]
        st.success(f"Highest average placement among the selected colleges: **{best['College']}** "
                   f"({best['Average (%)']:.2f}%).")
        st.subheader("Comparison table")
        st.dataframe(comp, hide_index=True)

        st.subheader("Year-wise performance")
        pivot = sub.pivot_table(index="graduation_year", columns="college_name",
                                values="placement_percentage", aggfunc="mean").sort_index()
        l, r = st.columns(2)
        with l:
            st.markdown("**Line chart**")
            st.line_chart(pivot, height=340)
        with r:
            st.markdown("**Bar chart**")
            show_fig(pl.grouped_bar_figure(overall, picked, colors))
        st.dataframe(pivot.round(2).reset_index().rename(columns={"graduation_year": "Graduation year"}),
                     hide_index=True)

        st.subheader("Average placement")
        show_fig(pl.avg_bar_figure(sub.groupby("college_name")["placement_percentage"].mean(), colors,
                                   "Average placement percentage (all years)"))

# ===========================================================================
# PAGE 5 - PREDICTION
# ===========================================================================
elif page == "Placement Prediction":
    st.header(f"Placement Prediction · {TARGET_LABEL}")
    st.warning("These predictions are **estimates** based on a straight-line trend of past placement "
               "percentages. They are **not guaranteed actual results**.")
    college = st.selectbox("Select a college", colleges)
    d = overall[overall["college_name"] == college].sort_values("year_number")
    fit = an.fit_predict(d, TARGET_YEAR)

    st.subheader("Historical placement records")
    hist = d[["graduation_year", "total_student", "placed_students", "placement_percentage"]].copy()
    hist["total_student"] = hist["total_student"].astype("Int64")
    hist["placed_students"] = hist["placed_students"].astype("Int64")
    hist["placement_percentage"] = hist["placement_percentage"].round(2)
    hist.columns = ["Graduation year", "Total students", "Placed students", "Placement (%)"]
    st.dataframe(hist, hide_index=True)

    if not fit["ok"]:
        st.error(f"A prediction is not possible for {college}: {fit['message']}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, f"Estimate {TARGET_LABEL}", f"{fit['prediction']:.2f}%", "Linear Regression")
        kpi(c2, "Latest actual", f"{d['placement_percentage'].iloc[-1]:.2f}%", d["graduation_year"].iloc[-1])
        kpi(c3, "Trend", f"{fit['slope']:+.2f} pts/year", "Slope of the fitted line")
        kpi(c4, "Records used", f"{fit['n']}", "Training data points")
        st.progress(int(round(fit["prediction"])))
        show_fig(pl.prediction_figure(fit, college, TARGET_YEAR, TARGET_LABEL))

        if fit["clipped"]:
            st.warning(f"The straight line gives {fit['raw_prediction']:.2f}%, which is outside the possible range "
                       "0-100%. As in the original project, the estimate is limited to the 0-100% range. "
                       "This means the recent trend is too steep to continue for another year.")
        if fit["n"] == 2:
            st.info("Only two records are available, so the line passes exactly through both points. "
                    "Treat the estimate with extra caution.")
        else:
            st.caption(f"Only {fit['n']} yearly records are used for this college. A prediction from so few "
                       "points is a rough indication of direction, not a precise forecast.")
        with st.expander("How is the estimate calculated?"):
            st.markdown(
                "A Linear Regression model fits a straight line  \n"
                "**placement % = intercept + slope × year**  \n"
                f"to this college's past records (year = first year of the session, e.g. 2023 for 2023-24).  \n"
                f"For {college}: intercept = {fit['intercept']:.2f}, slope = {fit['slope']:.2f}. "
                f"Putting year = {TARGET_YEAR} into the line gives {fit['raw_prediction']:.2f}%.")

    st.subheader(f"Estimates for all colleges ({TARGET_LABEL})")
    st.dataframe(an.predict_all(overall, TARGET_YEAR), hide_index=True)

# ===========================================================================
# PAGE 6 - MODEL EVALUATION
# ===========================================================================
elif page == "Model Evaluation":
    st.header("Model Evaluation")
    st.markdown(
        "**Method (same as the original project):** for every college with at least 3 yearly records, "
        "a Linear Regression is trained on all years *except the latest one*, then used to predict that "
        "latest year. The prediction is compared with the real value. Errors of all colleges are pooled "
        "to compute the metrics."
    )
    ev = an.evaluate_model(overall)
    if ev["n_test"] == 0:
        st.info("Evaluation needs at least one college with 3 or more yearly records.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "MAE", f"{ev['mae']:.3f}", "Mean Absolute Error")
        kpi(c2, "MSE", f"{ev['mse']:.3f}", "Mean Squared Error")
        kpi(c3, "RMSE", f"{ev['rmse']:.3f}", "Root Mean Squared Error")
        kpi(c4, "R² score", "Not available" if ev["r2"] is None else f"{ev['r2']:.3f}",
            "Needs 2+ test points")

        st.subheader("Actual vs predicted (test year)")
        t = ev["table"].copy()
        l, r = st.columns([3, 2])
        with l:
            show = t.copy()
            for col in ["Actual (%)", "Predicted (%)", "Error (pred - actual)", "Absolute error"]:
                show[col] = show[col].round(2)
            st.dataframe(show, hide_index=True)
        with r:
            show_fig(pl.evaluation_figure(t))

        st.warning(f"The metrics are based on only {ev['n_test']} test point(s) (one per college), so they "
                   "give a rough idea of accuracy and not a reliable measure.")
        if ev["skipped"]:
            st.caption("Not evaluated (fewer than 3 records): " + ", ".join(ev["skipped"]))

        st.subheader("What do the metrics mean?")
        st.markdown(
            f"- **MAE = {ev['mae']:.2f}** - on average the predicted placement % is about "
            f"{ev['mae']:.1f} percentage points away from the real value.\n"
            f"- **MSE = {ev['mse']:.2f}** - the average of the squared errors; large errors are punished more.\n"
            f"- **RMSE = {ev['rmse']:.2f}** - the square root of MSE, in the same unit as placement %, so it "
            "can be read directly as a 'typical error'.\n"
            "- **R² score** - 1 means perfect predictions, 0 means no better than always guessing the average, "
            "and negative values mean worse than that."
        )
        if ev["r2"] is not None:
            st.caption("Note: the test points belong to different colleges whose placement levels differ a lot, "
                       "so R² can look high even when individual errors are large. MAE and RMSE are the more "
                       "informative numbers here.")
    st.caption("Placement % is "
               + ("recomputed from placed ÷ total." if PCT_MODE == an.PCT_RECOMPUTED
                  else "taken exactly as written in the CSV.")
               + " Changing this in the sidebar (Data settings) changes the metrics.")

# ===========================================================================
# PAGE 7 - DATASET EXPLORER
# ===========================================================================
else:
    st.header("Dataset Explorer")
    show_cols = ["college_name", "Branch", "graduation_year", "total_student", "placed_students",
                 "placement_percentage", "average_package", "highest_package", "total companies"]

    search = st.text_input("Search (college, branch or year)", "")
    f1, f2, f3, f4 = st.columns(4)
    sel_colleges = f1.multiselect("College", sorted(clean["college_name"].unique()))
    sel_branches = f2.multiselect("Branch", sorted(clean["Branch"].unique()))
    sel_years = f3.multiselect("Graduation year", sorted(clean["graduation_year"].unique()))
    level = f4.selectbox("Record type", ["All records", "Overall records only", "Branch records only"])

    view = clean
    if sel_colleges:
        view = view[view["college_name"].isin(sel_colleges)]
    if sel_branches:
        view = view[view["Branch"].isin(sel_branches)]
    if sel_years:
        view = view[view["graduation_year"].isin(sel_years)]
    if level == "Overall records only":
        view = view[view["Branch"] == "Overall"]
    elif level == "Branch records only":
        view = view[view["Branch"] != "Overall"]
    if search.strip():
        text = view["college_name"] + " " + view["Branch"] + " " + view["graduation_year"]
        view = view[text.str.contains(search.strip(), case=False, regex=False)]

    st.write(f"Showing **{len(view)}** of {len(clean)} records.")
    helper = st.checkbox("Show helper columns (year number, CSV % and recomputed %)")
    cols = show_cols + (["year_number", "placement_percentage_csv", "placement_percentage_recomputed"]
                        if helper else [])
    st.dataframe(view[cols], hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button("⬇ Download cleaned dataset (CSV)",
                       data=clean.to_csv(index=False).encode("utf-8"),
                       file_name="aktu_cleaned_placement_data.csv", mime="text/csv")
    d2.download_button("⬇ Download current filtered view (CSV)",
                       data=view[cols].to_csv(index=False).encode("utf-8"),
                       file_name="aktu_filtered_placement_data.csv", mime="text/csv")

    st.subheader("Dataset information")
    tab_a, tab_b, tab_c, tab_d = st.tabs(["Overview", "Data types", "Missing values", "Data quality notes"])
    with tab_a:
        a1, a2, a3, a4 = st.columns(4)
        kpi(a1, "Rows (after cleaning)", f"{len(clean)}", f"{len(raw)} rows in the CSV")
        kpi(a2, "Columns in CSV", f"{raw.shape[1]}")
        kpi(a3, "Overall / Branch rows", f"{len(overall_all)} / {len(branches_all)}")
        kpi(a4, "Colleges · Years", f"{clean['college_name'].nunique()} · {clean['graduation_year'].nunique()}")
    with tab_b:
        info = pd.DataFrame({
            "Column": clean.columns,
            "Data type": [str(t) for t in clean.dtypes],
            "Non-missing": clean.notna().sum().values,
            "Unique values": clean.nunique().values,
        })
        st.dataframe(info, hide_index=True)
    with tab_c:
        miss = pd.DataFrame({
            "Column": raw.columns,
            "Missing values": raw.isna().sum().values,
            "Missing (%)": (raw.isna().mean() * 100).round(1).values,
        })
        st.markdown("Missing values in the **original CSV**:")
        st.dataframe(miss, hide_index=True)
        st.caption("Blank package / companies values mean 'not reported'. They are never replaced by zero.")
    with tab_d:
        if issues.empty:
            st.success("No data problems were detected.")
        else:
            st.dataframe(issues, hide_index=True)
        st.markdown("**Do the branch rows add up to the Overall row?**")
        st.dataframe(an.coverage_report(clean), hide_index=True)
        st.caption("Where they do not add up, branch data is incomplete. This is why Overall and branch records "
                   "are never combined in calculations.")

st.sidebar.divider()
st.sidebar.caption("AKTU Placement PBL Project · Mathematics")
