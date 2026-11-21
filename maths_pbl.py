import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AKTU Placement Analysis",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>
.main {background-color: #f7f9fc;}
h1, h2, h3 {color: #17365d;}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e6eaf0;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

st.sidebar.title("🎓 AKTU Placement Portal")
st.sidebar.caption("Statistical Analysis & Prediction")

uploaded_file = st.sidebar.file_uploader(
    "Upload placement CSV (optional)",
    type=["csv"]
)

try:
    if uploaded_file is not None:
        raw_df = load_data(uploaded_file)
    else:
        raw_df = load_data("placement_data_filled.csv")
except FileNotFoundError:
    st.error(
        "Dataset file नहीं मिली। placement_data_filled.csv को app.py के "
        "साथ उसी folder में रखें या sidebar से CSV upload करें।"
    )
    st.stop()
except Exception as e:
    st.error(f"Dataset पढ़ने में समस्या हुई: {e}")
    st.stop()

required_columns = [
    "Branch", "college_name", "graduation_year",
    "placement_percentage", "total_student", "placed_students"
]
missing_columns = [col for col in required_columns if col not in raw_df.columns]

if missing_columns:
    st.error("CSV में ये जरूरी columns नहीं मिले: " + ", ".join(missing_columns))
    st.write("CSV में उपलब्ध columns:", list(raw_df.columns))
    st.stop()

# ---------------- CLEANING / PREPROCESSING ----------------
data = raw_df.copy()
data["Branch"] = data["Branch"].astype(str).str.strip()
data = data[data["Branch"].str.lower() == "overall"].copy()

data["college_name"] = data["college_name"].astype(str).str.strip()
data["graduation_year"] = data["graduation_year"].astype(str).str.strip()

for col in ["placement_percentage", "total_student", "placed_students"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Extract the first four-digit year, e.g. 2023 from "2023-24"
data["year_number"] = data["graduation_year"].str.extract(r"(\d{4})")[0]
data["year_number"] = pd.to_numeric(data["year_number"], errors="coerce")

data = data.dropna(
    subset=["college_name", "year_number", "placement_percentage"]
)
data = data[
    data["placement_percentage"].between(0, 100)
].copy()
data["year_number"] = data["year_number"].astype(int)

if data.empty:
    st.error(
        "Cleaning के बाद कोई Overall placement record नहीं बचा। "
        "CSV में Branch column के अंदर 'Overall' records और सही values जाँचें।"
    )
    st.stop()

colleges = sorted(data["college_name"].dropna().unique().tolist())

# ---------------- HEADER ----------------
st.title("🎓 AKTU Affiliated Colleges")
st.subheader("Statistical Analysis & Prediction of Placement Records")
st.write(
    "Explore historical placement performance, compare colleges, "
    "and estimate placement trends using Linear Regression."
)

# ---------------- NAVIGATION ----------------
page = st.sidebar.radio(
    "Navigate",
    [
        "Home Dashboard",
        "College Analysis",
        "Statistical Analysis",
        "College Comparison",
        "Placement Prediction",
        "Model Evaluation",
        "Dataset"
    ]
)

# ---------------- HOME DASHBOARD ----------------
if page == "Home Dashboard":
    st.header("Overall Placement Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Colleges", f"{data['college_name'].nunique():,}")
    c2.metric("Total Records", f"{len(data):,}")
    c3.metric("Average Placement", f"{data['placement_percentage'].mean():.2f}%")
    c4.metric("Highest Placement", f"{data['placement_percentage'].max():.2f}%")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Average Placement by College")
        college_avg = (
            data.groupby("college_name")["placement_percentage"]
            .mean().sort_values(ascending=False).head(15)
        )
        st.bar_chart(college_avg)

    with right:
        st.subheader("Year-wise Average Placement")
        year_avg = (
            data.groupby("graduation_year")["placement_percentage"]
            .mean().sort_index()
        )
        st.line_chart(year_avg)

    st.caption(
        "Note: Dashboard values are calculated from rows where Branch is 'Overall'."
    )

# ---------------- COLLEGE ANALYSIS ----------------
elif page == "College Analysis":
    st.header("College-wise Placement Analysis")
    selected_college = st.selectbox("Select an AKTU College", colleges)

    college_data = data[
        data["college_name"] == selected_college
    ].sort_values("year_number")

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Placement", f"{college_data['placement_percentage'].mean():.2f}%")
    c2.metric("Highest Placement", f"{college_data['placement_percentage'].max():.2f}%")
    placed_total = college_data["placed_students"].sum(min_count=1)
    c3.metric(
        "Total Placed Students",
        "N/A" if pd.isna(placed_total) else f"{int(placed_total):,}"
    )

    st.subheader("Previous Years Placement Records")
    display_cols = [
        "graduation_year", "total_student",
        "placed_students", "placement_percentage"
    ]
    st.dataframe(
        college_data[display_cols].reset_index(drop=True),
        use_container_width=True
    )

    st.subheader("Placement Trend")
    chart_data = college_data.set_index("graduation_year")[
        ["placement_percentage"]
    ]
    st.line_chart(chart_data)

# ---------------- STATISTICAL ANALYSIS ----------------
elif page == "Statistical Analysis":
    st.header("Statistical Analysis")
    st.write("Statistics are calculated using the cleaned Overall placement records.")

    values = data["placement_percentage"]
    mode_values = values.mode()

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean", f"{values.mean():.2f}%")
    c2.metric("Median", f"{values.median():.2f}%")
    c3.metric(
        "Mode",
        "No mode" if mode_values.empty
        else ", ".join(f"{x:.2f}%" for x in mode_values.tolist()[:3])
    )

    c4, c5, c6 = st.columns(3)
    c4.metric("Standard Deviation", f"{values.std():.2f}")
    c5.metric("Variance", f"{values.var():.2f}")
    c6.metric("Minimum – Maximum", f"{values.min():.2f}% – {values.max():.2f}%")

    st.subheader("College-wise Statistical Summary")
    college_stats = data.groupby("college_name")["placement_percentage"].agg(
        ["mean", "median", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
         "std", "var", "min", "max"]
    )
    college_stats.columns = [
        "Mean", "Median", "Mode", "Standard Deviation",
        "Variance", "Minimum", "Maximum"
    ]
    st.dataframe(college_stats.round(2), use_container_width=True)

    st.subheader("Placement Percentage Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(values, bins=15, kde=True, ax=ax)
    ax.set_xlabel("Placement Percentage")
    ax.set_ylabel("Number of Records")
    st.pyplot(fig)
    plt.close(fig)

# ---------------- COLLEGE COMPARISON ----------------
elif page == "College Comparison":
    st.header("Compare Colleges")
    selected_colleges = st.multiselect(
        "Choose two or more colleges",
        colleges,
        default=colleges[:min(3, len(colleges))]
    )

    if len(selected_colleges) < 2:
        st.info("Comparison देखने के लिए कम-से-कम दो colleges select करें।")
    else:
        comparison = data[
            data["college_name"].isin(selected_colleges)
        ].pivot_table(
            index="graduation_year",
            columns="college_name",
            values="placement_percentage",
            aggfunc="mean"
        ).sort_index()

        st.subheader("Year-wise Placement Comparison")
        st.line_chart(comparison)

        st.subheader("Average Placement Comparison")
        avg_comparison = (
            data[data["college_name"].isin(selected_colleges)]
            .groupby("college_name")["placement_percentage"]
            .mean().sort_values(ascending=False)
        )
        st.bar_chart(avg_comparison)
        st.dataframe(avg_comparison.round(2).rename("Average Placement (%)"))

# ---------------- PREDICTION ----------------
elif page == "Placement Prediction":
    st.header("Placement Prediction – 2026-27")
    st.warning(
        "Prediction historical placement percentages पर आधारित एक estimate है; "
        "यह actual placement result या guarantee नहीं है।"
    )

    selected_college = st.selectbox(
        "Select college for prediction", colleges
    )
    college_data = data[
        data["college_name"] == selected_college
    ].sort_values("year_number")

    if len(college_data) < 2:
        st.error("इस college के लिए prediction हेतु कम-से-कम 2 yearly records चाहिए।")
    else:
        X = college_data[["year_number"]]
        y = college_data["placement_percentage"]

        model = LinearRegression()
        model.fit(X, y)

        prediction = float(model.predict(pd.DataFrame({"year_number": [2026]}))[0])
        prediction = float(np.clip(prediction, 0, 100))

        st.metric("Estimated Placement for 2026-27", f"{prediction:.2f}%")
        st.progress(int(round(prediction)))

        trend = college_data[["year_number", "placement_percentage"]].copy()
        trend = trend.set_index("year_number")
        trend.loc[2026] = prediction
        trend = trend.sort_index()
        trend.columns = ["Placement Percentage"]
        st.subheader("Historical Data + Prediction")
        st.line_chart(trend)

        st.write("Records used for training:")
        st.dataframe(
            college_data[
                ["graduation_year", "placement_percentage"]
            ].reset_index(drop=True),
            use_container_width=True
        )

# ---------------- MODEL EVALUATION ----------------
elif page == "Model Evaluation":
    st.header("Linear Regression Model Evaluation")
    actual = []
    predicted = []

    for college in colleges:
        college_data = data[
            data["college_name"] == college
        ].sort_values("year_number")

        # At least 3 records are needed to keep training and test data separate.
        if len(college_data) < 3:
            continue

        train = college_data.iloc[:-1]
        test = college_data.iloc[-1:]

        model = LinearRegression()
        model.fit(train[["year_number"]], train["placement_percentage"])
        pred = model.predict(test[["year_number"]])

        actual.extend(test["placement_percentage"].tolist())
        predicted.extend(pred.tolist())

    if len(actual) == 0:
        st.info(
            "Evaluation के लिए किसी college में कम-से-कम 3 yearly records चाहिए।"
        )
    else:
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.3f}")
        c2.metric("RMSE", f"{rmse:.3f}")
        if len(actual) >= 2:
            r2 = r2_score(actual, predicted)
            c3.metric("R² Score", f"{r2:.3f}")
        else:
            c3.metric("R² Score", "Not available")

        st.caption(
            "Evaluation method: for each college, the last year is used as test data "
            "and earlier years are used for training. Colleges with fewer than 3 "
            "records are excluded."
        )

# ---------------- DATASET ----------------
elif page == "Dataset":
    st.header("Cleaned Placement Dataset")
    st.write(f"Rows: {len(data)} | Columns: {len(data.columns)}")
    st.dataframe(data, use_container_width=True)
    st.download_button(
        "Download cleaned data as CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="aktu_cleaned_placement_data.csv",
        mime="text/csv"
    )

st.sidebar.divider()
st.sidebar.caption("AKTU Placement PBL Project")
