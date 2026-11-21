# 🎓 AKTU PLACEMENT ANALYTICS

**Statistical Analysis and Prediction of Placement Records of AKTU Affiliated Colleges**

An interactive Streamlit dashboard built for a B.Tech Mathematics PBL project. It explores the placement
records of AKTU affiliated colleges, applies descriptive statistics, compares colleges, and estimates the next
year's placement percentage with Linear Regression.

---

## Features

| Page | What it shows |
|------|---------------|
| **Home Dashboard** | KPI cards (colleges, records, average / highest / lowest placement, students placed), year-wise trend, per-college statistics |
| **College-wise Analysis** | Pick a college and year: total students, placed students, placement %, average / highest package (only when reported), trend graphs, branch-wise breakdown |
| **Statistical Analysis** | Mean, median, mode, variance, standard deviation, min / max, quartiles, frequency distribution, histogram, box plot, college-wise table, plain-language explanation |
| **College Comparison** | Select several colleges; comparison table, line chart, bar charts |
| **Placement Prediction** | Linear Regression estimate for the next session (2026-27) with graph and clear "estimate, not guarantee" warning |
| **Model Evaluation** | MAE, MSE, RMSE and R² using a hold-out-the-latest-year method |
| **Dataset Explorer** | Search / filter the data, download the cleaned CSV, data types, missing values, data-quality notes |

## Technologies

Python · Streamlit · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn

## Project structure

```
AKTU_Placement_Project/
├── app.py                      # Streamlit user interface (7 pages)
├── analysis.py                 # Data cleaning, statistics, Linear Regression, evaluation (no Streamlit code)
├── plots.py                    # Matplotlib / Seaborn charts
├── maths_pbl.py                # Original project file (kept unchanged for reference)
├── placement_data_filled.csv   # Dataset
├── requirements.txt            # Python libraries
├── README.md
├── DEPLOY_GUIDE.md             # Beginner step-by-step: GitHub + Streamlit Cloud
├── .gitignore
├── .streamlit/config.toml      # Blue / white theme
└── tests/test_analysis.py      # Automated checks of the calculations
```

## Run locally

1. Install **Python 3.10 or newer** (3.12 recommended) from python.org.
2. Open a terminal / Command Prompt in the project folder and run:

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

3. The app opens at `http://localhost:8501`.

Run the automated checks (optional): `python tests/test_analysis.py`

## Dataset (`placement_data_filled.csv`)

77 rows × 9 columns: 3 colleges (IET lucknow, BIET jhansi, KNIT sultanpur) and 3 sessions (2023-24, 2024-25,
2025-26).

| Column | Meaning |
|--------|---------|
| `college_name` | College |
| `Branch` | Branch name, or `Overall` for the whole college |
| `graduation_year` | Session, e.g. `2024-25` |
| `total_student` | Number of students |
| `placed_students` | Number of students placed |
| `placement_percentage` | Placement percentage |
| `average_package`, `highest_package` | Package values, only for `Overall` rows and only when reported (unit not stated in the file) |
| `total companies` | Companies visited, as text such as `50+` |

There are 9 `Overall` rows (3 colleges × 3 years) and 68 branch rows. Package and company information exists only
on `Overall` rows; blanks mean "not reported" and are **never treated as zero**.

### Data cleaning (same logic as the original project, plus documented fixes)

* College-level analysis uses **only `Overall` rows**. Branch rows are analysed separately and are never mixed with
  `Overall` rows, because that would count students twice.
* The year number is the first four digits of `graduation_year` (`2023-24` → `2023`).
* Text is trimmed; branch spelling is made consistent (`Mis` / `MIS` → `MIS`).
* Numbers are converted safely; unreadable values become missing, not zero; duplicates are removed.
* **Inconsistency found and fixed:** for *IET lucknow 2024-25 (Overall)* the CSV gives 67.71 %, but
  518 placed ÷ 845 students = **61.30 %** (67.71 % equals 518 ÷ 765, the previous year's student count). The app
  recomputes the percentage as `placed ÷ total × 100` by default. A switch in the sidebar (*Data settings*) lets
  you use the CSV value exactly as written instead. The Dataset Explorer lists every such note.
* KNIT sultanpur's `MIS` rows have no `total_student`, so their percentage cannot be calculated and is left blank.
* In several college-years the branch rows do not add up to the `Overall` row (branch data is incomplete); the
  Dataset Explorer shows this check.

## Statistical methods

Mean, median, mode (values rounded to 2 decimals; "no mode" when no value repeats), **sample** variance and
standard deviation (divide by *n − 1*, as pandas does), minimum, maximum, range, quartiles (Q1, Q3), IQR,
coefficient of variation, a grouped frequency distribution (class width 10), histogram and box plot. Every result is
accompanied by a plain-language explanation.

## Machine-learning methodology

* **Model:** one `LinearRegression` per college, `placement % ~ year` (year = first year of the session), trained
  on that college's `Overall` records.
* **Target:** the session after the latest one in the data (2026-27). Predictions are limited to 0–100 %.
* A college needs at least 2 yearly records for a prediction; otherwise the app shows a message instead of failing.

### Evaluation

For each college with at least 3 yearly records, the model is trained on all years except the latest, then
predicts the latest year. The errors of all colleges are pooled and summarised with **MAE, MSE, RMSE and R²**.

### Limitations

* With 3 colleges × 3 years the evaluation has only **3 test points** and each model is fitted on 2–3 points, so
  the metrics and predictions are rough indicators, not reliable forecasts.
* A straight line cannot capture sudden changes; a steeply falling trend can predict a value close to 0 %.
* R² is computed over test points from different colleges whose placement levels differ a lot, so it can look high
  even when individual errors are large. MAE and RMSE are more informative here.

**Predictions are estimates, not guaranteed actual results.**

## Upload to GitHub and deploy on Streamlit Community Cloud

See **[DEPLOY_GUIDE.md](DEPLOY_GUIDE.md)** for the full click-by-click guide. In short:

1. Create a free GitHub account and a **public** repository.
2. Upload every file and folder from this project (`app.py` must be at the top level of the repository).
3. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**, choose *"Yup, I have an app"*,
   select your repository, branch `main`, main file path `app.py`, open **Advanced settings** and pick a recent
   Python version (3.12 or newer), then click **Deploy**.

No API keys, secrets or paid services are required. The CSV is read with a path relative to `app.py`.
