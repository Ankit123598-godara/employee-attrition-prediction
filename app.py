from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Palo_Alto_Networks_Attrition.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
SCORES_PATH = OUTPUT_DIR / "employee_risk_scores.csv"
METRICS_PATH = OUTPUT_DIR / "model_comparison_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "project_summary.json"

st.set_page_config(page_title="Employee Attrition Risk Dashboard", page_icon="📊", layout="wide")
st.title("📊 Employee Attrition Prediction & Risk Dashboard")
st.caption("Academic demonstration using the provided employee attrition dataset. Scores are model estimates, not proof of an employee's intentions.")

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

if not SCORES_PATH.exists():
    st.error("Risk-score output was not found. First run all cells in Employee_Attrition_Prediction.ipynb, including the final save-outputs cell.")
    st.code("employee_risk_scores.csv")
    st.stop()

scores = load_csv(SCORES_PATH)
metrics = load_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()
summary = {}
if SUMMARY_PATH.exists():
    try:
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        summary = {}

# Normalize common column names for compatibility with notebook output versions.
if "RiskPercent" not in scores.columns and "Risk_Percentage" in scores.columns:
    scores["RiskPercent"] = scores["Risk_Percentage"]
if "RiskLevel" not in scores.columns and "Risk_Category" in scores.columns:
    scores["RiskLevel"] = scores["Risk_Category"]
if "AttritionProbability" not in scores.columns and "Attrition_Probability" in scores.columns:
    scores["AttritionProbability"] = scores["Attrition_Probability"]
if "RiskPercent" not in scores.columns and "AttritionProbability" in scores.columns:
    scores["RiskPercent"] = scores["AttritionProbability"] * 100
if "RiskLevel" not in scores.columns and "RiskPercent" in scores.columns:
    scores["RiskLevel"] = pd.cut(scores["RiskPercent"], bins=[-float("inf"), 30, 60, float("inf")], labels=["Low", "Medium", "High"], right=True).astype(str)

# Apply interactive risk thresholds to the probability score where available.
st.sidebar.header("Dashboard filters")
threshold_high = st.sidebar.slider("High-risk threshold (%)", min_value=30, max_value=90, value=60, step=5)
threshold_medium = st.sidebar.slider("Low-risk upper threshold (%)", min_value=10, max_value=50, value=30, step=5)
if threshold_medium >= threshold_high:
    st.sidebar.warning("Low-risk threshold should be lower than the high-risk threshold.")
    threshold_medium = max(10, threshold_high - 10)

if "RiskPercent" in scores.columns:
    scores = scores.copy()
    scores["DashboardRiskLevel"] = scores["RiskPercent"].apply(
        lambda p: "Low" if p < threshold_medium else ("Medium" if p < threshold_high else "High")
    )
else:
    scores["DashboardRiskLevel"] = scores.get("RiskLevel", "Unknown")

for col in ["Department", "JobRole", "OverTime"]:
    if col not in scores.columns:
        scores[col] = "Unknown"

all_departments = sorted(scores["Department"].dropna().astype(str).unique().tolist())
all_roles = sorted(scores["JobRole"].dropna().astype(str).unique().tolist())
departments = st.sidebar.multiselect("Department", all_departments, default=all_departments)
roles = st.sidebar.multiselect("Job role", all_roles, default=all_roles)
filtered = scores[
    scores["Department"].astype(str).isin(departments)
    & scores["JobRole"].astype(str).isin(roles)
].copy()

# KPI cards
n = len(filtered)
high_count = int((filtered["DashboardRiskLevel"] == "High").sum()) if n else 0
medium_count = int((filtered["DashboardRiskLevel"] == "Medium").sum()) if n else 0
low_count = int((filtered["DashboardRiskLevel"] == "Low").sum()) if n else 0
avg_risk = float(filtered["RiskPercent"].mean()) if n and "RiskPercent" in filtered.columns else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Employees in view", f"{n:,}")
c2.metric("High risk", f"{high_count:,}")
c3.metric("Medium risk", f"{medium_count:,}")
c4.metric("Low risk", f"{low_count:,}")
c5.metric("Average predicted risk", f"{avg_risk:.1f}%")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Risk overview", "Employee profile", "Department & role", "Model evaluation"])

with tab1:
    st.subheader("Risk category distribution")
    if n:
        risk_order = ["Low", "Medium", "High"]
        counts = filtered["DashboardRiskLevel"].value_counts().reindex(risk_order, fill_value=0).rename_axis("Risk category").reset_index(name="Employees")
        fig = px.bar(counts, x="Risk category", y="Employees", color="Risk category", category_orders={"Risk category": risk_order}, title="Employees by predicted risk category")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Employees requiring review")
        high_risk = filtered[filtered["DashboardRiskLevel"] == "High"].sort_values("RiskPercent", ascending=False)
        display_cols = [c for c in ["Employee_ID", "Department", "JobRole", "RiskPercent", "DashboardRiskLevel", "ReviewReasonCodes"] if c in high_risk.columns]
        if display_cols:
            st.dataframe(high_risk[display_cols].head(100), use_container_width=True, hide_index=True)
        else:
            st.info("No employee detail columns were found in the risk-score file.")
    else:
        st.info("No records match the selected filters.")

with tab2:
    st.subheader("Individual employee risk profile")
    if n:
        id_col = "Employee_ID" if "Employee_ID" in filtered.columns else None
        if id_col:
            ids = filtered[id_col].astype(str).tolist()
            selected_id = st.selectbox("Select an employee record", ids)
            row = filtered[filtered[id_col].astype(str) == selected_id].iloc[0]
        else:
            selected_index = st.selectbox("Select a record", filtered.index.tolist())
            row = filtered.loc[selected_index]

        a, b, c = st.columns(3)
        a.metric("Predicted attrition risk", f"{float(row.get('RiskPercent', 0)):.1f}%")
        b.metric("Risk category", str(row.get("DashboardRiskLevel", "Unknown")))
        actual = row.get("Attrition", row.get("Actual_Attrition", None))
        c.metric("Historical target", "Left" if str(actual) == "1" else ("Stayed" if str(actual) == "0" else "Not available"))

        profile_fields = ["Department", "JobRole", "OverTime", "Age", "MonthlyIncome", "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "YearsAtCompany", "YearsSinceLastPromotion", "YearsInCurrentRole", "DistanceFromHome", "NumCompaniesWorked"]
        profile = {field: row[field] for field in profile_fields if field in row.index}
        st.write("**Available employee attributes**")
        st.json({k: (v.item() if hasattr(v, "item") else v) for k, v in profile.items()})
        st.write("**Simple review indicators**")
        reason = row.get("ReviewReasonCodes", "No reason-code field available.")
        st.write(str(reason) if pd.notna(reason) else "No reason codes recorded.")
        st.warning("Review these indicators with context. Do not use a prediction as the sole basis for decisions about an employee.")
    else:
        st.info("No records match the selected filters.")

with tab3:
    st.subheader("Average predicted risk by department")
    if n and "RiskPercent" in filtered.columns:
        dept = filtered.groupby("Department", dropna=False).agg(
            Employees=("RiskPercent", "size"),
            AverageRiskPercent=("RiskPercent", "mean"),
            HighRiskEmployees=("DashboardRiskLevel", lambda x: (x == "High").sum())
        ).reset_index().sort_values("AverageRiskPercent", ascending=False)
        st.dataframe(dept.round(2), use_container_width=True, hide_index=True)
        fig = px.bar(dept, x="Department", y="AverageRiskPercent", title="Average predicted risk by department", hover_data=["Employees", "HighRiskEmployees"])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Average predicted risk by job role")
        role = filtered.groupby("JobRole", dropna=False).agg(
            Employees=("RiskPercent", "size"),
            AverageRiskPercent=("RiskPercent", "mean"),
            HighRiskEmployees=("DashboardRiskLevel", lambda x: (x == "High").sum())
        ).reset_index().sort_values("AverageRiskPercent", ascending=False)
        st.dataframe(role.round(2), use_container_width=True, hide_index=True)
    else:
        st.info("Risk percentages are unavailable for the current data.")

with tab4:
    st.subheader("Model evaluation")
    if not metrics.empty:
        st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)
        metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "F1-score", "ROC-AUC"] if c in metrics.columns]
        if "Model" in metrics.columns and metric_cols:
            long_metrics = metrics.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score")
            fig = px.bar(long_metrics, x="Model", y="Score", color="Metric", barmode="group", title="Model comparison")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Model comparison file not found. Run the notebook's model training and save-output cells first.")
    if summary:
        st.write("**Project summary**")
        st.json(summary)

st.divider()
st.download_button(
    "Download filtered employee risk scores (CSV)",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_employee_risk_scores.csv",
    mime="text/csv"
)
st.caption("Risk scores are predictions generated from historical data. They may be inaccurate or biased and should only support careful, fair, human-reviewed retention planning.")
