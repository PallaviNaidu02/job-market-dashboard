import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="AI Job Market Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Generate Synthetic Data ----------------
@st.cache_data
def generate_data(n=12000):
    np.random.seed(42)

    roles = [
        "Data Scientist", "Machine Learning Engineer", "AI Researcher",
        "Software Engineer", "Backend Engineer", "Frontend Engineer",
        "DevOps Engineer", "Cloud Engineer", "Data Analyst",
        "Business Analyst", "Product Manager", "Cybersecurity Analyst",
        "QA Engineer", "IT Consultant"
    ]

    countries = ["Germany", "USA", "UK", "Canada", "Netherlands"]
    work_modes = ["Remote", "Hybrid", "On-site"]
    education = ["Bachelor", "Master", "PhD"]
    company_size = ["Startup", "Mid-size", "Enterprise"]

    df = pd.DataFrame({
        "Role": np.random.choice(roles, n),
        "Country": np.random.choice(countries, n),
        "Experience": np.random.randint(0, 20, n),
        "Skill_Score": np.random.randint(50, 100, n),
        "Work_Mode": np.random.choice(work_modes, n),
        "Education": np.random.choice(education, n),
        "Company_Size": np.random.choice(company_size, n)
    })

    role_base_salary = {
        "Data Scientist": 70, "Machine Learning Engineer": 75, "AI Researcher": 85,
        "Software Engineer": 65, "Backend Engineer": 68, "Frontend Engineer": 60,
        "DevOps Engineer": 72, "Cloud Engineer": 74, "Data Analyst": 55,
        "Business Analyst": 58, "Product Manager": 80,
        "Cybersecurity Analyst": 73, "QA Engineer": 50, "IT Consultant": 62
    }

    country_bonus = {
        "USA": 15, "Germany": 10, "UK": 8,
        "Canada": 6, "Netherlands": 7
    }

    work_bonus = {"Remote": 5, "Hybrid": 3, "On-site": 0}
    edu_bonus = {"Bachelor": 0, "Master": 5, "PhD": 10}
    company_bonus = {"Startup": 0, "Mid-size": 4, "Enterprise": 8}

    df["Salary"] = (
        df["Role"].map(role_base_salary)
        + df["Country"].map(country_bonus)
        + df["Work_Mode"].map(work_bonus)
        + df["Education"].map(edu_bonus)
        + df["Company_Size"].map(company_bonus)
        + df["Experience"] * 2
        + df["Skill_Score"] * 0.3
        + np.random.normal(0, 5, n)
    )

    return df

df = generate_data()

# ---------------- Sidebar Filters (ALL SELECTBOXES) ----------------
st.sidebar.title("🔍 Job Search Filters")

selected_role = st.sidebar.selectbox(
    "Select Job Role",
    sorted(df["Role"].unique())
)

selected_country = st.sidebar.selectbox(
    "Select Location",
    sorted(df["Country"].unique())
)

selected_work = st.sidebar.selectbox(
    "Select Work Mode",
    sorted(df["Work_Mode"].unique())
)

filtered_df = df[
    (df["Role"] == selected_role) &
    (df["Country"] == selected_country) &
    (df["Work_Mode"] == selected_work)
]

# ---------------- Title ----------------
st.title(f"💼 {selected_role} Jobs in {selected_country}")
st.markdown(f"### Work Mode: {selected_work} | AI-driven salary insights")

# ---------------- KPIs ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Available Jobs", len(filtered_df))
c2.metric("Average Salary (€K)", round(filtered_df["Salary"].mean(), 2))
c3.metric("Maximum Salary (€K)", round(filtered_df["Salary"].max(), 2))
c4.metric("Avg Experience (Years)", round(filtered_df["Experience"].mean(), 1))

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "📈 Trends", "🤖 AI Salary Predictor"])

# -------- TAB 1: Market Overview --------
with tab1:
    col1, col2 = st.columns(2)

    fig_company = px.box(
        filtered_df,
        x="Company_Size",
        y="Salary",
        title="Salary Distribution by Company Size",
        labels={"Salary": "Salary (€K)"}
    )
    col1.plotly_chart(fig_company, use_container_width=True)

    fig_edu = px.bar(
        filtered_df.groupby("Education")["Salary"].mean().reset_index(),
        x="Education",
        y="Salary",
        title="Average Salary by Education Level",
        labels={"Salary": "Salary (€K)"}
    )
    col2.plotly_chart(fig_edu, use_container_width=True)

# -------- TAB 2: Trends --------
with tab2:
    fig_exp_salary = px.scatter(
        filtered_df,
        x="Experience",
        y="Salary",
        color="Company_Size",
        title="Experience vs Salary Trend",
        labels={"Salary": "Salary (€K)"}
    )
    st.plotly_chart(fig_exp_salary, use_container_width=True)

# -------- TAB 3: AI Salary Predictor --------
with tab3:
    X = df[["Experience", "Skill_Score"]]
    y = df["Salary"]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    col1, col2 = st.columns(2)
    exp = col1.slider("Years of Experience", 0, 20, 5)
    skill = col2.slider("Skill Score", 50, 100, 75)

    if st.button("Predict Salary"):
        prediction = model.predict([[exp, skill]])[0]
        st.success(f"💰 Estimated Salary: €{prediction:.2f}K")

# ---------------- Data Preview ----------------
st.subheader("📄 Simulated Job Listings")
st.dataframe(filtered_df.head(20))
