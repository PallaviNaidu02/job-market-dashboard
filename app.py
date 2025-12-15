import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="AI Job Market Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.metric-box {
    background-color: #0e1117;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Generate Synthetic Data ------------------
@st.cache_data
def generate_data(n=12000):
    np.random.seed(42)

    roles = [
        "Data Scientist", "ML Engineer", "AI Researcher",
        "Software Engineer", "Backend Engineer", "Frontend Engineer",
        "DevOps Engineer", "Cloud Engineer", "Data Analyst",
        "Business Analyst", "Product Manager", "Cybersecurity Analyst",
        "QA Engineer", "IT Consultant"
    ]

    countries = ["Germany", "USA", "UK", "Canada", "Netherlands"]
    work_type = ["Remote", "Hybrid", "On-site"]
    education = ["Bachelor", "Master", "PhD"]
    company_size = ["Startup", "Mid-size", "Enterprise"]

    df = pd.DataFrame({
        "Role": np.random.choice(roles, n),
        "Country": np.random.choice(countries, n),
        "Experience": np.random.randint(0, 20, n),
        "Skill_Score": np.random.randint(50, 100, n),
        "Work_Mode": np.random.choice(work_type, n),
        "Education": np.random.choice(education, n),
        "Company_Size": np.random.choice(company_size, n)
    })

    role_base = {
        "Data Scientist": 70, "ML Engineer": 75, "AI Researcher": 85,
        "Software Engineer": 65, "Backend Engineer": 68,
        "Frontend Engineer": 60, "DevOps Engineer": 72,
        "Cloud Engineer": 74, "Data Analyst": 55,
        "Business Analyst": 58, "Product Manager": 80,
        "Cybersecurity Analyst": 73, "QA Engineer": 50,
        "IT Consultant": 62
    }

    country_bonus = {
        "USA": 15, "Germany": 10, "UK": 8,
        "Canada": 6, "Netherlands": 7
    }

    work_bonus = {"Remote": 5, "Hybrid": 3, "On-site": 0}
    edu_bonus = {"Bachelor": 0, "Master": 5, "PhD": 10}
    company_bonus = {"Startup": 0, "Mid-size": 4, "Enterprise": 8}

    df["Salary"] = (
        df["Role"].map(role_base)
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

# ------------------ Sidebar ------------------
st.sidebar.title("🎛️ Filters")

roles = st.sidebar.multiselect("Job Role", df["Role"].unique(), df["Role"].unique())
countries = st.sidebar.multiselect("Country", df["Country"].unique(), df["Country"].unique())
work = st.sidebar.multiselect("Work Mode", df["Work_Mode"].unique(), df["Work_Mode"].unique())

filtered = df[
    df["Role"].isin(roles) &
    df["Country"].isin(countries) &
    df["Work_Mode"].isin(work)
]

# ------------------ Title ------------------
st.title("💼 AI-Driven Job Market & Salary Trend Analysis")
st.markdown("### Advanced interactive dashboard using AI & simulated big data")

# ------------------ KPIs ------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Jobs", len(filtered))
c2.metric("Avg Salary (€K)", round(filtered["Salary"].mean(), 2))
c3.metric("Max Salary (€K)", round(filtered["Salary"].max(), 2))
c4.metric("Avg Experience", round(filtered["Experience"].mean(), 1))

# ------------------ Tabs ------------------
tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "📈 Trends", "🤖 AI Salary Predictor"])

# -------- TAB 1 --------
with tab1:
    col1, col2 = st.columns(2)

    fig_role = px.bar(
        filtered.groupby("Role")["Salary"].mean().sort_values(),
        title="Average Salary by Job Role",
        labels={"value": "Salary (€K)", "index": "Role"}
    )
    col1.plotly_chart(fig_role, use_container_width=True)

    fig_country = px.box(
        filtered,
        x="Country",
        y="Salary",
        title="Salary Distribution by Country"
    )
    col2.plotly_chart(fig_country, use_container_width=True)

# -------- TAB 2 --------
with tab2:
    fig_exp = px.scatter(
        filtered,
        x="Experience",
        y="Salary",
        color="Role",
        title="Experience vs Salary"
    )
    st.plotly_chart(fig_exp, use_container_width=True)

# -------- TAB 3 --------
with tab3:
    X = df[["Experience", "Skill_Score"]]
    y = df["Salary"]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    col1, col2 = st.columns(2)
    exp = col1.slider("Experience (Years)", 0, 20, 5)
    skill = col2.slider("Skill Score", 50, 100, 75)

    if st.button("Predict Salary"):
        pred = model.predict([[exp, skill]])[0]
        st.success(f"💰 Estimated Salary: €{pred:.2f}K")

# ------------------ Data Preview ------------------
st.subheader("📄 Simulated Dataset Preview")
st.dataframe(filtered.head(20))
