import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="AI Job Market Dashboard",
    layout="wide"
)

# -----------------------------
# Generate Synthetic Job Market Data
# -----------------------------
@st.cache_data
def generate_data(n=1000):
    np.random.seed(42)

    roles = ["Data Scientist", "Software Engineer", "Business Analyst", "ML Engineer"]
    locations = ["Berlin", "Munich", "Hamburg", "Frankfurt"]

    data = {
        "Role": np.random.choice(roles, n),
        "Location": np.random.choice(locations, n),
        "Experience": np.random.randint(0, 15, n),
        "Skill_Score": np.random.randint(50, 100, n)
    }

    df = pd.DataFrame(data)

    base_salary = {
        "Data Scientist": 65,
        "Software Engineer": 60,
        "Business Analyst": 55,
        "ML Engineer": 70
    }

    df["Salary"] = (
        df["Role"].map(base_salary)
        + df["Experience"] * 2
        + df["Skill_Score"] * 0.3
        + np.random.normal(0, 5, n)
    )

    return df

df = generate_data()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.title("🔍 Filters")

role_filter = st.sidebar.multiselect(
    "Select Job Role",
    df["Role"].unique(),
    default=df["Role"].unique()
)

location_filter = st.sidebar.multiselect(
    "Select Location",
    df["Location"].unique(),
    default=df["Location"].unique()
)

filtered_df = df[
    (df["Role"].isin(role_filter)) &
    (df["Location"].isin(location_filter))
]

# -----------------------------
# Dashboard Title
# -----------------------------
st.title("💼 AI-Driven Job Market & Salary Trend Analysis Dashboard")
st.markdown("Simulated job market data for salary analysis and prediction")

# -----------------------------
# KPIs
# -----------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Total Jobs", len(filtered_df))
c2.metric("Average Salary (€K)", round(filtered_df["Salary"].mean(), 2))
c3.metric("Max Salary (€K)", round(filtered_df["Salary"].max(), 2))

# -----------------------------
# Salary Distribution
# -----------------------------
st.subheader("📊 Salary Distribution")

fig, ax = plt.subplots()
ax.hist(filtered_df["Salary"], bins=30)
ax.set_xlabel("Salary (€K)")
ax.set_ylabel("Number of Jobs")
st.pyplot(fig)

# -----------------------------
# Average Salary by Role
# -----------------------------
st.subheader("🏷️ Average Salary by Job Role")

role_salary = filtered_df.groupby("Role")["Salary"].mean()

fig2, ax2 = plt.subplots()
role_salary.plot(kind="bar", ax=ax2)
ax2.set_ylabel("Average Salary (€K)")
st.pyplot(fig2)

# -----------------------------
# ML Model – Salary Prediction
# -----------------------------
st.subheader("🤖 Salary Prediction (AI Model)")

X = df[["Experience", "Skill_Score"]]
y = df["Salary"]

model = LinearRegression()
model.fit(X, y)

col1, col2 = st.columns(2)

exp = col1.slider("Years of Experience", 0, 15, 3)
skill = col2.slider("Skill Score", 50, 100, 75)

if st.button("Predict Salary"):
    pred_salary = model.predict([[exp, skill]])[0]
    st.success(f"💰 Predicted Salary: €{pred_salary:.2f}K")

# -----------------------------
# Data Preview
# -----------------------------
st.subheader("📄 Simulated Dataset Preview")
st.dataframe(filtered_df.head(20))
