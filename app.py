import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Job Market & Salary Dashboard",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/2018_usa_jobs.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Clean salary column
df['Salary Estimate'] = df['Salary Estimate'].str.replace('$','', regex=False)
df['Salary Estimate'] = df['Salary Estimate'].str.replace('K','', regex=False)
df[['Min Salary','Max Salary']] = df['Salary Estimate'].str.split('-', expand=True)
df['Min Salary'] = pd.to_numeric(df['Min Salary'], errors='coerce')
df['Max Salary'] = pd.to_numeric(df['Max Salary'], errors='coerce')
df['Avg Salary'] = (df['Min Salary'] + df['Max Salary']) / 2
df = df.dropna(subset=['Avg Salary'])

# Sidebar
st.sidebar.title("🔍 Filters")

state = st.sidebar.multiselect(
    "Select State",
    options=df['State'].unique(),
    default=['CA', 'NY']
)

job_keyword = st.sidebar.text_input("Job Title Keyword", "Data")

filtered_df = df[
    (df['State'].isin(state)) &
    (df['Title'].str.contains(job_keyword, case=False))
]

# Dashboard Title
st.title("💼 Job Market & Salary Trend Analysis")
st.markdown("Interactive dashboard to explore job demand and salary trends")

# Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total Jobs", len(filtered_df))
col2.metric("Average Salary ($K)", round(filtered_df['Avg Salary'].mean(), 2))
col3.metric("Top State", filtered_df['State'].mode()[0])

# Salary Distribution
st.subheader("📊 Salary Distribution")

fig, ax = plt.subplots()
ax.hist(filtered_df['Avg Salary'], bins=30)
ax.set_xlabel("Salary ($K)")
ax.set_ylabel("Number of Jobs")
st.pyplot(fig)

# Top Job Titles
st.subheader("🏷️ Top Job Titles")

top_titles = filtered_df['Title'].value_counts().head(10)

fig2, ax2 = plt.subplots()
top_titles.plot(kind='barh', ax=ax2)
ax2.set_xlabel("Number of Jobs")
st.pyplot(fig2)

# Location Analysis
st.subheader("📍 Jobs by State")

state_counts = filtered_df['State'].value_counts().head(10)

fig3, ax3 = plt.subplots()
state_counts.plot(kind='bar', ax=ax3)
ax3.set_ylabel("Job Count")
st.pyplot(fig3)

# Data Preview
st.subheader("📄 Data Preview")
st.dataframe(filtered_df[['Title', 'Company', 'State', 'Avg Salary']].head(20))
