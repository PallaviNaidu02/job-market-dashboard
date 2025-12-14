import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Iris Flower Classification",
    layout="wide"
)

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    return pd.read_csv(url)

df = load_data()

# Sidebar
st.sidebar.title("🌸 Controls")
species = st.sidebar.multiselect(
    "Select Species",
    options=df["species"].unique(),
    default=df["species"].unique()
)

filtered_df = df[df["species"].isin(species)]

# Title
st.title("🌼 Iris Flower Species Classification Dashboard")
st.markdown("Explore flower data and predict species using machine learning")

# Metrics
c1, c2, c3 = st.columns(3)
c1.metric("Total Samples", len(filtered_df))
c2.metric("Features Used", 4)
c3.metric("Species Count", filtered_df["species"].nunique())

# Data Preview
st.subheader("📄 Dataset Preview")
st.dataframe(filtered_df.head(20))

# Pairplot
st.subheader("📊 Feature Relationships")
fig = sns.pairplot(filtered_df, hue="species")
st.pyplot(fig)

# Train model
X = df.drop("species", axis=1)
y = df["species"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Prediction Section
st.subheader("🔮 Predict Flower Species")

col1, col2, col3, col4 = st.columns(4)

sepal_length = col1.slider("Sepal Length", float(df.sepal_length.min()), float(df.sepal_length.max()), 5.0)
sepal_width  = col2.slider("Sepal Width",  float(df.sepal_width.min()),  float(df.sepal_width.max()),  3.0)
petal_length = col3.slider("Petal Length", float(df.petal_length.min()), float(df.petal_length.max()), 4.0)
petal_width  = col4.slider("Petal Width",  float(df.petal_width.min()),  float(df.petal_width.max()),  1.0)

if st.button("Predict Species"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    st.success(f"🌸 Predicted Species: **{prediction}**")
