import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="InsightAI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 InsightAI")
st.subheader("AI-Powered Data Analysis Assistant")

st.write(
    """
Welcome to **InsightAI**.

This application helps you:

- 📁 Upload CSV or Excel files
- 📊 Analyze datasets automatically
- 📈 Visualize data with interactive charts
- 🤖 Chat with your data using AI
- 🔮 Generate Machine Learning predictions
"""
)

if st.button("🚀 Start Analysis"):
    st.divider()
    st.header("📂 Upload Your Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.subheader("📊 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.write("### Column Names")
    st.write(list(df.columns))

    st.subheader("❗ Missing Values")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if missing.empty:
        st.success("No missing values found! 🎉")
    else:
        st.dataframe(missing)