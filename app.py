import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="InsightAI",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# Title
# -------------------------------
st.title("📊 InsightAI")
st.subheader("AI-Powered Data Analysis Assistant")

st.write("""
Welcome to **InsightAI**.

This application helps you:

- 📁 Upload CSV or Excel files
- 📊 Analyze datasets automatically
- 📈 Visualize data with interactive charts
- 🤖 Chat with your data using AI
- 🔮 Generate Machine Learning predictions
""")

# -------------------------------
# Upload Section
# -------------------------------
st.divider()
st.header("📂 Upload Your Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx"]
)

# -------------------------------
# Dataset Analysis
# -------------------------------
if uploaded_file is not None:

    # Load Dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ Dataset uploaded successfully!")

    # -------------------------------
    # Dataset Preview
    # -------------------------------
    st.subheader("📄 Dataset Preview")
    st.dataframe(df)

    # -------------------------------
    # Dataset Information
    # -------------------------------
    st.subheader("📊 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.write("### Column Names")
    st.write(list(df.columns))

    # -------------------------------
    # Missing Values
    # -------------------------------
    st.subheader("❗ Missing Values")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        st.success("No missing values found! 🎉")
    else:
        st.dataframe(missing)

    # -------------------------------
    # Duplicate Rows
    # -------------------------------
    st.subheader("🧹 Duplicate Rows")

    duplicates = df.duplicated().sum()

    if duplicates == 0:
        st.success("No duplicate rows found! 🎉")
    else:
        st.warning(f"Duplicate Rows: {duplicates}")

    # -------------------------------
    # Data Types
    # -------------------------------
    st.subheader("📋 Data Types")

    data_types = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(data_types)

    # -------------------------------
    # Statistical Summary
    # -------------------------------
    st.subheader("📈 Statistical Summary")

    st.dataframe(df.describe())
    #intractive 📊 Interactive Visualization------------------

    st.subheader("📊 Interactive Visualization")
    numeric_columns = df.select_dtypes(include=["number"]).columns

    selected_column = st.selectbox(
        "selected a Numeric Column",
        numeric_columns
    )
    
    fig = px.histogram(
        df,
        x=selected_column,
        title=f"Distribution of {selected_column}"
    )

    st.plotly_chart(fig, use_container_width=True)