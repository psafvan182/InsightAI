import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
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

    # =====================================================
    # NUMERIC VISUALIZATION
    # =====================================================
    st.subheader("📊 Interactive Visualization")

    numeric_columns = df.select_dtypes(include=["number"]).columns

    selected_column = st.selectbox(
        "Select a Numeric Column",
        numeric_columns
    )

    chart_type = st.selectbox(
        "Select Chart Type",
        [
            "Histogram",
            "Box Plot",
            "Bar Chart"
        ]
    )

    if chart_type == "Histogram":

        fig = px.histogram(
            df,
            x=selected_column,
            title=f"Distribution of {selected_column}"
        )

    elif chart_type == "Box Plot":

        fig = px.box(
            df,
            y=selected_column,
            title=f"Box Plot of {selected_column}"
        )

    else:

        counts = df[selected_column].value_counts().reset_index()
        counts.columns = [selected_column, "Count"]

        fig = px.bar(
            counts,
            x=selected_column,
            y="Count",
            title=f"Bar Chart of {selected_column}"
        )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # CATEGORICAL VISUALIZATION
    # =====================================================
    st.subheader("📊 Categorical Data Visualization")

    categorical_columns = df.select_dtypes(include=["object"]).columns

    if len(categorical_columns) > 0:

        categorical_column = st.selectbox(
            "Select a Categorical Column",
            categorical_columns
        )


    
        chart = st.selectbox(
            "Select Category Chart",
            [
                "Bar Chart",
                "Pie Chart"
            ]
        )

        counts = df[categorical_column].value_counts().reset_index()
        counts.columns = [categorical_column, "Count"]

        if chart == "Bar Chart":

            fig = px.bar(
                counts,
                x=categorical_column,
                y="Count",
                title=f"{categorical_column} Distribution"
            )

        else:

            fig = px.pie(
                counts,
                names=categorical_column,
                values="Count",
                title=f"{categorical_column} Distribution"
            )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔥 Correlation Heatmap")

        numeric_df = df.select_dtypes(include="number")
        corr = numeric_df.corr()

        fig =   ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            annotation_text=corr.round(2).values,
            colorscale="Viridis",
            showscale=True

        )

        st.plotly_chart(fig,use_container_width=True)



    else:
        st.info("No categorical columns found in this dataset.")