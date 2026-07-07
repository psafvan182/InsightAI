import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor 
from sklearn.metrics import accuracy_score, mean_absolute_error
# from pandas.api.types import is_numeric_dtype
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

    categorical_columns = df.select_dtypes(include=["object", "string"]).columns

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
# =====================================================
# DATA CLEANING
# =====================================================


        st.subheader("🧹 Data Cleaning")

        clean_option = st.selectbox(
            "Choose a Cleaning Operation",
            [
                "Select",
                "Remove Missing Values",
                "Fill Missing Values",
                "Remove Duplicate Rows"
            ]
        )

        #Remove missing values 

        if clean_option == "Remove Missing Values":

            orginal_rows = df.shape[0]

            cleaned_df = df.dropna()
            removed_rows = orginal_rows - cleaned_df.shape[0]

            st.success(f"✅ Removed {removed_rows} rows with missing values.")
            st.write("###Cleaned Data Set")
            st.dataframe(cleaned_df)


            # Fill missing Values
            # -----------------------

        elif clean_option == "Fill Missing Values":

            filled_df = df.copy()

            numeric_columns = filled_df.select_dtypes(include=["number"]).columns

            filled_df[numeric_columns] = filled_df[numeric_columns].fillna(
                filled_df[numeric_columns].mean()
            )

            # Fill categorical columns with mode

            categorical_columns = filled_df.select_dtypes(include=["object"]).columns

            for col in categorical_columns:
                filled_df[col] = filled_df[col].fillna(
                    filled_df[col].mode()[0]

                )

            st.success("✅ Missing values filled successfully.")

            st.write("### Cleaned Dataset")
            st.dataframe(filled_df)

            # Remove Duplicate Rows

        elif clean_option == "Remove Duplicate Rows":

            orginal_rows = df.shape[0]

            cleaned_df = df.drop_duplicates()
            removed_rows = orginal_rows - cleaned_df.shape[0]

            st.success(f"✅ Removed {removed_rows} duplicate rows.")
            st.write("### Cleaned Dataset")
            st.dataframe(cleaned_df)

            st.download_button(
                label="📥 Download Cleaned Dataset",
                data=cleaned_df.to_csv(index=False),
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )


    # =====================================================
    # MACHINE LEARNING
    # =====================================================

    st.header("🤖 Machine Learning")

    target_column = st.selectbox(
        "🎯 Select Target Column",
        df.columns
    )

# ==========================================
# Validate Target Column
# ==========================================


    invalid_targets = [
        "id",
        "customerid",
        "customer_id",
        "employeeid",
        "employee_id",
        "name",
        "rollno",
        "roll_no"
    ]

    if target_column.lower() in invalid_targets:
        st.error(
    f"❌ '{target_column}' is an identifier column.\n\n"
    "Please select a meaningful target like Salary, Purchased, Churn, Price or Income."
)

        st.stop()

    # Detect problem type
    from pandas.api.types import is_numeric_dtype

    if is_numeric_dtype(df[target_column]):
        problem_type = "Regression"
    else:
        problem_type = "Classification"

    st.info(f"🧠 Detected Problem Type: {problem_type}")

    train_button = st.button("🚀 Train Model")

    if train_button:

        # ==========================================
        # Features & Target
        # ==========================================

        df = df.dropna(subset=[target_column])


        X = df.drop(columns=[target_column])
# ==========================================
# Remove Identifier Columns
# ==========================================

        identifier_columns =   [
            "id",
            "customerid",
            "customer_id",
            "employeeid",
            "employee_id",
            "name",
            "rollno",
            "roll_no"
        ]

        columns_to_remove = []
        for col in X.columns:
            if col.lower() in identifier_columns:
                columns_to_remove.append(col)

        if columns_to_remove:
            X = X.drop(columns=columns_to_remove)
            st.warning(f"⚠️ Removed identifier columns: {columns_to_remove}")

        y = df[target_column]

        st.success("✅ Features and Target created successfully!")

        st.write("### Features (X)")
        st.dataframe(X.head())

        st.write("### Target (y)")
        st.dataframe(y.head())

        # ==========================================
        # Label Encoding
        # ==========================================

        categorical_columns = X.select_dtypes(include=["object", "string"]).columns

        if len(categorical_columns) > 0:

            label_encoder = LabelEncoder()

            for col in categorical_columns:
                X[col] = label_encoder.fit_transform(X[col].astype(str))

            st.success("✅ Categorical columns encoded successfully!")

        else:
            st.info("ℹ️ No categorical columns found.")

        # ==========================================
        # Train-Test Split
        # ==========================================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        st.success("✅ Train-Test Split completed!")

        st.write("### Training Data Shape")
        st.write(X_train.shape)

        st.write("### Testing Data Shape")
        st.write(X_test.shape)

        # ==========================================
        # Train Model
        # ==========================================

        if problem_type == "Regression":
            model = RandomForestRegressor(random_state=42)
        else:
            model = RandomForestClassifier(random_state=42)

        model.fit(X_train, y_train)

        st.success("✅ Model trained successfully!")

        # ==========================================
        # Prediction
        # ==========================================

        predictions = model.predict(X_test)

        st.success("✅ Prediction completed!")

        st.subheader("📋 Predictions")

        prediction_df = pd.DataFrame({
            "Prediction": predictions
        })

        st.dataframe(prediction_df)

        
        st.subheader("⭐ Feature Importance")
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        st.dataframe(importance_df)

        fig = px.bar(
            importance_df,
            x="Feature",
            y="Importance",
            title="Feature Importance"
        )

        st.plotly_chart(fig, use_container_width=True)

        

        # ==========================================
        # Model Evaluation
        # ==========================================

        st.subheader("📈 Model Evaluation")

        if problem_type == "Classification":

            accuracy = accuracy_score(y_test, predictions)

            st.success(f"🎯 Model Accuracy: {accuracy:.2%}")

        else:

            mae = mean_absolute_error(y_test, predictions)

            st.success(f"📉 Mean Absolute Error (MAE): {mae:.2f}")

        # ==========================================
        # Actual vs Predicted
        # ==========================================

        st.subheader("📋 Actual vs Predicted")

        results = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": predictions
        })

        st.dataframe(results)
# ==========================================
# Download Predictions
# ==========================================


        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"

        )


        #    if problem_type == "Regression":


  
    # else:
    #     st.info("No categorical columns found in this dataset.")