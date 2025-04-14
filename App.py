import streamlit as st
import pandas as pd
import altair as alt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide", page_title="SUD Risk Analyzer")
st.title("💊 SUD Risk Analysis Dashboard")

# File uploader
uploaded_file = st.sidebar.file_uploader("📁 Upload Excel or CSV", type=["xlsx", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    # Sidebar filters
    st.sidebar.header("Filter Data")
    selected_gender = st.sidebar.multiselect("Gender", options=df['GENDER'].unique(), default=df['GENDER'].unique())
    selected_employment = st.sidebar.multiselect("Employment Status", options=df['Employment status - current'].unique(), default=df['Employment status - current'].unique())
    age_range = st.sidebar.slider("AGE GROUP", int(df['AGE'].min()), int(df['AGE'].max()), (30, 60))
    selected_SUDRISK = st.sidebar.multiselect("SUD Risk", options=df['sud risk'].unique(), default=df['sud risk'].unique())
    selected_Marital = st.sidebar.multiselect("Marital Status", options=df['MARITAL'].unique(), default=df['MARITAL'].unique())

    all_cities = df['CITY'].dropna().unique().tolist()
    all_cities.sort()

    city_options = ["All"] + all_cities

    selected_cities = st.multiselect(
        "📍 Select City",
        options=city_options,
        default=["All"]
    )

    # Filter logic
    if "All" in selected_cities:
        filtered_df = df.copy()
    else:
        filtered_df = df[df["CITY"].isin(selected_cities)]

    
    # Apply filters
    filtered_df = df[
        (df['GENDER'].isin(selected_gender)) &
        (df['Employment status - current'].isin(selected_employment)) &
        (df['AGE'].between(age_range[0], age_range[1]))&
        (df['sud risk'].isin(selected_SUDRISK)) &
        (df['MARITAL'].isin(selected_Marital))
    ]

    # Tabs
    tab1, tab2 = st.tabs(["📊 Visualizations", "🤖 Predictive Modeling"])

    # ----------------------------- 📊 Visualization Tab -----------------------------
    with tab1:
        st.subheader("📈 Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Patients", len(filtered_df))
        col2.metric("⚠️ At Risk (SUD)", filtered_df['sud risk'].value_counts().get("yes", 0))
        col3.metric("💵 Avg Income", f"${filtered_df['INCOME'].mean():,.2f}")
        col4.metric("🏥 Avg Healthcare Cost", f"${filtered_df['HEALTHCARE_EXPENSES'].mean():,.2f}")

        st.markdown("---")

        # 1. SUD Risk by Age Group (Altair)
        st.markdown("### 📊 SUD Risk by Age Group")
        chart1 = alt.Chart(filtered_df).mark_bar().encode(
            y=alt.Y('AGE GROUP:N', sort='-x'),
            x=alt.X('count():Q', title='SUD Count'),
            color=alt.Color('sud risk:N', scale=alt.Scale(scheme='set1')),
            tooltip=['AGE GROUP', 'sud risk', 'count()']
        ).properties(height=300)
        st.altair_chart(chart1, use_container_width=True)

        # 2. Pie Chart - Race
        st.markdown("### 🧬 SUD Risk by Race")
        pie_data = filtered_df['RACE'].value_counts().reset_index()
        pie_data.columns = ['RACE', 'Count']
        chart2 = alt.Chart(pie_data).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="RACE", type="nominal", scale=alt.Scale(scheme='category20b')),
            tooltip=['RACE', 'Count']
        )
        st.altair_chart(chart2, use_container_width=True)

        # 3. Average Income by SUD Risk (Seaborn)
        st.markdown("### 🧍‍♂️🧍‍♀️ Average INCOME by SUD Risk")

        # Grouping by INCOME and sud risk, calculating mean healthcare expenses
        INCOME_group = filtered_df.groupby(['sud risk'])['INCOME'].mean().reset_index()

        # Altair chart
        INCOME_chart = alt.Chart(INCOME_group).mark_bar().encode(
            x=alt.X('sud risk:N', title='Sud Risk'),
            y=alt.Y('INCOME:Q', title='INCOME'),
            color=alt.Color('sud risk:N', title='SUD Risk', scale=alt.Scale(scheme='set2')),
            tooltip=['INCOME', 'sud risk']
        ).properties(height=300)

        st.altair_chart(INCOME_chart, use_container_width=True)

        # 4. Income vs Stress and Tobacco
        st.markdown("### 📊 Average Income by Stress Level and Smoking Status")

        # Group and aggregate the data
        income_group = filtered_df.groupby(['Stress level', 'Tobacco smoking status'])['INCOME'].mean().reset_index()

        # Create Altair chart
        income_chart = alt.Chart(income_group).mark_bar().encode(
            x=alt.X('INCOME:Q', title='Average Income'),
            y=alt.Y('Stress level:N', title='Stress Level', sort='-x'),
            color=alt.Color('Tobacco smoking status:N', title='Smoking Status', scale=alt.Scale(scheme='tableau20')),
            tooltip=['Stress level', 'Tobacco smoking status', alt.Tooltip('INCOME:Q', format=",.0f")]
        ).properties(
            height=400
        )

        st.altair_chart(income_chart, use_container_width=True)



    # ----------------------------- 🤖 Predictive Modeling Tab -----------------------------
    with tab2:
        st.subheader("🔮 Predict SUD Risk")
        st.markdown("Predict risk of Substance Use Disorder (SUD) using health and socio-economic factors.")

    if uploaded_file:
        # Load data
        final_df = pd.read_excel(uploaded_file)

        # Preprocess
        oneHotCols = [
            "MARITAL", "RACE", "ETHNICITY", "GENDER", "Tobacco smoking status",
            "Housing status", "Employment status - current", "Highest level of education",
            "Primary insurance", "Stress level"
        ]
        data = pd.get_dummies(final_df, columns=oneHotCols)

        # Define features and labels
        X = data.drop(["sud risk", "Id", "CITY", "STATE", "AGE", "AGE GROUP"], axis=1)
        y = data["sud risk"]

        # Split the data
        x_train, x_rest, y_train, y_rest = train_test_split(X, y, test_size=0.2, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=0.5, random_state=42)

        # Train the model
        gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
        gb_clf.fit(x_train, y_train)

        st.markdown("---")
        st.markdown("### 🧾 Enter Patient Details for Prediction")

        # Start the form
        with st.form("sud_prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0)
                income = st.number_input("Annual Income ($)", min_value=0)
                expenses = st.number_input("Healthcare Expenses ($)", min_value=0)
                hba1c = st.number_input("Hemoglobin A1c (%)", min_value=0.0)
                coverage = st.number_input("Healthcare Coverage (%)", min_value=0.0)

            with col2:
                hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=0.0)
                glucose = st.number_input("Glucose (mg/dL)", min_value=0.0)
                cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=0.0)
                systolic = st.number_input("Systolic BP (mmHg)", min_value=0.0)
                bmi = st.number_input("BMI", min_value=0.0)

            # Submit button inside the form
            submitted = st.form_submit_button("🔍 Predict")

            if submitted:
                user_input = {
                    "Creatinine [Mass/volume] in Blood": creatinine,
                    "INCOME": income,
                    "HEALTHCARE_EXPENSES": expenses,
                    "Hemoglobin A1c/Hemoglobin.total in Blood": hba1c,
                    "HEALTHCARE_COVERAGE": coverage,
                    "Cholesterol in HDL [Mass/volume] in Serum or Plasma": hdl,
                    "Glucose [Mass/volume] in Blood": glucose,
                    "Cholesterol [Mass/volume] in Serum or Plasma": cholesterol,
                    "Systolic Blood Pressure": systolic,
                    "Body mass index (BMI) [Ratio]": bmi
                }

                user_df = pd.DataFrame([user_input])

                # Align user_df with training features
                missing_cols = set(X.columns) - set(user_df.columns)
                for col in missing_cols:
                    user_df[col] = 0
                user_df = user_df[X.columns]  # Arrange columns

                # Make prediction
                prediction = gb_clf.predict(user_df)[0]
                prediction_proba = gb_clf.predict_proba(user_df).max()

                st.success(f"🎯 Predicted SUD Risk: **{prediction.upper()}** (Confidence: `{prediction_proba:.2f}`)")

    else:
        st.info("📎 Please upload a dataset to continue.")
