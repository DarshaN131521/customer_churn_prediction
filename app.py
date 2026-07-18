import streamlit as st
import pandas as pd
import joblib



st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
) 


@st.cache_resource
def load_models():

    preprocessor = joblib.load("models/preprocessor.pkl")

    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "Decision Tree": joblib.load("models/decision_tree.pkl"),
        "Tuned Decision Tree": joblib.load("models/tuned_decision_tree.pkl"),
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "GridSearch Random Forest": joblib.load("models/gridsearch_random_forest.pkl")
    }

    return preprocessor, models

preprocessor, models = load_models()


st.title("📊 Customer Churn Prediction Dashboard")

st.subheader("Welcome to the Customer Churn Prediction App.😊")

st.markdown("""
Welcome to the **Customer Churn Prediction Dashboard**.

This application predicts whether a customer is likely to churn using multiple Machine Learning models.

Select a model from the sidebar, enter customer information, and click **Predict**.
""")

st.sidebar.title("⚙️ Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "Tuned Decision Tree",
        "Random Forest",
        "GridSearch Random Forest"
    )
)

model = models[selected_model]

st.sidebar.success(f"Selected Model:\n\n{selected_model}")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Project Info")

st.sidebar.write("Dataset: IBM Telco Customer Churn")

st.sidebar.write("Number of Algorithms: 5")

st.sidebar.write("Primary Model :\n\nLogistic Regression")


model_metrics = {
    "Logistic Regression": {
        "Accuracy": 0.805,
        "Precision": 0.652,
        "Recall": 0.575,
        "F1 Score": 0.611,
        "ROC-AUC": 0.836
    },

    "Decision Tree": {
        "Accuracy": 0.710,
        "Precision": 0.456,
        "Recall": 0.473,
        "F1 Score": 0.465,
        "ROC-AUC": 0.634
    },

    "Tuned Decision Tree": {
        "Accuracy": 0.785,
        "Precision": 0.611,
        "Recall": 0.529,
        "F1 Score": 0.567,
        "ROC-AUC": 0.815
    },

    "Random Forest": {
        "Accuracy": 0.790,
        "Precision": 0.631,
        "Recall": 0.508,
        "F1 Score": 0.563,
        "ROC-AUC": 0.817
    },

    "GridSearch Random Forest": {
        "Accuracy": 0.798,
        "Precision": 0.655,
        "Recall": 0.508,
        "F1 Score": 0.572,
        "ROC-AUC": 0.836
    }
}

metrics = model_metrics[selected_model]

st.subheader("📈 Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", metrics["Accuracy"])
col2.metric("Precision", metrics["Precision"])
col3.metric("Recall", metrics["Recall"])
col4.metric("F1 Score", metrics["F1 Score"])
col5.metric("ROC-AUC", metrics["ROC-AUC"])



st.subheader("📝 Customer Information")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)


    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            help="Select 1 if the customer is a senior citizen, otherwise select 0."
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )


    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1000.0
        )

    st.markdown("---")

    predict_button = st.form_submit_button(
        "🔮 Predict Churn",
        use_container_width=True
    )

if predict_button:

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    try:

        with st.spinner("🔄 Predicting..."):

            transformed_data = preprocessor.transform(input_data)

            prediction = model.predict(transformed_data)[0]

            probability = model.predict_proba(transformed_data)[0][1]

    except Exception as e:

        st.error("Prediction Failed!")

        st.exception(e)

        st.stop()

        st.markdown("---")


    with st.expander("📋 Customer Details Used for Prediction"):

        st.dataframe(input_data)

    st.subheader("🎯 Prediction Result")

    if prediction == 1:

        st.error("⚠️ Customer is likely to Churn")

    else:

        st.success("✅ Customer is NOT likely to Churn")

    st.write(f"**Probability of Churn:** {probability:.2%}")

    st.progress(float(probability))

    st.info(f"🤖 Prediction made using **{selected_model}**")

st.markdown("---")
st.caption(
    "Developed by Darshan Panchal | Python • Scikit-Learn • Streamlit | "
    "Gmail: darshanpanchal151102@gmail.com | "
    "GitHub: https://github.com/DarshaN131521"
)
