# 📊 Customer Churn Prediction System

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn based on demographic information, account details, and subscribed services.

The project includes data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, model evaluation, and deployment using **Streamlit**.

---

## 🚀 Live Features

- Predict customer churn using multiple Machine Learning models.
- Interactive Streamlit web application.
- Compare predictions from different trained models.
- Display prediction probability and confidence.
- View model performance metrics.
- Clean and user-friendly dashboard.

  
## 🚀 Live Demo

**Streamlit App:** https://telecom-customer-churn-darshan.streamlit.app
---

## 📂 Dataset

**Dataset:** IBM Telco Customer Churn Dataset

Source:

- IBM: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv
- Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains customer demographic information, subscribed services, billing details, and churn status.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit
- Joblib

---

## 📈 Machine Learning Workflow

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Data Preprocessing
   - One-Hot Encoding
   - Standard Scaling
   - ColumnTransformer
5. Train-Test Split
6. Model Training
7. Hyperparameter Tuning
8. Model Evaluation
9. Model Serialization
10. Streamlit Deployment

---

## 🤖 Models Trained

- Logistic Regression
- Decision Tree
- Tuned Decision Tree
- Random Forest
- GridSearchCV Tuned Random Forest

---

## 📊 Model Performance

| Model                    |   Accuracy |  Precision |     Recall |   F1 Score |   ROC-AUC |
| ------------------------ | ---------: | ---------: | ---------: | ---------: | --------: |
| Logistic Regression      | **80.53%** | **65.15%** | **57.49%** | **61.08%** | **0.836** |
| Decision Tree            |     71.00% |     45.62% |     47.33% |     46.46% |     0.634 |
| Tuned Decision Tree      |     78.54% |     61.11% |     52.94% |     56.73% |     0.815 |
| Random Forest            |     79.03% |     63.12% |     50.80% |     56.30% |     0.817 |
| GridSearch Random Forest |     79.82% |     65.52% |     50.80% |     57.23% |     0.836 |

**Best Overall Model:** Logistic Regression

---

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── Customer_Churn_Prediction.ipynb
├── README.md
├── requirements.txt
│
├── dataset/
│   └── Telco-Customer-Churn.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── tuned_decision_tree.pkl
│   ├── random_forest.pkl
│   ├── gridsearch_random_forest.pkl
│   └── preprocessor.pkl
│
└── images/
```

---

## ▶️ How to Run

### Clone the repository

```bash
git clone https://github.com/yourusername/Customer-Churn-Prediction.git
```

### Navigate to the project

```bash
cd Customer-Churn-Prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📸 Application Preview

> Add screenshots of your Streamlit dashboard here.

Example:

- Home Page
- Prediction Result
- Model Comparison

---

## 🎯 Key Learnings

Through this project I gained practical experience in:

- Data preprocessing using ColumnTransformer
- Building reusable ML pipelines
- Comparing multiple classification algorithms
- Hyperparameter tuning using GridSearchCV
- Model evaluation using multiple metrics
- Saving and loading trained models using Joblib
- Deploying Machine Learning models using Streamlit

---

## 🔮 Future Improvements

- Add XGBoost and LightGBM models.
- Deploy the application online using Streamlit Community Cloud.
- Improve UI with custom themes.
- Add downloadable prediction reports.
- Explain predictions using SHAP.

---

## 👨‍💻 Author

**Darshan Panchal**


