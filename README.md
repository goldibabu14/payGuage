#  Payguage 

Payguage is a machine learning web app that predicts whether a person's income exceeds **$50,000 annually** based on demographic and work-related features from the **U.S. Census Bureau's Adult Income dataset**. It is designed with a modern and user-friendly interface using **Streamlit**.

---

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository - Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Purpose**: Predict if an individual's annual income is `>50K` or `<=50K`
- **Demographics**: This dataset represents US population statistics and income brackets.
- **Note**: The "race" feature has been intentionally removed from this project to maintain ethical AI standards.

---

## 🚀 Features

- Clean and intuitive Streamlit UI
- Predicts whether a person earns over $50K/year in **USD**
- Dropdown for readable **education levels** (e.g., Masters, 12th, etc.)
- Categorical feature encoding and feature scaling
- 🔒 Fair model — no use of racial features
- Model trained using **K-Nearest Neighbors (KNN)** algorithm

---

## 🔧 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn (KNN, Label Encoding, Scaling)
- Streamlit (Web UI)
- Joblib (Model Serialization)

---

## Live Demo

Try out the live Salary Predictor app: [PayGuage on Streamlit](https://payguage.streamlit.app/)



