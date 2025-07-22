import streamlit as st
import pandas as pd
import joblib

# Load model & transformers
model = joblib.load("model/model_knn.pkl")
encoder = joblib.load("model/encoder.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title(" PayGuage ")
st.markdown("Predict whether a person earns **more or less than $50,000 annually** in USD based on demographic and work details.")

# Education level mapping for UI
education_map = {
    1: 'Preschool',
    2: '1st-4th',
    3: '5th-6th',
    4: '7th-8th',
    5: '9th',
    6: '10th',
    7: '11th',
    8: '12th',
    9: 'HS-grad',
    10: 'Some-college',
    11: 'Assoc-voc',
    12: 'Assoc-acdm',
    13: 'Bachelors',
    14: 'Masters',
    15: 'Doctorate'
}

# UI Inputs
age = st.slider("Age", 17, 75, 30)
workclass = st.selectbox("Workclass", encoder['workclass'].classes_)
education_name = st.selectbox("Education Level", list(education_map.values()))
education_num_key = [k for k, v in education_map.items() if v == education_name][0]
marital_status = st.selectbox("Marital Status", encoder['marital-status'].classes_)
occupation = st.selectbox("Occupation", encoder['occupation'].classes_)
relationship = st.selectbox("Relationship", encoder['relationship'].classes_)
gender = st.selectbox("Gender", encoder['gender'].classes_)
capital_gain = st.number_input("Capital Gain ($)", 0, 100000, 0)
capital_loss = st.number_input("Capital Loss ($)", 0, 100000, 0)
hours_per_week = st.slider("Hours per Week", 1, 100, 40)
native_country = st.selectbox("Native Country", encoder['native-country'].classes_)

if st.button("Predict"):
    # Create DataFrame
    input_df = pd.DataFrame([[
        age, workclass, education_num_key, marital_status, occupation,
        relationship, gender, capital_gain, capital_loss, hours_per_week,
        native_country
    ]], columns=[
        'age', 'workclass', 'educational-num', 'marital-status', 'occupation',
        'relationship', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
        'native-country'
    ])

    # Encode categorical features
    for col in encoder:
        input_df[col] = encoder[col].transform(input_df[col])

    # Scale numeric features
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)

    # Show result
    if prediction[0] == '>50K':
        st.success("💰 Predicted Income: **> $50,000 annually**")
    else:
        st.success("💰 Predicted Income: **≤ $50,000 annually**")
