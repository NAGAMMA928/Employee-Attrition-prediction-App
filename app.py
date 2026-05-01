import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os

st.set_page_config(page_title="Employee Attrition Prediction")

st.title("💼 Employee Attrition Prediction App")

# -------- CHECK FILES --------
required_files = ["attrition_model.pkl", "scaler.pkl", "columns.pkl", "encoders.pkl"]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"❌ Missing file: {file}")
        st.stop()

# -------- LOAD FILES --------
model = pickle.load(open("attrition_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

st.write("Enter employee details:")

# -------- INPUTS (MATCH DATASET) --------
Education = st.selectbox("Education", [1, 2, 3])
JoiningYear = st.number_input("Joining Year", 2000, 2025)
City = st.selectbox("City", encoders["City"].classes_)
PaymentTier = st.selectbox("Payment Tier", [1, 2, 3])
Age = st.number_input("Age", 18, 60)
Gender = st.selectbox("Gender", encoders["Gender"].classes_)
EverBenched = st.selectbox("Ever Benched", encoders["EverBenched"].classes_)
Experience = st.number_input("Experience in Current Domain", 0, 20)

# -------- ENCODE INPUT --------
def prepare_input():
    data = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    input_dict = {
        "Education": Education,
        "JoiningYear": JoiningYear,
        "City": encoders["City"].transform([City])[0],
        "PaymentTier": PaymentTier,
        "Age": Age,
        "Gender": encoders["Gender"].transform([Gender])[0],
        "EverBenched": encoders["EverBenched"].transform([EverBenched])[0],
        "ExperienceInCurrentDomain": Experience
    }

    for col in input_dict:
        if col in data.columns:
            data[col] = input_dict[col]

    return data

# -------- PREDICTION --------
if st.button("Predict"):

    input_df = prepare_input()

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    st.subheader("Result:")

    if prediction == 1:
        st.error("⚠️ Employee is likely to leave")
    else:
        st.success("✅ Employee is likely to stay")

    st.write(f"📊 Stay Probability: {prob[0]:.2f}")
    st.write(f"📊 Leave Probability: {prob[1]:.2f}")