import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Employee Attrition Prediction")

st.title("📊 Employee Attrition Prediction App")

# -------------------- SAFE LOAD FUNCTION --------------------
def load_file(file):
    if not os.path.exists(file):
        st.error(f"❌ Missing file: {file}")
        st.stop()
    try:
        with open(file, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading {file}: {e}")
        st.stop()

# -------------------- LOAD FILES --------------------
model = load_file("attrition_model.pkl")
scaler = load_file("scaler.pkl")
columns = load_file("columns.pkl")
encoders = load_file("encoders.pkl")

st.success("✅ Model loaded successfully")

# -------------------- INPUTS --------------------
st.subheader("Enter Employee Details")

Education = st.selectbox("Education", [1, 2, 3])
JoiningYear = st.number_input("Joining Year", 2000, 2025)

# Safe encoder usage
try:
    City = st.selectbox("City", list(encoders["City"].classes_))
except:
    City = st.text_input("City (enter manually)")

PaymentTier = st.selectbox("Payment Tier", [1, 2, 3])
Age = st.number_input("Age", 18, 60)
Gender = st.selectbox("Gender", ["Male", "Female"])
EverBenched = st.selectbox("Ever Benched", ["Yes", "No"])
ExperienceInCurrentDomain = st.slider("Experience (Years)", 0, 10)

# -------------------- PREDICTION --------------------
if st.button("Predict"):

    try:
        # Encode categorical values safely
        gender_val = 1 if Gender == "Male" else 0
        bench_val = 1 if EverBenched == "Yes" else 0

        if "City" in encoders:
            city_val = encoders["City"].transform([City])[0]
        else:
            city_val = 0

        # Create input array
        input_data = np.array([[Education, JoiningYear, city_val,
                                PaymentTier, Age, gender_val,
                                bench_val, ExperienceInCurrentDomain]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Output
        if prediction == 1:
            st.error("⚠️ Employee is likely to leave")
        else:
            st.success("✅ Employee is likely to stay")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")