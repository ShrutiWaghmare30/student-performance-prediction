import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open("student_performance_model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

st.title("Student Performance Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
hours_studied = st.slider("Hours Studied", 0, 12, 6)
attendance = st.slider("Attendance (%)", 0, 100, 75)
previous_score = st.slider("Previous Exam Score", 0, 100, 60)
sleep_hours = st.slider("Sleep Hours", 0, 10, 7)
study_breaks = st.slider("Study Breaks (hours)", 0, 5, 1)

# Encode Gender (same as training)
gender_val = 1 if gender == "Male" else 0

# Create input array
input_data = np.array([[gender_val, hours_studied, attendance,
                        previous_score, sleep_hours, study_breaks]])

# 🔥 VERY IMPORTANT: scale input
input_data_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data_scaled)[0]
    st.success(f"Predicted Exam Score: {prediction:.2f}")
