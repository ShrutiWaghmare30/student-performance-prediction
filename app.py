import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load trained model & scaler
with open("student_performance_model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

# Title & description
st.markdown("<h1 style='text-align: center;'>🎓 Student Performance Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter student details to predict final performance</p>", unsafe_allow_html=True)
st.divider()

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        study_hours = st.slider("Study Hours per Day", 0, 10, 3)
        attendance = st.slider("Attendance (%)", 50, 100, 75)

    with col2:
        previous_score = st.slider("Previous Exam Score", 0, 100, 60)
        sleep_hours = st.slider("Sleeping Hours (per day)", 4, 10, 7)
        study_breaks = st.slider("Study Breaks (per day)", 0, 5, 2)

    submit = st.form_submit_button("🔮 Predict Performance")

# Prediction logic
if submit:
    # Encode Gender
    gender_val = 1 if gender == "Male" else 0

    # ⚠️ SAME FEATURE ORDER AS MODEL TRAINING
    input_data = np.array([[
        gender_val,
        float(study_hours),
        float(attendance),
        float(previous_score),
        float(sleep_hours),
        float(study_breaks)
    ]])

    # Scale input
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)[0]

    # Output
    st.success(f"📊 Predicted Final Score: **{round(prediction, 2)}**")

    if prediction >= 75:
        st.balloons()
        st.info("🌟 Excellent performance expected")
    elif prediction >= 50:
        st.info("✅ Average performance expected")
    else:
        st.warning("⚠️ Needs improvement")

# Footer
st.divider()
st.markdown(
    "<p style='text-align: center; font-size: 12px;'>Developed by Shruti Waghmare | B.Sc Data Science</p>",
    unsafe_allow_html=True
)
