import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load model
with open("student_performance_model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

st.markdown("<h1 style='text-align:center;'>🎓 Student Performance Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter details to predict exam score</p>", unsafe_allow_html=True)
st.divider()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        study_hours = st.slider("Study Hours per Day", 0, 10, 3)
        attendance = st.slider("Attendance (%)", 50, 100, 75)

    with col2:
        previous_score = st.slider("Previous Exam Score", 0, 100, 60)
        study_breaks = st.slider("Study Breaks (per day)", 0, 5, 2)
        sleep_hours = st.slider("Sleep Hours", 4, 10, 7)

    submit = st.form_submit_button("🔮 Predict")

if submit:
    gender_val = 1 if gender == "Male" else 0

    input_data = np.array([[ 
        gender_val,
        study_hours,
        attendance,
        previous_score,
        sleep_hours,
        study_breaks
    ]], dtype=float)

    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]

    st.success(f"📊 Predicted Exam Score: **{prediction:.2f}**")

    if prediction >= 75:
        st.balloons()
        st.info("🌟 Excellent performance expected")
    elif prediction >= 50:
        st.info("✅ Average performance expected")
    else:
        st.warning("⚠️ Needs improvement")
