import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle

# Load data
df = pd.read_csv("cleaned_student_performance.csv")

# Encode Gender
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

X = df[["Gender", "Hours_Studied", "Attendance",
        "Previous_Score", "Sleep_Hours", "Study_Breaks"]]
y = df["Exam_Score"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save model + scaler
with open("student_performance_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("✅ Model & scaler saved successfully")
