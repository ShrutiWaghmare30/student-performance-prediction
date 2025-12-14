import pandas as pd 
from sklearn.preprocessing import StandardScaler


# Load cleaned dataset
df=pd.read_csv("cleaned_student_performance.csv")

#Encode ctegorical column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Gender"]=le.fit_transform(df["Gender"])

#Feature Slection(x) and (y)
X = df[["Gender", "Hours_Studied", "Attendance",
        "Previous_Score", "Sleep_Hours", "Study_Breaks"]]

y = df["Exam_Score"]


#Spliting dataset into training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Using Liner Regression on taining dataset
from sklearn.linear_model import LinearRegression

model= LinearRegression()
model.fit(X_train_scaled, y_train)

#model trained Successfully

#Predict using Testing Dataset
y_pred = model.predict(X_test_scaled)


# Model Evaluation
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np 

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)
print("R2 Score:",r2)

#Actual vs Predicted
import matplotlib.pyplot as plt

plt.scatter(y_test,y_pred)
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Actual vs Predicted Exam Scores")
plt.savefig("Actual_vs_predicted")
plt.show()

#Percentage of accuracy 
import numpy as np

# Convert to numpy arrays
actual = np.array(y_test)
predicted = np.array(y_pred)

# Percentage accuracy for each prediction
percentage_accuracy = (1 - np.abs(actual - predicted) / actual) * 100

# Average accuracy
average_accuracy = np.mean(percentage_accuracy)

print("Average Prediction Accuracy (%):", average_accuracy)

#save file in .pkl
import pickle

with open("student_performance_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)


print("Model trained and saved successfully.")

