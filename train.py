"""Student Placement Predictor - run once: python train.py  ->  creates model.pkl"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

FEATURES = ["cgpa", "backlogs", "projects", "internships", "skills"]

# 1. Load data (real CSV if present, otherwise demo data for testing)
print(os.path.exists("../placement_dataset/placement.csv"))
if os.path.exists("../placement_dataset/placement.csv"):
    df = pd.read_csv("../placement_dataset/placement.csv")

X = df[FEATURES]
y = df["placed"]  # 1 = placed, 0 = not placed

# 2. Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Train: scale the features, then fit Logistic Regression
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)

# 4. Evaluate on the unseen 20%
pred = model.predict(X_test)
print("Accuracy :", round(accuracy_score(y_test, pred), 2))
print("Precision:", round(precision_score(y_test, pred), 2))
print("Recall   :", round(recall_score(y_test, pred), 2))

# 5. Save the model for the Flask app
joblib.dump(model, "model.pkl")
print("Saved model.pkl")

# 6. predict the for the particular person
student = pd.DataFrame([{
    "cgpa": 5.5, 
    "backlogs": 3, 
    "projects": 0, 
    "internships": 0, 
    "skills": 1
}])

prob = model.predict_proba(student)[0][1]
result = model.predict(student)[0]


if result == 1:
    print(f"Student have the chance of PLACED ({prob*100}% chance)")
else:
    print(f"Student have the chance of NOT PLACED ({prob*100}% chance of being placed)")