"""
Diabetes Prediction System using Decision Tree Classifier
Verveox Technologies AI & ML Internship - Week 3 Task
-------------------------------------------------------
This script loads the Pima Indians Diabetes Dataset, preprocesses it,
trains a Decision Tree Classifier, evaluates its performance, visualizes
the tree, and saves the trained model for later use in the Streamlit app.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import joblib

# ---------------------------------------------------------
# STEP 1: DATASET PREPARATION
# ---------------------------------------------------------
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv("diabetes.csv", names=columns)

print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset shape:", df.shape)

# ---------------------------------------------------------
# STEP 2: DATA PREPROCESSING
# ---------------------------------------------------------
print("\nMissing values (NaN) per column:")
print(df.isnull().sum())

# In this dataset, missing data is encoded as 0 in medical columns
# where a 0 is not physiologically possible.
zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("\nCount of invalid zero entries per column:")
print((df[zero_invalid_cols] == 0).sum())

# Replace invalid zeros with NaN, then impute using median (robust to outliers)
df[zero_invalid_cols] = df[zero_invalid_cols].replace(0, np.nan)
for col in zero_invalid_cols:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing/invalid values after cleaning:")
print(df[zero_invalid_cols].isnull().sum())

# Separate features (X) and target (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ---------------------------------------------------------
# STEP 3: MODEL TRAINING
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# STEP 4: MODEL EVALUATION
# ---------------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Non-Diabetic", "Diabetic"])

print(f"\nAccuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# Save evaluation results to a text file (used for the report)
with open("evaluation_results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# Plot & save confusion matrix
fig1, ax1 = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Diabetic", "Diabetic"])
disp.plot(ax=ax1, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix - Decision Tree")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close(fig1)

# Visualize and save the Decision Tree
fig2, ax2 = plt.subplots(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Non-Diabetic", "Diabetic"],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax2
)
plt.title("Decision Tree Visualization - Diabetes Prediction")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
plt.close(fig2)

# ---------------------------------------------------------
# STEP 5: SAVE THE MODEL
# ---------------------------------------------------------
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")
print("\nModel and feature list saved successfully (diabetes_model.pkl, feature_names.pkl)")
print("Plots saved: confusion_matrix.png, decision_tree.png")
