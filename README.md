# Diabetes Prediction System 🩺

A Machine Learning-based web application that predicts whether a patient is likely to have diabetes using a **Decision Tree Classifier** trained on the **Pima Indians Diabetes Dataset**. The project includes model training, evaluation, visualization, and an interactive Streamlit web application for real-time predictions.

---

## 📌 Project Overview

The system performs the complete machine learning workflow, including data preprocessing, model training, evaluation, visualization, and deployment through a user-friendly Streamlit interface.

### The application can:

- Load and preprocess the diabetes dataset
- Handle missing and invalid values
- Train a Decision Tree Classification model
- Evaluate model performance
- Visualize the trained Decision Tree
- Save the trained model
- Predict diabetes from user-entered medical parameters
- Display prediction confidence scores

---

## 🚀 Features

- ✅ Data Cleaning & Preprocessing
- ✅ Decision Tree Classifier
- ✅ Model Evaluation
- ✅ Confusion Matrix Visualization
- ✅ Decision Tree Visualization
- ✅ Model Serialization using Joblib
- ✅ Interactive Streamlit Web App
- ✅ Real-Time Prediction

---

## 📊 Dataset

**Pima Indians Diabetes Dataset**

### Features

| Feature | Description |
|----------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-Hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age of the patient |
| Outcome | 0 = Non-Diabetic, 1 = Diabetic |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

---

## 📁 Project Structure

```text
Diabetes-Prediction-System/
│
├── diabetes.csv
├── train_model.py
├── app.py
│
├── diabetes_model.pkl
├── feature_names.pkl
│
├── confusion_matrix.png
├── decision_tree.png
├── evaluation_results.txt
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Machine Learning Workflow

### 1. Data Preprocessing

- Load the dataset
- Replace invalid zero values with NaN
- Fill missing values using median imputation

### 2. Model Training

- Train-Test Split (80:20)
- Decision Tree Classifier
- Gini Index
- Maximum Depth = 4
- Minimum Samples per Leaf = 5

### 3. Model Evaluation

Performance is evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report

---

## 📈 Generated Outputs

Running the training script automatically creates:

- `diabetes_model.pkl`
- `feature_names.pkl`
- `evaluation_results.txt`
- `confusion_matrix.png`
- `decision_tree.png`

---

## 🌐 Streamlit Web Application

The web application allows users to enter patient medical details and instantly predict whether the patient is likely to have diabetes.

### User Inputs

- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin Level
- BMI
- Diabetes Pedigree Function
- Age

### Prediction Output

The application displays:

- Diabetes Prediction
- Confidence Score
- Prediction Probabilities



## ▶️ Train the Model

```bash
python train_model.py
```

---

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🖥️ Sample Workflow

1. Run the training script.
2. Launch the Streamlit application.
3. Enter patient medical details.
4. Click **Predict**.
5. View the prediction and confidence score.

---

## 🎯 Future Improvements

- Random Forest Classifier
- XGBoost Implementation
- Hyperparameter Optimization
- Model Comparison Dashboard
- Cloud Deployment
- Explainable AI (SHAP/LIME)
- Patient History Tracking

---

## 📚 Learning Outcomes

- Data preprocessing techniques
- Decision Tree Classification
- Model evaluation using classification metrics
- Machine learning model serialization
- Streamlit application development
- End-to-end machine learning deployment

---

## ⭐ If you found this project useful, consider giving it a Star on GitHub!
