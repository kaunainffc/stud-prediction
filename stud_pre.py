import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==========================
# STEP 1: TRAIN + SAVE MODEL
# ==========================
def train_and_save_model():
    # Dummy dataset
    data = pd.DataFrame({
        "Hours Studied": [5, 7, 8, 4, 9],
        "Previous Scores": [70, 85, 90, 60, 95],
        "Extracurricular Activities": ['Yes', 'No', 'Yes', 'No', 'Yes'],
        "Sleep Hours": [7, 6, 8, 5, 8],
        "Sample Question Papers Practiced": [4, 3, 5, 2, 6],
        "Performance": [75, 80, 95, 65, 98]
    })

    le = LabelEncoder()
    data['Extracurricular Activities'] = le.fit_transform(data['Extracurricular Activities'])

    X = data.drop(columns=["Performance"])
    y = data["Performance"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    # Save the model, scaler, and label encoder
    with open("linear_regression_model.pkl", "wb") as f:
        pickle.dump((model, scaler, le), f)

# ====================================
# STEP 2: LOAD MODEL WITH FILE CHECKING
# ====================================
def load_model():
    if not os.path.exists("linear_regression_model.pkl"):
        train_and_save_model()

    with open("linear_regression_model.pkl", "rb") as file:
        try:
            model, scaler, le = pickle.load(file)
        except EOFError:
            # If file exists but is corrupted/empty, recreate
            train_and_save_model()
            with open("linear_regression_model.pkl", "rb") as file2:
                model, scaler, le = pickle.load(file2)
    return model, scaler, le

# ======================
# STEP 3: PREDICTION LOGIC
# ======================
def preprocess_input_data(data, scaler, le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model, scaler, le = load_model()
    processed_data = preprocess_input_data(data, scaler, le)
    prediction = model.predict(processed_data)
    return prediction[0]

# ==================
# STEP 4: STREAMLIT UI
# ==================
def main():
    st.title("📚 Student Performance Predictor")
    st.write("Enter details below to predict performance:")

    hours_studied = st.number_input("🕐 Hours Studied", 1, 10, 5)
    previous_score = st.number_input("📄 Previous Score (%)", 40, 100, 70)
    extracurricular = st.selectbox("🏅 Extracurricular Activities", ["Yes", "No"])
    sleep_hours = st.number_input("💤 Sleep Hours", 4, 10, 7)
    papers_solved = st.number_input("🧠 Papers Solved", 0, 10, 5)

    if st.button("🔮 Predict Performance"):
        user_data = {
            "Hours Studied": hours_studied,
            "Previous Scores": previous_score,
            "Extracurricular Activities": extracurricular,
            "Sleep Hours": sleep_hours,
            "Sample Question Papers Practiced": papers_solved
        }
        try:
            prediction = predict_data(user_data)
            st.success(f"📈 Predicted Score: {round(prediction, 2)}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ================
# RUN THE APP
# ================
if __name__ == "__main__":
    # Ensure model is trained and saved
    if not os.path.exists("linear_regression_model.pkl"):
        train_and_save_model()
    main()
