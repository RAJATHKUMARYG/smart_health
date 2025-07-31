import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Load model and encoders
model = joblib.load("health_diagnosis_model.pkl")
binarizer = joblib.load("symptom_binarizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load treatment data
treatment_df = pd.read_csv("treatment_data.csv")

# Clean white background with black text
st.markdown("""
    <style>
        body, .stApp {
            background-color: white;
            color: black;
        }
        h1, h2, h3, h4, h5, h6, p, label, div, input, select, textarea {
            color: black !important;
        }
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div {
            background-color: #f0f0f0;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# Page title
st.title("🧠 Smart Health Diagnosis Bot")

# Session state for user data
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_data = {}

# User Login Form
if not st.session_state.logged_in:
    with st.form("user_form"):
        st.subheader("🔐 User Login")
        name = st.text_input("Name")
        age = st.text_input("Age")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        phone = st.text_input("Phone Number")
        submit = st.form_submit_button("Login")

        if submit:
            if name and age and phone:
                st.session_state.logged_in = True
                st.session_state.user_data = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "phone": phone,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.success("Logged in successfully!")
            else:
                st.error("Please fill all details.")
    st.stop()

# Disease Prediction Section
st.subheader("💡 Select Your Symptoms")
all_symptoms = binarizer.classes_
selected_symptoms = st.multiselect("Choose symptoms from the list below:", sorted(all_symptoms))

if st.button("🔍 Predict Disease"):
    input_vector = binarizer.transform([selected_symptoms])
    input_vector = input_vector.reshape(1, -1)

    if input_vector.sum() == 0:
        st.error("Please select at least one symptom.")
    else:
        prediction = model.predict(input_vector)[0]
        predicted_disease = label_encoder.inverse_transform([prediction])[0]

        proba = model.predict_proba(input_vector)[0]
        top3_indices = proba.argsort()[-3:][::-1]
        top3_diseases = label_encoder.inverse_transform(top3_indices)

        st.success(f"Most likely diagnosis: **{predicted_disease}**")
        st.markdown("### 🩺 Other possible diseases:")
        for i, disease in enumerate(top3_diseases, start=1):
            st.write(f"{i}. {disease}")

        # Show treatment
        treatment_row = treatment_df[treatment_df['Disease'].str.lower() == predicted_disease.lower()]
        if not treatment_row.empty:
            treatment = treatment_row['Treatment'].values[0]
            st.markdown("### 💊 Suggested Treatment")
            st.info(treatment)
        else:
            st.warning("No treatment information available for this disease.")

        # Save to history log
        with open("prediction_logs.csv", "a") as f:
            f.write(f"{st.session_state.user_data['name']},{st.session_state.user_data['age']}," +
                    f"{st.session_state.user_data['gender']},{st.session_state.user_data['phone']}," +
                    f"{predicted_disease},{','.join(selected_symptoms)},{st.session_state.user_data['timestamp']}\n")

# Admin Panel
st.markdown("---")
st.subheader("🔒 Admin Panel")
admin_key = st.text_input("Enter Admin Access Code", type="password")

if admin_key == "admin123":  # Change this to a secure password
    st.success("Access granted!")
    if os.path.exists("prediction_logs.csv"):
        df_logs = pd.read_csv("prediction_logs.csv", header=None)
        df_logs.columns = ["Name", "Age", "Gender", "Phone", "Predicted Disease", "Symptoms", "Time"]
        st.dataframe(df_logs)
        st.download_button("Download Logs", df_logs.to_csv(index=False), "logs.csv")
    else:
        st.info("No history found.")
elif admin_key:
    st.error("Invalid admin key.")
