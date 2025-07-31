import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np

# Set page config
st.set_page_config(page_title="Smart Health Diagnosis Bot", page_icon="🧠", layout="centered")

# Load the trained model and encoders
model = joblib.load("health_diagnosis_model.pkl")
mlb = joblib.load("symptom_binarizer.pkl")
le = joblib.load("label_encoder.pkl")

# Load treatment data
treatment_df = pd.read_csv("treatment_data.csv")  # Should have 'Disease' and 'Treatment' columns

# CSS Styling
st.markdown("""
    <style>
        .main {
            background: linear-gradient(145deg, #e6f0ff, #ffffff);
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
        }
        .stDownloadButton>button {
            background-color: #0073e6;
            color: white;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Get treatment by disease name
def get_treatment_for(disease_name):
    disease_name = disease_name.lower()
    match = treatment_df[treatment_df['Disease'].str.lower() == disease_name]
    if not match.empty:
        return match['Treatment'].values[0]
    return "No specific treatment found. Please consult a doctor."

# Save prediction logs
def save_prediction(user_info, symptoms, prediction):
    log = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Name": user_info["name"],
        "Age": user_info["age"],
        "Gender": user_info["gender"],
        "Phone": user_info["phone"],
        "Symptoms": ', '.join(symptoms),
        "Predicted Disease": prediction
    }
    df = pd.DataFrame([log])
    with open("history.csv", "a") as f:
        df.to_csv(f, header=f.tell() == 0, index=False)

# App title
st.title("🧠 Smart Health Diagnosis Bot")

# Sidebar menu
menu = st.sidebar.radio("📌 Navigation", ["User Diagnosis", "Admin Panel"])

if menu == "User Diagnosis":
    st.markdown("## 🧑‍⚕️ Enter Your Details")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    phone = st.text_input("Phone Number")

    st.markdown("## 🤒 Select Your Symptoms")
    all_symptoms = mlb.classes_
    selected_symptoms = st.multiselect("Choose symptoms from the list", all_symptoms)

    if st.button("🔍 Diagnose"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            input_vector = mlb.transform([selected_symptoms])

            try:
                # Try top 3 predictions
                proba = model.predict_proba(input_vector)[0]
                top_indices = proba.argsort()[-3:][::-1]
                top_diseases = le.inverse_transform(top_indices)
                top_probs = proba[top_indices]

                predicted = top_diseases[0]
                st.success(f"🩺 Most likely disease: **{predicted}**")

                st.markdown("### 🔍 Other possible predictions:")
                for i in range(1, 3):
                    st.info(f"{i+1}. {top_diseases[i]} (Confidence: {top_probs[i]:.2f})")

            except Exception as e:
                # Fallback to just predict
                predicted = model.predict(input_vector)[0]
                predicted = le.inverse_transform([predicted])[0]
                st.success(f"🩺 Predicted Disease: **{predicted}**")
                st.info("Top 3 predictions not available due to model compatibility.")

            # Treatment suggestion
            treatment = get_treatment_for(predicted)
            st.markdown("## 💊 Suggested Treatment")
            st.write(treatment)

            # Save user data
            user_info = {
                "name": name,
                "age": age,
                "gender": gender,
                "phone": phone
            }
            save_prediction(user_info, selected_symptoms, predicted)

elif menu == "Admin Panel":
    st.markdown("## 🔐 Admin Login")
    password = st.text_input("Enter Admin Password", type="password")

    if password == "admin123":
        st.success("Access Granted ✅")
        try:
            history = pd.read_csv("history.csv")
            st.dataframe(history)
            csv = history.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download Diagnosis History", csv, "user_diagnosis_history.csv", "text/csv")
        except FileNotFoundError:
            st.warning("No history data available.")
    else:
        if password:
            st.warning("Incorrect Password ❌")
