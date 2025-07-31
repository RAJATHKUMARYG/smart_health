import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import pyttsx3
import speech_recognition as sr

# Load model and encoders
model = joblib.load("health_diagnosis_model.pkl")
mlb = joblib.load("symptom_binarizer.pkl")
le = joblib.load("label_encoder.pkl")

# Load treatment data
TREATMENT_FILE = "treatment_data.csv"
if os.path.exists(TREATMENT_FILE):
    treatment_df = pd.read_csv(TREATMENT_FILE)
else:
    treatment_df = pd.DataFrame(columns=["Disease", "Treatment"])

HISTORY_FILE = "user_history.csv"
if os.path.exists(HISTORY_FILE):
    user_history = pd.read_csv(HISTORY_FILE).to_dict('records')
else:
    user_history = []

def save_history(data):
    pd.DataFrame(data).to_csv(HISTORY_FILE, index=False)

def get_treatment_for(disease):
    treatment = treatment_df[treatment_df['Disease'].str.lower() == disease.lower()]
    if not treatment.empty:
        return treatment.iloc[0]['Treatment']
    return "No treatment suggestion available."

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen_symptoms():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        st.success(f"You said: {text}")
        return [sym.strip().capitalize() for sym in text.split(',')]
    except sr.UnknownValueError:
        st.error("Sorry, could not understand audio.")
        return []
    except sr.RequestError:
        st.error("Speech service error.")
        return []

# Page styling
st.markdown(
    """
    <style>
    body {
        background-color: #2196F3;
        color: white;
    }
    .stApp {
        background-color: #2196F3;
    }
    div[data-testid="stText"] label, div[data-testid="stTextArea"] label, .stButton>button {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("https://cdn.pixabay.com/photo/2017/03/02/12/28/health-2110734_960_720.jpg", use_column_width=True)

st.title("🤖 Smart Health Diagnosis Bot")

st.sidebar.header("🧑 User Login")
name = st.sidebar.text_input("Name")
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
age = st.sidebar.number_input("Age", min_value=1, max_value=120)
phone = st.sidebar.text_input("Phone Number")
user_details = None

if name and gender and age and phone:
    user_details = {
        "name": name,
        "gender": gender,
        "age": age,
        "phone": phone
    }
    st.sidebar.success("✅ Logged in successfully")
else:
    st.sidebar.warning("Please fill in all login fields")

# Main UI
st.subheader("🔍 Enter your symptoms (comma separated):")
symptom_input = st.text_input("E.g. headache, fever, cough")
symptoms = [sym.strip().capitalize() for sym in symptom_input.split(',') if sym.strip()]

if st.button("🎙️ Speak Symptoms"):
    symptoms = listen_symptoms()

if st.button("🧠 Predict Disease"):
    if symptoms:
        try:
            input_vector = mlb.transform([symptoms])
            prediction = model.predict(input_vector)[0]
            proba = model.predict_proba(input_vector)[0]

            top_indices = proba.argsort()[-3:][::-1]
            top_diseases = le.inverse_transform(top_indices)
            top_probs = proba[top_indices]

            if user_details:
                history_entry = {
                    "Name": user_details["name"],
                    "Gender": user_details["gender"],
                    "Age": user_details["age"],
                    "Phone": user_details["phone"],
                    "Symptoms": ", ".join(symptoms),
                    "Prediction": prediction,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                user_history.append(history_entry)
                save_history(user_history)

            st.success(f"🩺 Predicted Disease: **{prediction}**")
            speak_text(f"You might be suffering from {prediction}")

            st.subheader("📋 Top 3 Predictions")
            for i, (disease, prob) in enumerate(zip(top_diseases, top_probs)):
                st.write(f"**{i+1}. {disease}** — Confidence: `{prob*100:.2f}%`")

            st.subheader("💊 Suggested Treatment")
            st.info(get_treatment_for(prediction))

        except Exception as e:
            st.error("Prediction failed. Check model or input format.")
            st.exception(e)
    else:
        st.warning("Please enter or speak at least one symptom.")

# Admin section to view history
st.sidebar.subheader("🔒 Admin Access")
admin_password = st.sidebar.text_input("Enter Admin Password", type="password")
if admin_password == "admin123":
    st.sidebar.success("Admin authenticated")
    st.subheader("📜 User Prediction History")
    if user_history:
        st.dataframe(pd.DataFrame(user_history))
        if st.download_button("📥 Download History (CSV)", pd.DataFrame(user_history).to_csv(index=False), "user_history.csv"):
            st.success("Downloaded successfully.")
    else:
        st.info("No user history found.")
else:
    st.sidebar.info("Admin can view user history after login.")
