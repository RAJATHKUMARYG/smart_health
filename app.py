import streamlit as st
import pandas as pd
import joblib
import datetime

# Load model and encoders
model = joblib.load("health_diagnosis_model.pkl")
mlb = joblib.load("symptom_binarizer.pkl")
le = joblib.load("label_encoder.pkl")

# Load treatment data
treatment_df = pd.read_csv("treatment_data.csv")

# Custom Styling
# Styling
st.markdown("""
    <style>
    body {
        background-color: #007BFF;
    }
    .main {
        background-color: #007BFF;
        color: black;
    }
    .stApp {
        background-color: #007BFF;
    }
    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: black !important;
    }
    .block-container {
        padding: 2rem;
    }
    .box {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
    }
    button {
        background-color: white;
        color: #007BFF;
        font-weight: bold;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title with image
st.image("https://cdn.pixabay.com/photo/2016/03/31/19/14/stethoscope-1295046_960_720.png", width=100)
st.title("🤖 Smart Health Diagnosis Bot")

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["User Diagnosis", "Admin Panel"])

# Function: Get treatment
def get_treatment_for(disease_name):
    disease_name = disease_name.lower()
    match = treatment_df[treatment_df['Disease'].str.lower() == disease_name]
    if not match.empty:
        return match['Treatment'].values[0]
    return "No specific treatment found. Please consult a doctor."

# Function: Save history
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

# --- User Diagnosis ---
if menu == "User Diagnosis":
    with st.container():
        st.subheader("🧑‍⚕️ Enter Your Details")
        st.image("https://cdn.pixabay.com/photo/2017/08/06/22/01/checklist-2593755_960_720.png", width=200)
        st.markdown('<div class="box">', unsafe_allow_html=True)
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        phone = st.text_input("Phone Number")
        st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("🤒 Select Your Symptoms")
        st.image("https://cdn.pixabay.com/photo/2021/02/09/13/29/symptoms-5997855_960_720.jpg", use_column_width=True)
        st.markdown('<div class="box">', unsafe_allow_html=True)
        all_symptoms = mlb.classes_
        selected_symptoms = st.multiselect("Choose symptoms", all_symptoms)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🩺 Diagnose"):
            if not selected_symptoms:
                st.warning("Please select at least one symptom.")
            else:
                input_vector = mlb.transform([selected_symptoms])
                proba = model.predict_proba(input_vector)[0]
                top_indices = proba.argsort()[-3:][::-1]
                top_diseases = le.inverse_transform(top_indices)
                top_probs = proba[top_indices]

                predicted = top_diseases[0]
                st.success(f"🩺 Most likely disease: **{predicted}**")

                st.info("🔍 Other possible predictions:")
                for i in range(1, 3):
                    st.write(f"{i+1}. {top_diseases[i]} (Confidence: {top_probs[i]:.2f})")

                treatment = get_treatment_for(predicted)
                st.subheader("💊 Suggested Treatment")
                st.image("https://cdn.pixabay.com/photo/2020/03/11/20/37/medicine-4923084_960_720.jpg", use_column_width=True)
                st.write(treatment)

                user_info = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "phone": phone
                }
                save_prediction(user_info, selected_symptoms, predicted)

# --- Admin Panel ---
elif menu == "Admin Panel":
    st.subheader("🔐 Admin Login")
    st.markdown('<div class="box">', unsafe_allow_html=True)
    password = st.text_input("Enter Admin Password", type="password")
    st.markdown('</div>', unsafe_allow_html=True)

    if password == "admin123":
        st.success("Access Granted")
        try:
            history = pd.read_csv("history.csv")
            st.dataframe(history)
            csv = history.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download History", csv, "user_diagnosis_history.csv", "text/csv")
        except FileNotFoundError:
            st.warning("No history data found.")
    elif password:
        st.warning("Unauthorized access.")
