import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

# Load dataset
df = pd.read_csv("smart_health_dataset.csv")

# Rename 'Diagnosis' column to 'Disease' for consistency
df.rename(columns={"Diagnosis": "Disease"}, inplace=True)

# Remove rows with missing symptoms
df = df.dropna(subset=["Symptom_1", "Symptom_2", "Symptom_3"])

# Remove 'Healthy' cases to avoid defaulting prediction
df = df[df["Disease"].str.lower() != "healthy"]

# Add custom disease samples
custom_data = pd.DataFrame([
    {
        "Symptom_1": "chest_pain",
        "Symptom_2": "shortness_of_breath",
        "Symptom_3": "sweating",
        "Disease": "Heart Attack"
    },
    {
        "Symptom_1": "abdominal_pain",
        "Symptom_2": "nausea",
        "Symptom_3": "vomiting",
        "Disease": "Gastritis"
    },
    {
        "Symptom_1": "fatigue",
        "Symptom_2": "swelling_legs",
        "Symptom_3": "reduced_urine_output",
        "Disease": "Kidney Failure"
    }
])
df = pd.concat([df, custom_data], ignore_index=True)

# Combine symptoms into a list
df["Symptoms"] = df[["Symptom_1", "Symptom_2", "Symptom_3"]].values.tolist()

# Encode symptoms using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["Symptoms"])

# Encode target disease labels
le = LabelEncoder()
y = le.fit_transform(df["Disease"])

# Train the model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "health_diagnosis_model.pkl")
joblib.dump(mlb, "symptom_binarizer.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Model training complete. All files saved.")
