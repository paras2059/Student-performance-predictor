import streamlit as st
import pickle
import numpy as np

# Load model and scaler
import joblib
model = joblib.load("model.pkl")

scaler = joblib.load("scaler.pkl")

st.title("🎓 Student Performance Predictor")
st.write("Predict whether a student will perform at a Low, Medium, or High level based on demographic and educational factors.")

# Input features
st.header("📥 Enter Student Information")

# Ordered inputs
parent_edu = st.selectbox(
    "Parental Level of Education",
    ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
)

lunch = st.radio("Lunch Type", ["standard", "free/reduced"])
prep = st.radio("Test Preparation Course", ["none", "completed"])
gender = st.radio("Gender", ["female", "male"])

race = st.selectbox(
    "Race/Ethnicity",
    ["group A", "group B", "group C", "group D", "group E"]
)

# Marks input (will be converted to average and scaled)
math = st.slider("Math Score", 0, 100, 70)
reading = st.slider("Reading Score", 0, 100, 70)
writing = st.slider("Writing Score", 0, 100, 70)

# Encode categorical values
education_order = {
    "some high school": 0,
    "high school": 1,
    "some college": 2,
    "associate's degree": 3,
    "bachelor's degree": 4,
    "master's degree": 5
}

prep_order = {"none": 0, "completed": 1}
lunch_order = {"free/reduced": 0, "standard": 1}

# One-hot encoding for race and gender
race_features = {
    'race/ethnicity_group B': 0,
    'race/ethnicity_group C': 0,
    'race/ethnicity_group D': 0,
    'race/ethnicity_group E': 0
}

if race in ["group B", "group C", "group D", "group E"]:
    race_features[f"race/ethnicity_{race}"] = 1  # set correct one to 1

gender_male = 1 if gender == "male" else 0

# Average score and scaling
avg_score = np.mean([math, reading, writing])
avg_score_scaled = scaler.transform([[avg_score]])[0][0]

# Build feature vector
input_vector = [
    education_order[parent_edu],
    lunch_order[lunch],
    prep_order[prep],
    gender_male,
    race_features['race/ethnicity_group B'],
    race_features['race/ethnicity_group C'],
    race_features['race/ethnicity_group D'],
    race_features['race/ethnicity_group E'],
    avg_score_scaled
]

# Predict
if st.button("🔍 Predict Performance"):
    prediction = model.predict([input_vector])[0]
    probs = model.predict_proba([input_vector])[0]

    st.subheader(f"🎯 Predicted Performance Level: **{prediction.upper()}**")
    st.write("Confidence:")
    st.progress(int(probs.max() * 100))

    st.write("🔢 Probability distribution:")
    st.json(dict(zip(model.classes_, np.round(probs, 2))))
