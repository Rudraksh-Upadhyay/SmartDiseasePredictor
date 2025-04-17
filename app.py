import streamlit as st 
import joblib 
import numpy as np

model = joblib.load('disease_predictor_model.joblib')

# st.title("Smart Disease Predictor")
st.set_page_config(page_title="Smart Disease Predictor", page_icon="🩺", layout="centered")

#sidebar
st.sidebar.title("ABOUT")
st.sidebar.info(
    "This is a simple Machine Learning based Disease Predictor App.\n\n"
    "Select your symptoms and click Predict to see the result."
)
st.markdown("_____")
st.caption("Made With Love")

# st.write("Select your symptoms and get a prediction")
# st.title("Smart Disease Predictor")

st.markdown("Select your Symptoms:")

fever = st.checkbox("Fever")
cough = st.checkbox("Cough")
headache = st.checkbox("Headache")
fatigue = st.checkbox("Fatigue")

if st.button('Predict Disease'):
    symptoms = np.array([[int(fever), int(cough), int(headache), int(fatigue)]])
    
    prediction = model.predict(symptoms)
    
    st.success(f"You might have: {prediction[0]}")
    
# st.caption("Made with love using Streamlit + Machine Learning")