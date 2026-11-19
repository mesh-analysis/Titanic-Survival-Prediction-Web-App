import os
import joblib
import pandas as pd
import streamlit as st

FEATURES = ["pclass", "sex", "age", "fare"]
MODEL_PATH = "model.joblib"

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to see the model's estimated survival probability.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

with st.sidebar:
    st.header("Passenger details")
    pclass = st.selectbox("Ticket class", [1, 2, 3], index=2)
    sex = st.radio("Sex", ["female", "male"], horizontal=True)
    age = st.slider("Age", min_value=1, max_value=80, value=30)
    fare = st.slider("Fare paid", min_value=0.0, max_value=513.0, value=32.0, step=1.0)

if st.button("Predict survival", type="primary"):
    if not os.path.exists(MODEL_PATH):
        st.error("The model file is missing. Please run train_model.py first to create model.joblib.")
    else:
        try:
            model = load_model()
            row = pd.DataFrame([[pclass, sex, age, fare]], columns=FEATURES)
            probability = float(model.predict_proba(row)[0][1])
            prediction = int(model.predict(row)[0])

            st.subheader("Prediction")
            st.metric("Estimated chance of survival", f"{probability:.1%}")
            if prediction == 1:
                st.success("The model predicts survival.")
            else:
                st.warning("The model predicts non-survival.")
            st.progress(probability)
        except Exception as exc:
            st.error(f"Sorry, the prediction could not be generated: {exc}")

st.caption(
    "Educational example trained on a historical Titanic passenger dataset. "
    "Predictions are estimates from a machine learning model, not historical certainty or advice."
)
