
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

st.title("🚢 Titanic Survival Prediction")
st.write("Find out the estimated survival probability of a Titanic passenger.")

st.divider()

st.subheader("Enter Passenger Details")

pclass = st.selectbox(
    "Ticket Class",
    [1, 2, 3]
)

sex = st.radio(
    "Gender",
    ["female", "male"],
    horizontal=True
)

age = st.slider(
    "Passenger Age",
    1, 80, 30
)

fare = st.slider(
    "Ticket Fare",
    0.0, 513.0, 32.0
)

if st.button("Predict Survival", type="primary"):

    if not os.path.exists("model.joblib"):
        st.error("Model file not found. Please upload model.joblib.")
    else:
        model = load_model()

        passenger = pd.DataFrame(
            [[pclass, sex, age, fare]],
            columns=["pclass", "sex", "age", "fare"]
        )

        probability = model.predict_proba(passenger)[0][1]
        prediction = model.predict(passenger)[0]

        st.subheader("Prediction Result")

        st.metric(
            "Estimated Survival Probability",
            f"{probability:.1%}"
        )

        if prediction == 1:
            st.success("The model predicts that this passenger may have survived!")
        else:
            st.warning("The model predicts that this passenger may not have survived.")

        st.progress(float(probability))

st.divider()

st.caption(
    "Created by Meesha Sinha | Educational Machine Learning Project. "
    "Predictions are estimates based on historical Titanic passenger data."
)
