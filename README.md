# Titanic Survival Prediction – Streamlit App

Created by: Meesha Sinha

## Project overview
An educational machine-learning web app that estimates Titanic passenger survival probability from ticket class, sex, age and fare.

## 1. Train the model
Run these commands in a terminal from this folder:

```bash
pip install -r requirements.txt
python train_model.py
```

The training script downloads the public Titanic dataset, prepares the four input features, trains and evaluates a Decision Tree pipeline, and saves `model.joblib`.

## 2. Run the app locally

```bash
streamlit run app.py
```

## 3. Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `model.joblib`, and `requirements.txt`.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose the repository, branch, and `app.py` as the main file.
4. Deploy and test the public app link.

## Important
Keep the model feature names and order identical to those in the training script: `pclass`, `sex`, `age`, `fare`. The app uses the saved preprocessing-and-model pipeline, so categorical encoding and missing-value handling are applied consistently.
