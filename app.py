import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
st.title("Breast Cancer Prediction System")
st.warning("Decision-support tool only. The final diagnosis must be made by a qualified doctor.")

tab1, tab2 = st.tabs(["Manual entry", "Upload CSV"])

with tab1:
    patient_id = st.text_input("Patient ID (anonymized)")
    cols = st.columns(3)
    values = []
    for i, f in enumerate(features):
        values.append(cols[i % 3].number_input(f, value=0.0, format="%.4f"))

    if st.button("Predict"):
        X = scaler.transform(pd.DataFrame([values], columns=features))
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]  # [malignant, benign]
        if pred == 1:
            st.success("Result: Benign")
        else:
            st.error("Result: Malignant. Recommend further clinical review.")
        st.write(f"Malignant probability: {prob[0]:.2%} | Benign probability: {prob[1]:.2%}")
        st.caption(f"Patient: {patient_id or 'N/A'} | {datetime.now():%Y-%m-%d %H:%M}")

with tab2:
    file = st.file_uploader("CSV with the 30 feature columns", type="csv")
    if file:
        df = pd.read_csv(file)
        missing = [c for c in features if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            X = scaler.transform(df[features])
            df["Prediction"] = np.where(model.predict(X) == 1, "Benign", "Malignant")
            st.dataframe(df)
            st.download_button("Download results", df.to_csv(index=False), "predictions.csv")

