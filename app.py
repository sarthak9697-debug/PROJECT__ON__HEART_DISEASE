import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# HEART DISEASE PREDICTION — STREAMLIT APP
# =========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

SAMPLE_PATIENTS = {
    "Sample 1 — Typical higher-risk profile": {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    },
    "Sample 2 — Typical lower-risk profile": {
        "age": 37, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250,
        "fbs": 0, "restecg": 1, "thalach": 187, "exang": 0,
        "oldpeak": 3.5, "slope": 0, "ca": 0, "thal": 2
    },
    "Sample 3 — No-disease example": {
        "age": 61, "sex": 0, "cp": 0, "trestbps": 130, "chol": 330,
        "fbs": 0, "restecg": 0, "thalach": 169, "exang": 0,
        "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2
    },
    "Sample 4 — No-disease example": {
        "age": 58, "sex": 1, "cp": 0, "trestbps": 150, "chol": 270,
        "fbs": 0, "restecg": 0, "thalach": 111, "exang": 1,
        "oldpeak": 0.8, "slope": 2, "ca": 0, "thal": 3
    },
    "Sample 5 — Disease example": {
        "age": 54, "sex": 1, "cp": 2, "trestbps": 120, "chol": 258,
        "fbs": 0, "restecg": 0, "thalach": 147, "exang": 0,
        "oldpeak": 0.4, "slope": 1, "ca": 0, "thal": 3
    },
    "Sample 6 — Disease example": {
        "age": 68, "sex": 1, "cp": 0, "trestbps": 144, "chol": 193,
        "fbs": 1, "restecg": 1, "thalach": 141, "exang": 0,
        "oldpeak": 3.4, "slope": 1, "ca": 2, "thal": 3
    }
}

MODEL_PATH = Path(__file__).parent / "model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.pkl was not found. Keep model.pkl in the same folder as app.py."
        )

    with open(MODEL_PATH, "rb") as model_file:
        saved_model = pickle.load(model_file)

    # Supports the direct model saved by the notebook.
    if hasattr(saved_model, "predict"):
        return saved_model

    # Also supports a dictionary bundle if model.pkl is later packaged that way.
    if isinstance(saved_model, dict) and "model" in saved_model:
        return saved_model["model"]

    raise ValueError("model.pkl does not contain a usable scikit-learn model.")


def make_input(values):
    return pd.DataFrame(
        [[values[feature] for feature in FEATURES]],
        columns=FEATURES
    )


# =========================================================
# HEADER
# =========================================================

st.title("❤️ Heart Disease Prediction")
st.write(
    "Machine Learning classification using Logistic Regression, "
    "Decision Tree and Random Forest models from the original project."
)

st.info(
    "Recruiter demo: select a built-in patient sample and click "
    "**Predict Heart Disease**. No external dataset is required."
)

st.warning(
    "Educational demonstration only — this prediction is not a medical diagnosis "
    "and should not be used for clinical decisions."
)


# =========================================================
# MODEL
# =========================================================

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load model.pkl: {error}")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("📌 Project Information")
    st.write("**Dataset:** Heart Disease")
    st.write("**Best model:** Random Forest")
    st.write("**Training split:** 70% / 30%")
    st.write("**Target:** 0 = No Disease, 1 = Disease")

    st.markdown("---")

    st.subheader("Feature Guide")
    st.caption("The controls below use sensible ranges for this dataset.")

    st.write("**CP:** 0–3 chest-pain category")
    st.write("**RestECG:** 0–2")
    st.write("**Slope:** 0–2")
    st.write("**CA:** 0–4")
    st.write("**Thal:** 0–3")


# =========================================================
# SAMPLE / MANUAL MODE
# =========================================================

st.subheader("🧪 Test the Model")

mode = st.radio(
    "Input method",
    ["Built-in sample patient", "Manual patient input"],
    horizontal=True
)

if mode == "Built-in sample patient":

    selected_sample = st.selectbox(
        "Choose a built-in sample",
        list(SAMPLE_PATIENTS.keys())
    )

    values = SAMPLE_PATIENTS[selected_sample]

    st.caption(
        "These sample values are embedded in app.py, so the recruiter "
        "does not need a separate CSV or local notebook."
    )

else:

    st.caption(
        "Enter a patient profile using the ranges shown below. "
        "Values are model inputs, not medical recommendations."
    )

    values = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        values["age"] = st.slider("Age", 18, 100, 55)
        values["sex"] = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
        )
        values["cp"] = st.selectbox("Chest Pain (CP)", [0, 1, 2, 3], index=1)
        values["trestbps"] = st.slider("Resting Blood Pressure", 80, 220, 130)

    with col2:
        values["chol"] = st.slider("Cholesterol", 100, 600, 240)
        values["fbs"] = st.selectbox(
            "Fasting Blood Sugar > 120",
            [0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
        )
        values["restecg"] = st.selectbox("Resting ECG", [0, 1, 2], index=1)
        values["thalach"] = st.slider("Maximum Heart Rate", 60, 220, 150)
        values["exang"] = st.selectbox(
            "Exercise-Induced Angina",
            [0, 1],
            format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
        )

    with col3:
        values["oldpeak"] = st.slider(
            "Oldpeak", 0.0, 7.0, 1.0, step=0.1
        )
        values["slope"] = st.selectbox("Slope", [0, 1, 2], index=1)
        values["ca"] = st.selectbox("Major Vessels (CA)", [0, 1, 2, 3, 4])
        values["thal"] = st.selectbox("Thal", [0, 1, 2, 3], index=2)


# =========================================================
# DISPLAY INPUT
# =========================================================

st.markdown("---")
st.subheader("📋 Patient Input")

input_df = make_input(values)

display_df = input_df.T.rename(columns={0: "Value"})
st.dataframe(display_df, use_container_width=True)


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🔍 Predict Heart Disease",
    type="primary",
    use_container_width=True
):

    try:

        prediction = int(model.predict(input_df)[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            no_disease_probability = float(probabilities[0])
            disease_probability = float(probabilities[1])
        else:
            no_disease_probability = None
            disease_probability = None

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.error("Prediction: **Disease**")
        else:
            st.success("Prediction: **No Disease**")

        if disease_probability is not None:

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric(
                    "No Disease Probability",
                    f"{no_disease_probability * 100:.2f}%"
                )

            with metric2:
                st.metric(
                    "Disease Probability",
                    f"{disease_probability * 100:.2f}%"
                )

            st.subheader("Prediction Probabilities")

            probability_df = pd.DataFrame(
                {
                    "Class": ["No Disease", "Disease"],
                    "Probability": [
                        no_disease_probability,
                        disease_probability
                    ]
                }
            ).set_index("Class")

            st.bar_chart(probability_df)

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# =========================================================
# EXPLANATION
# =========================================================

with st.expander("📖 How this project works"):

    st.markdown(
        """
**1. Data Analysis**

The original notebook performs dataset inspection, descriptive statistics
and correlation analysis.

**2. Model Training**

The original notebook trains:

- Logistic Regression
- Decision Tree
- Random Forest

**3. Model Selection**

The original notebook compares accuracy, precision, recall, F1-score and
confusion matrices and selects the model with the highest accuracy.

**4. Deployment**

The selected model is saved as `model.pkl` and loaded by this Streamlit app.

**5. Recruiter Testing**

Built-in patient profiles make it possible to test the application
immediately without locating the original dataset.
"""
    )

st.caption(
    "Heart Disease Prediction — Streamlit Portfolio Demonstration"
)
