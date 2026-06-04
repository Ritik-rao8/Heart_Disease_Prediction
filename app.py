from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀", layout="wide")
BASE_DIR = Path(__file__).resolve().parent

# A small stylesheet keeps the layout clean without changing the standard widgets.
st.markdown("""
<style>
.stApp { background: #f5f7f8; }
.block-container { max-width: 1180px; padding-top: 2.5rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.035em; }
h1 { color: #173c3b; font-size: clamp(2rem, 4vw, 2.8rem) !important; }
[data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: 16px;
}
[data-testid="stForm"] { background: #ffffff; border: 1px solid #e1e8e8; padding: 1.5rem; }
[data-testid="stMetricValue"] { font-weight: 650; color: #173c3b; }
[data-baseweb="tab-list"] { gap: 1.7rem; }
[data-baseweb="tab"] { padding-left: 0; padding-right: 0; }
[data-testid="stFormSubmitButton"] button { border-radius: 9px; min-height: 3rem; font-weight: 600; }
.eyebrow { color: #267b73; font-size: .75rem; font-weight: 700; letter-spacing: .14em; margin-bottom: .6rem; }
.intro { color: #627475; font-size: 1.05rem; max-width: 700px; line-height: 1.7; margin-bottom: 1.2rem; }
.section-label { color: #28766f; font-size: .73rem; font-weight: 700; letter-spacing: .1em; margin: .3rem 0 .5rem; }
.empty-result { padding: 1.5rem .25rem; text-align: center; }
.empty-icon { color: #247c72; font-size: 2.3rem; margin-bottom: .6rem; }
.empty-title { font-size: 1.1rem; font-weight: 600; color: #264847; }
.empty-text { color: #6b7e7f; font-size: .9rem; line-height: 1.7; margin-top: .6rem; }
.footer { border-top: 1px solid #dfe7e7; color: #728383; margin-top: 2rem; padding-top: 1rem; font-size: .8rem; }
@media (max-width: 640px) {
    .block-container { padding-top: 1.5rem; }
    [data-testid="stForm"] { padding: 1rem; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "heart_disease_model.joblib")


if not (BASE_DIR / "heart_disease_model.joblib").exists() or not (BASE_DIR / "model_results.json").exists():
    st.error("Model files are missing. Run all notebook cells, then restart the app.")
    st.stop()

model = load_model()
metadata = json.loads((BASE_DIR / "model_results.json").read_text())
ranges = metadata["numeric_ranges"]

st.markdown('<div class="eyebrow">HEART HEALTH · MACHINE LEARNING PROJECT</div>', unsafe_allow_html=True)
st.title("Heart Disease Prediction")
st.markdown('<div class="intro">Explore how clinical measurements translate into a machine learning prediction. Enter a record below to get started.</div>', unsafe_allow_html=True)

predict_tab, insights_tab = st.tabs(["Prediction", "Model insights"])

# Readable labels are converted to the original Cleveland dataset codes.
sex_options = {"Female": 0, "Male": 1}
yes_no = {"No": 0, "Yes": 1}
cp_options = {"Typical angina": 1, "Atypical angina": 2, "Non-anginal pain": 3, "Asymptomatic": 4}
ecg_options = {"Normal": 0, "ST-T abnormality": 1, "Left ventricular hypertrophy": 2}
slope_options = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
ca_options = {"Unknown": np.nan, "0": 0, "1": 1, "2": 2, "3": 3}
thal_options = {"Unknown": np.nan, "Normal": 3, "Fixed defect": 6, "Reversible defect": 7}

with predict_tab:
    st.caption("Educational use only. This model is not a medical diagnosis tool.")
    form_column, result_column = st.columns([1.9, 1], gap="large")

    with form_column:
        with st.form("prediction_form"):
            st.subheader("Clinical measurements")
            st.caption("Numerical fields contain example values. Replace them with your record's measurements.")
            st.markdown('<div class="section-label">01 / BASIC INFORMATION</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            age = a.number_input("Age", min_value=18, max_value=120,
                                 value=int(ranges["age"]["median"]), help="Age in years.", key="age")
            sex = b.selectbox("Recorded sex", list(sex_options), key="sex")
            cp = a.selectbox("Chest pain category", list(cp_options), key="cp")
            fbs = b.selectbox("Fasting blood sugar > 120 mg/dL", list(yes_no), key="fbs")

            st.divider()
            st.markdown('<div class="section-label">02 / HEART & BLOOD MEASUREMENTS</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            trestbps = a.number_input("Resting blood pressure · mm Hg", min_value=50, max_value=300,
                                      value=int(ranges["trestbps"]["median"]), key="trestbps")
            chol = b.number_input("Serum cholesterol · mg/dL", min_value=50, max_value=800,
                                  value=int(ranges["chol"]["median"]), key="chol")
            thalach = a.number_input("Maximum heart rate · beats/min", min_value=30, max_value=250,
                                     value=int(ranges["thalach"]["median"]), key="thalach")
            restecg = b.selectbox("Resting ECG result", list(ecg_options), key="restecg")

            st.divider()
            st.markdown('<div class="section-label">03 / EXERCISE & IMAGING RESULTS</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            exang = a.selectbox("Exercise-induced angina", list(yes_no), key="exang")
            oldpeak = b.number_input("ST depression · oldpeak", min_value=0.0, max_value=10.0,
                                     value=float(ranges["oldpeak"]["median"]), step=0.1, key="oldpeak")
            slope = a.selectbox("Peak exercise ST slope", list(slope_options), key="slope")
            ca = b.selectbox("Major vessels colored · 0–3", list(ca_options), key="ca",
                             help="Number of vessels colored by fluoroscopy. Choose Unknown if unavailable.")
            thal = a.selectbox("Thallium scan result", list(thal_options), key="thal")
            st.caption("Unknown vessel and scan results use values learned from the training data.")
            submitted = st.form_submit_button("Generate prediction", type="primary", width="stretch")

        if submitted:
            values = {
                "age": age, "sex": sex_options[sex], "cp": cp_options[cp],
                "trestbps": trestbps, "chol": chol, "fbs": yes_no[fbs],
                "restecg": ecg_options[restecg], "thalach": thalach,
                "exang": yes_no[exang], "oldpeak": oldpeak, "slope": slope_options[slope],
                "ca": ca_options[ca], "thal": thal_options[thal]
            }
            input_data = pd.DataFrame([values], columns=metadata["features"])
            st.session_state["result"] = {
                "prediction": int(model.predict(input_data)[0]),
                "score": float(model.predict_proba(input_data)[0, 1]),
                "inputs": input_data,
                "outside": [column for column, limits in ranges.items()
                            if pd.notna(values[column]) and
                            not limits["min"] <= values[column] <= limits["max"]]
            }

    with result_column:
        with st.container(border=True):
            st.subheader("Prediction result")
            if "result" not in st.session_state:
                st.markdown('''<div class="empty-result">
                    <div class="empty-icon">♡</div>
                    <div class="empty-title">Ready when you are</div>
                    <div class="empty-text">Complete the measurements and select<br><b>Generate prediction</b> to see the result.</div>
                    </div>''', unsafe_allow_html=True)
            else:
                result = st.session_state["result"]
                st.caption("Result for the last submitted measurements")
                if result["prediction"] == 1:
                    st.warning("Predicted label: disease present · Class 1")
                else:
                    st.info("Predicted label: disease absent · Class 0")
                st.metric("Model score for disease class", f"{result['score']:.1%}")
                st.progress(result["score"])
                st.caption("An uncalibrated model score, not your personal medical risk or diagnostic confidence.")
                if result["outside"]:
                    st.warning("Outside observed training ranges: " + ", ".join(result["outside"]) +
                               ". Predictions may be unreliable for these measurements.")
                if result["inputs"][["ca", "thal"]].isna().any().any():
                    st.caption("Unknown clinical fields were filled by the saved training-data imputer.")
                with st.expander("Review submitted inputs"):
                    st.dataframe(result["inputs"].T.rename(columns={0: "Value"}), width="stretch")
            st.divider()
            st.caption("A class-0 result does not rule out disease. Clinical decisions require a qualified healthcare professional.")

        st.markdown("#### How it works")
        st.write("**1. Enter measurements**  \nUse recorded clinical and test results.")
        st.write("**2. Generate a prediction**  \nThe saved model processes the inputs.")
        st.write("**3. Review the output**  \nExplore the label and model score.")
        st.caption("Some fields require exercise testing or imaging. This is not a symptom checker.")

with insights_tab:
    st.subheader("A closer look at the model")
    st.caption("Performance on a small held-out test set, not an estimate of clinical reliability.")
    selected = next(row for row in metadata["test_results"] if row["Model"] == metadata["selected_model"])
    a, b, c = st.columns(3)
    a.metric("Test accuracy", f"{selected['Accuracy']:.1%}")
    b.metric("ROC-AUC", f"{selected['ROC-AUC']:.3f}")
    c.metric("Disease recall", f"{selected['Recall (1)']:.1%}")
    st.write("**Selected model:** " + metadata["selected_model"])
    st.caption(f"Selected by training cross-validation ROC-AUC · {metadata['train_rows']} training records · {metadata['test_rows']} test records")
    with st.container(border=True):
        st.markdown("#### Model comparison")
        st.dataframe(pd.DataFrame(metadata["test_results"]).round(3), hide_index=True, width="stretch")
    with st.container(border=True):
        st.markdown("#### What the Random Forest uses")
        st.caption("Importance describes predictive influence in the Random Forest, not medical causation.")
        importance = pd.DataFrame(metadata["random_forest_importance"])
        feature_labels = {"thal": "Thallium scan", "cp": "Chest pain category", "thalach": "Maximum heart rate",
                          "ca": "Vessel count", "oldpeak": "ST depression", "age": "Age", "chol": "Cholesterol",
                          "exang": "Exercise-induced angina", "trestbps": "Resting blood pressure",
                          "slope": "ST slope", "sex": "Recorded sex", "restecg": "Resting ECG", "fbs": "Fasting blood sugar"}
        importance["Feature"] = importance["Feature"].replace(feature_labels)
        st.bar_chart(importance.set_index("Feature")["Importance"], horizontal=True, color="#237b70", sort=True)
    st.caption("Historical Cleveland dataset; no external clinical validation. Assumed target mapping: 0 = absent, 1 = present.")
    st.markdown("[View dataset definitions ↗](https://archive.ics.uci.edu/dataset/45/heart+disease)")

st.markdown('<div class="footer">Heart Disease Prediction · Student ML project · Cleveland dataset</div>', unsafe_allow_html=True)
