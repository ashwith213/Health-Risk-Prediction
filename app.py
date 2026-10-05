import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Health Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>
.stApp { background: #f5f7fb; }
.block-container { max-width: 1400px; padding-top: 0.8rem; padding-bottom: 3rem; }

/* Sticky header */
.sticky-header {
    position: sticky;
    top: 0;
    z-index: 9999;
    background: linear-gradient(135deg, #102a43 0%, #176b87 55%, #1f8a9e 100%);
    padding: 22px 30px;
    margin: -10px -10px 25px -10px;
    border-radius: 0 0 20px 20px;
    box-shadow: 0 7px 22px rgba(16,42,67,0.20);
}
.sticky-title {
    color: #ffffff !important;
    font-size: 38px !important;
    font-weight: 800 !important;
    line-height: 1.15 !important;
    margin: 0 !important;
}
.sticky-subtitle {
    color: #eaf8fb !important;
    font-size: 17px !important;
    line-height: 1.5 !important;
    margin: 7px 0 0 0 !important;
}
.badge {
    display: inline-block;
    color: white !important;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 999px;
    padding: 5px 11px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 9px;
}

/* Section headings */
.section-box {
    background: #ffffff;
    padding: 16px 20px;
    border-left: 6px solid #176b87;
    border-radius: 12px;
    margin: 27px 0 15px 0;
    box-shadow: 0 4px 12px rgba(16,42,67,0.08);
}
.section-title {
    color: #102a43 !important;
    font-size: 25px !important;
    font-weight: 800 !important;
    line-height: 1.25 !important;
    margin: 0 !important;
}
.section-subtitle {
    color: #486581 !important;
    font-size: 14px !important;
    margin: 5px 0 0 0 !important;
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #d9e2ec;
    border-radius: 17px;
    padding: 20px 18px;
    min-height: 112px;
    box-shadow: 0 5px 18px rgba(16,42,67,0.07);
}
.metric-value {
    color: #102a43 !important;
    font-size: 29px !important;
    font-weight: 800 !important;
    margin: 0 !important;
}
.metric-label {
    color: #627d98 !important;
    font-size: 12px !important;
    font-weight: 800 !important;
    margin-top: 7px !important;
}

/* Inputs */
.input-card {
    background: #ffffff;
    border: 1px solid #d9e2ec;
    border-radius: 17px;
    padding: 17px 20px 5px 20px;
    margin-bottom: 14px;
    box-shadow: 0 5px 18px rgba(16,42,67,0.06);
}
.input-heading {
    color: #102a43 !important;
    font-size: 17px !important;
    font-weight: 800 !important;
    margin: 0 !important;
}
.input-help {
    color: #627d98 !important;
    font-size: 13px !important;
    margin: 4px 0 8px 0 !important;
}
label, .stSelectbox label, .stNumberInput label, .stSlider label {
    color: #243b53 !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

/* Light input boxes with dark, highly visible values */
.stNumberInput input {
    color: #102a43 !important;
    -webkit-text-fill-color: #102a43 !important;
    background-color: #ffffff !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    caret-color: #102a43 !important;
    border: 2px solid #bcccdc !important;
    border-radius: 10px !important;
}

.stNumberInput input::placeholder {
    color: #829ab1 !important;
    -webkit-text-fill-color: #829ab1 !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    color: #102a43 !important;
    background-color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    border: 2px solid #bcccdc !important;
    border-radius: 10px !important;
}

.stSelectbox div[data-baseweb="select"] span {
    color: #102a43 !important;
}

.stSelectbox svg {
    fill: #486581 !important;
}

.stNumberInput button {
    color: #102a43 !important;
    background-color: #f0f4f8 !important;
}

.stNumberInput button svg {
    fill: #102a43 !important;
}

/* Predict button */
.stButton > button {
    width: 100%;
    min-height: 54px;
    border-radius: 13px;
    border: 0;
    background: linear-gradient(90deg, #0f6b8a, #178ca3);
    color: #ffffff !important;
    font-size: 17px !important;
    font-weight: 800 !important;
    box-shadow: 0 7px 18px rgba(15,107,138,0.25);
}
.stButton > button:hover {
    background: linear-gradient(90deg, #0b5d78, #117d91);
    color: #ffffff !important;
}

/* Result */
.result-card {
    background: linear-gradient(135deg, #102a43, #176b87);
    border-radius: 20px;
    padding: 25px;
    margin-top: 18px;
    box-shadow: 0 10px 28px rgba(16,42,67,0.18);
}
.result-title {
    color: #ffffff !important;
    font-size: 25px !important;
    font-weight: 800 !important;
    margin: 0 !important;
}
.result-text {
    color: #eaf8fb !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
    margin-top: 6px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: #102a43; }
section[data-testid="stSidebar"] * { color: #ffffff !important; }
.sidebar-title { color: #ffffff !important; font-size: 22px !important; font-weight: 800 !important; }
.sidebar-text { color: #d9eaf2 !important; font-size: 14px !important; line-height: 1.55 !important; }

.footer {
    text-align: center;
    color: #627d98 !important;
    font-size: 12px !important;
    padding: 28px 0 5px 0;
}

@media (max-width: 900px) {
    .sticky-title { font-size: 29px !important; }
    .sticky-subtitle { font-size: 14px !important; }
    .section-title { font-size: 22px !important; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="sticky-header">
    <div class="badge">🤖 SUPERVISED MACHINE LEARNING • HEALTH ANALYTICS</div>
    <div class="sticky-title">🩺 Health Risk Prediction</div>
    <div class="sticky-subtitle">
        AI-powered health classification using health, lifestyle and clinical indicators.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD DATA AND TRAIN MODEL
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("novagen_dataset.csv")

@st.cache_resource
def train_model(df):
    data = df.copy()
    target = "Target"

    X = data.drop(columns=[target])
    y = data[target]

    # Convert categorical columns into numeric columns
    X = pd.get_dummies(X, drop_first=False)
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest WITHOUT max_depth restriction
    model = RandomForestClassifier(
        n_estimators=301,
        min_samples_split=5,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    return model, scaler, feature_columns, accuracy, len(X_test)

try:
    df = load_data()
    model, scaler, feature_columns, accuracy, test_samples = train_model(df)
except Exception as e:
    st.error("Unable to load the model or dataset.")
    st.info("Make sure novagen_dataset.csv is in the same GitHub repository as app.py.")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🩺 Health Risk AI</div>', unsafe_allow_html=True)
    st.markdown(
        f'''<div class="sidebar-text">
        <b>Purpose</b><br>
        Classify health outcomes using health, lifestyle and medical-history indicators.
        <br><br>
        <b>Model</b><br>
        Random Forest Classifier<br>
        301 trees • no max-depth restriction
        <br><br>
        <b>Dataset</b><br>
        {len(df):,} records
        <br><br>
        <b>Important</b><br>
        This application is an educational machine-learning demonstration and is not a medical diagnosis.
        </div>''',
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------
st.markdown("""
<div class="section-box">
    <div class="section-title">📊 Model Performance</div>
    <div class="section-subtitle">Random Forest performance on the held-out test dataset.</div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(f'''<div class="metric-card">
        <div class="metric-value">{accuracy*100:.2f}%</div>
        <div class="metric-label">TEST ACCURACY</div>
    </div>''', unsafe_allow_html=True)

with m2:
    st.markdown('''<div class="metric-card">
        <div class="metric-value">301</div>
        <div class="metric-label">RANDOM FOREST TREES</div>
    </div>''', unsafe_allow_html=True)

with m3:
    st.markdown(f'''<div class="metric-card">
        <div class="metric-value">{test_samples:,}</div>
        <div class="metric-label">TEST SAMPLES</div>
    </div>''', unsafe_allow_html=True)

# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------
st.markdown("""
<div class="section-box">
    <div class="section-title">🧑‍⚕️ Enter Health Information</div>
    <div class="section-subtitle">Enter the individual's health and lifestyle information below.</div>
</div>
""", unsafe_allow_html=True)

# Basic information
st.markdown('''<div class="input-card">
    <div class="input-heading">👤 Basic Information</div>
    <div class="input-help">Age, BMI and lifestyle-related information</div>
</div>''', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
with c2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5, step=0.1)
with c3:
    smoking = st.selectbox("Smoking", ["No", "Yes"])

c1, c2, c3 = st.columns(3)
with c1:
    alcohol = st.selectbox("Alcohol", ["No", "Yes"])
with c2:
    diet = st.selectbox("Diet", ["No", "Yes"])
with c3:
    mental_health = st.selectbox("Mental Health", ["No", "Yes"])

# Clinical indicators
st.markdown('''<div class="input-card">
    <div class="input-heading">❤️ Clinical Indicators</div>
    <div class="input-help">Enter basic physiological measurements</div>
</div>''', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=250, value=120)
with c2:
    cholesterol = st.number_input("Cholesterol", min_value=50, max_value=400, value=180)
with c3:
    glucose = st.number_input("Glucose Level", min_value=40, max_value=400, value=100)

c1, c2, c3 = st.columns(3)
with c1:
    heart_rate = st.number_input("Heart Rate", min_value=30, max_value=220, value=70)
with c2:
    sleep_hours = st.slider("Sleep Hours", 0.0, 14.0, 7.0, 0.5)
with c3:
    exercise_hours = st.slider("Exercise Hours", 0.0, 10.0, 2.0, 0.5)

# Lifestyle and history
st.markdown('''<div class="input-card">
    <div class="input-heading">🌿 Lifestyle & Medical History</div>
    <div class="input-help">Select the options that best describe the individual</div>
</div>''', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    water_intake = st.slider("Water Intake", 0.0, 200.0, 100.0, 5.0)
with c2:
    stress_level = st.slider("Stress Level", 0, 10, 5)
with c3:
    physical_activity = st.selectbox("Physical Activity", ["No", "Yes"])

c1, c2, c3 = st.columns(3)
with c1:
    medical_history = st.selectbox("Medical History", ["No", "Yes"])
with c2:
    allergies = st.selectbox("Allergies", ["No", "Yes"])
with c3:
    diet_type = st.selectbox("Diet Type", ["Other", "Vegan", "Vegetarian"])

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮  Predict Health Risk", use_container_width=True):
    input_data = pd.DataFrame([{
        "Age": age,
        "BMI": bmi,
        "Blood_Pressure": blood_pressure,
        "Cholesterol": cholesterol,
        "Glucose_Level": glucose,
        "Heart_Rate": heart_rate,
        "Sleep_Hours": sleep_hours,
        "Exercise_Hours": exercise_hours,
        "Water_Intake": water_intake,
        "Stress_Level": stress_level,
        "Smoking": smoking,
        "Alcohol": alcohol,
        "Diet": diet,
        "MentalHealth": mental_health,
        "PhysicalActivity": physical_activity,
        "MedicalHistory": medical_history,
        "Allergies": allergies,
        "Diet_Type": diet_type,
    }])

    input_encoded = pd.get_dummies(input_data, drop_first=False)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = float(np.max(probabilities)) * 100
    else:
        probabilities = None
        confidence = 0

    # Convert raw model classes into user-friendly health labels.
    if str(prediction) == "0":
        result_title = "🟢 Healthy"
        result_text = "The model classified this individual as Healthy."
        result_class = "Healthy"
    else:
        result_title = "🔴 Unhealthy"
        result_text = "The model classified this individual as Unhealthy."
        result_class = "Unhealthy"

    st.markdown(f'''<div class="result-card">
        <div class="result-title">{result_title}</div>
        <div class="result-text">{result_text}<br>
        Result: <b>{result_class}</b><br>
        Model confidence: <b>{confidence:.1f}%</b></div>
    </div>''', unsafe_allow_html=True)

    if probabilities is not None:
        st.markdown("""
        <div class="section-box">
            <div class="section-title">📈 Prediction Confidence</div>
            <div class="section-subtitle">Probability assigned by the Random Forest model.</div>
        </div>
        """, unsafe_allow_html=True)

        labels = ["Healthy" if str(c) == "0" else "Unhealthy" for c in model.classes_]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(labels, probabilities * 100)
        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Model Prediction Probability")
        ax.grid(axis="y", alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("""
<div class="footer">
    <b>Health Risk Prediction</b> • Supervised Machine Learning Project<br>
    Built with Python • Pandas • Scikit-learn • Streamlit<br><br>
    ⚠️ Educational demonstration only — not a medical diagnosis.
</div>
""", unsafe_allow_html=True)