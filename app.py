import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Health Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS (FIXED ACCESSIBILITY & VISIBILITY)
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
}

/* Main title */
.main-title {
    font-size: 40px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 4px;
}

.subtitle {
    font-size: 16px;
    font-weight: 500;
    color: #0284c7;
    margin-bottom: 25px;
}

/* Form Section Headers */
h4 {
    color: #0369a1 !important;
    font-weight: 700 !important;
    font-size: 18px !important;
    margin-bottom: 12px !important;
}

/* Cards */
.card {
    background: #ffffff;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #cbd5e1;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 18px;
}

.card-title {
    font-size: 22px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 15px;
}

/* Prediction result */
.result-card {
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 20px;
}

.healthy {
    background: #f0fdf4;
    border: 2px solid #22c55e;
}

.healthy .result-title {
    color: #15803d;
}

.healthy .result-score {
    color: #166534;
}

.risk {
    background: #fef2f2;
    border: 2px solid #ef4444;
}

.risk .result-title {
    color: #b91c1c;
}

.risk .result-score {
    color: #991b1b;
}

.result-title {
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 8px;
}

.result-score {
    font-size: 18px;
    font-weight: 600;
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #cbd5e1;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}

.metric-number {
    font-size: 26px;
    font-weight: 800;
    color: #0369a1;
}

.metric-label {
    font-size: 13px;
    font-weight: 600;
    color: #475569;
}

/* Button */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 17px;
    font-weight: 700;
    background: linear-gradient(90deg, #0284c7, #0369a1);
    color: #ffffff;
    border: none;
    transition: 0.2s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(2, 132, 199, 0.3);
}

/* Sidebar Styling Fixes */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}

section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3, 
section[data-testid="stSidebar"] p, 
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #f1f5f9 !important;
}

/* Disclaimer */
.disclaimer {
    background: #fffbeb;
    border-left: 5px solid #f59e0b;
    padding: 15px;
    border-radius: 10px;
    color: #78350f;
    margin-top: 20px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("novagen_dataset.csv")


@st.cache_resource
def train_model():
    df = load_data().copy()

    # Remove rows with missing target
    df = df.dropna(subset=["Target"])

    X = df.drop(columns=["Target"])
    y = df["Target"]

    # Convert categorical columns to numeric
    X = pd.get_dummies(X, drop_first=False)

    # Save feature columns
    feature_columns = X.columns.tolist()

    # Fill missing values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest
    model = RandomForestClassifier(
        n_estimators=301,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Test accuracy
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    return model, scaler, feature_columns, accuracy


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🩺 Health Risk Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered health classification using Supervised Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🩺 Health AI")
    st.markdown("---")
    st.markdown("""
    ### About the Project
    This application uses a **Random Forest classifier**
    to predict a health-risk category from selected
    health and lifestyle indicators.
    """)
    st.markdown("---")
    st.markdown("### Model Configuration")
    st.markdown("""
    🌲 **Random Forest**  
    • **Trees:** 301  
    • **Max Depth:** 7  
    • **Min Split:** 5
    """)
    st.markdown("---")
    st.info(
        "This application is for educational purposes only "
        "and should not be used for medical diagnosis."
    )


# =========================================================
# LOAD MODEL
# =========================================================

try:
    model, scaler, feature_columns, model_accuracy = train_model()
except Exception as e:
    st.error(
        "Unable to load the model. "
        "Please make sure `novagen_dataset.csv` is present "
        "in the same folder as `app.py`."
    )
    st.stop()


# =========================================================
# MODEL METRICS
# =========================================================

st.markdown("### 📊 Model Performance")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{model_accuracy*100:.2f}%</div>
            <div class="metric-label">Test Accuracy</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-number">301</div>
            <div class="metric-label">Random Forest Trees</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-number">7</div>
            <div class="metric-label">Maximum Tree Depth</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m4:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-number">2,865</div>
            <div class="metric-label">Test Samples</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="card-title">🧑‍⚕️ Enter Health Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 👤 Basic Information")
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5, step=0.1)
    blood_pressure = st.number_input("Blood Pressure", min_value=70, max_value=220, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=80, max_value=400, value=180)
    glucose = st.number_input("Glucose Level", min_value=50, max_value=400, value=100)

with col2:
    st.markdown("#### ❤️ Health & Lifestyle")
    heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=72)
    sleep_hours = st.slider("Sleep Hours", min_value=0.0, max_value=15.0, value=7.0, step=0.5)
    exercise_hours = st.slider("Exercise Hours / Day", min_value=0.0, max_value=8.0, value=1.0, step=0.5)
    water_intake = st.slider("Water Intake (Litres)", min_value=0.0, max_value=8.0, value=2.0, step=0.1)
    stress_level = st.slider("Stress Level", min_value=1, max_value=10, value=5)

with col3:
    st.markdown("#### 🧠 Behaviour & History")
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    mental_health = st.slider("Mental Health Score", min_value=0, max_value=100, value=70)
    physical_activity = st.slider("Physical Activity Level", min_value=0, max_value=100, value=60)
    medical_history = st.selectbox("Medical History", ["No", "Yes"])
    allergies = st.selectbox("Known Allergies", ["No", "Yes"])


# =========================================================
# PREDICTION
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button("🔍 Predict Health Risk")

if predict_button:
    input_data = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Blood_Pressure": [blood_pressure],
        "Cholesterol": [cholesterol],
        "Glucose_Level": [glucose],
        "Heart_Rate": [heart_rate],
        "Sleep_Hours": [sleep_hours],
        "Exercise_Hours": [exercise_hours],
        "Water_Intake": [water_intake],
        "Stress_Level": [stress_level],
        "Smoking": [1 if smoking == "Yes" else 0],
        "Alcohol": [1 if alcohol == "Yes" else 0],
        "MentalHealth": [mental_health],
        "PhysicalActivity": [physical_activity],
        "MedicalHistory": [1 if medical_history == "Yes" else 0],
        "Allergies": [1 if allergies == "Yes" else 0],
    })

    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[feature_columns]
    input_data = input_data.fillna(0)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    max_probability = max(probabilities)

    if prediction == 0:
        st.markdown(
            f"""
            <div class="result-card healthy">
                <div class="result-title">🟢 Lower Health Risk Classification</div>
                <div class="result-score">Model confidence: {max_probability*100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="result-card risk">
                <div class="result-title">🔴 Higher Health Risk Classification</div>
                <div class="result-score">Model confidence: {max_probability*100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 📈 Prediction Probability")
    probability_df = pd.DataFrame(
        {
            "Class": [str(c) for c in model.classes_],
            "Probability": probabilities
        }
    )
    st.bar_chart(probability_df.set_index("Class"))

    st.markdown("### 📋 Input Summary")
    summary = pd.DataFrame({
        "Parameter": [
            "Age", "BMI", "Blood Pressure", "Cholesterol", "Glucose Level",
            "Heart Rate", "Sleep Hours", "Exercise Hours", "Water Intake",
            "Stress Level", "Smoking", "Alcohol", "Mental Health",
            "Physical Activity", "Medical History", "Allergies"
        ],
        "Value": [
            age, bmi, blood_pressure, cholesterol, glucose,
            heart_rate, sleep_hours, exercise_hours, water_intake,
            stress_level, smoking, alcohol, mental_health,
            physical_activity, medical_history, allergies
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)


# =========================================================
# DISCLAIMER & FOOTER
# =========================================================

st.markdown(
    """
    <div class="disclaimer">
    ⚠️ <b>Important:</b> This application is developed for educational
    and machine-learning demonstration purposes. The prediction is not
    a medical diagnosis and should not replace advice or evaluation
    from a qualified healthcare professional.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <br>
    <center>
        <small style="color: #64748b;">
        Built with Python • Scikit-learn • Streamlit<br>
        Health Risk Prediction ML Project
        </small>
    </center>
    """,
    unsafe_allow_html=True
)
