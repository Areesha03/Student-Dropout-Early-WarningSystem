import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Student Dropout Early Warning System",
    layout="wide"
)

st.title("🎓 Student Dropout Early Warning System")
st.write("Early identification of students at risk of dropping out using engagement data.")

# -----------------------------
# Load trained model safely
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("dropout_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# -----------------------------
# Features used in training
# -----------------------------
FEATURES = [
    'gender','StageID','GradeID','Topic','Semester',
    'raisedhands','VisITedResources',
    'AnnouncementsView','Discussion',
    'StudentAbsenceDays','Relation'
]

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload student CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # clean column names

    # Debug: show columns
    st.write("Columns in uploaded file:", df.columns.tolist())

    # Check required columns
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.stop()

    # -----------------------------
    # Prediction
    # -----------------------------
    X = df[FEATURES]
    try:
        risk_scores = model.predict_proba(X)[:,1]
        st.success("✅ Risk scores calculated successfully!")
    except Exception as e:
        st.error(f"Error calculating risk scores: {e}")
        st.stop()

    df['risk_score'] = risk_scores

    # Assign risk labels
    def risk_label(score):
        if score >= 0.7:
            return "High"
        elif score >= 0.4:
            return "Medium"
        else:
            return "Low"

    df['risk_label'] = df['risk_score'].apply(risk_label)
    df['predicted_dropout'] = (df['risk_score'] >= 0.5).astype(int)

    # -----------------------------
    # Show top 20 risky students
    # -----------------------------
    st.subheader("🚨 Top 20 High-Risk Students")
    st.dataframe(
        df.sort_values("risk_score", ascending=False).head(20),
        use_container_width=True
    )

    # -----------------------------
    # Individual student view
    # -----------------------------
    st.subheader("🔍 Individual Student Risk Details")
    student_idx = st.selectbox("Select Student Index", df.index)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk Score", round(df.loc[student_idx, "risk_score"], 3))
    with col2:
        st.metric("Risk Level", df.loc[student_idx, "risk_label"])

    # -----------------------------
    # Simple explanation
    # -----------------------------
    st.subheader("📌 Why this student may be at risk")
    st.markdown("""
    **Key early warning signals include:**
    - Low class participation (few raised hands)
    - Low use of learning resources
    - High number of absences
    - Limited discussion engagement
    """)

    # -----------------------------
    # Download CSV
    # -----------------------------
    st.subheader("⬇️ Download Predictions")
    download_df = df[['risk_score','risk_label','predicted_dropout']].copy()
    csv = download_df.to_csv(index=False)
    st.download_button(
        label="Download Risk Predictions CSV",
        data=csv,
        file_name="student_risk_predictions.csv",
        mime="text/csv"
    )
