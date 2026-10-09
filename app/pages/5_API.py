"""Module 5 — Test de l'API FastAPI."""
import requests
import streamlit as st

st.set_page_config(page_title="API", page_icon="⚡", layout="wide")
st.title("⚡ Module 5 — API FastAPI")

try:
    default_url = st.secrets.get("API_URL", "http://localhost:8000")
except Exception:
    default_url = "http://localhost:8000"
API_URL = st.text_input("URL de l'API", default_url).rstrip("/")

if st.button("Vérifier /health"):
    try:
        st.json(requests.get(f"{API_URL}/health", timeout=60).json())
    except Exception as e:
        st.error(f"API injoignable : {e}")

try:
    models = requests.get(f"{API_URL}/models", timeout=5).json().get("models", [])
except Exception:
    models = []
model = st.selectbox("Modèle", models or ["(API indisponible)"])
threshold = st.slider("Seuil", 0.0, 1.0, 0.5, 0.05)

YN = ["No", "Yes", "No internet service"]
st.subheader("Profil client")
a, b, c = st.columns(3)
f = {
    "gender": a.selectbox("gender", ["Female", "Male"]),
    "SeniorCitizen": a.selectbox("SeniorCitizen", ["0", "1"]),
    "Partner": a.selectbox("Partner", ["Yes", "No"]),
    "Dependents": a.selectbox("Dependents", ["No", "Yes"]),
    "tenure": a.number_input("tenure (mois)", 0, 100, 12),
    "PhoneService": b.selectbox("PhoneService", ["Yes", "No"]),
    "MultipleLines": b.selectbox("MultipleLines", ["No", "Yes", "No phone service"]),
    "InternetService": b.selectbox("InternetService", ["Fiber optic", "DSL", "No"]),
    "OnlineSecurity": b.selectbox("OnlineSecurity", YN),
    "OnlineBackup": b.selectbox("OnlineBackup", YN),
    "DeviceProtection": b.selectbox("DeviceProtection", YN),
    "TechSupport": c.selectbox("TechSupport", YN),
    "StreamingTV": c.selectbox("StreamingTV", YN),
    "StreamingMovies": c.selectbox("StreamingMovies", YN),
    "Contract": c.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
    "PaperlessBilling": c.selectbox("PaperlessBilling", ["Yes", "No"]),
    "PaymentMethod": c.selectbox(
        "PaymentMethod",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    ),
    "MonthlyCharges": c.number_input("MonthlyCharges", 0.0, 200.0, 65.0),
}
f["TotalCharges"] = st.number_input("TotalCharges", 0.0, 10000.0, float(f["tenure"] * f["MonthlyCharges"]))

if st.button("Prédire", type="primary"):
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"features": f, "model": model, "threshold": threshold},
            timeout=60,
        )
        if resp.status_code == 200:
            d = resp.json()
            x, y = st.columns(2)
            x.metric("Probabilité de churn", f"{d['probability']:.1%}")
            y.metric("Décision", d["decision"])
        else:
            st.error(f"{resp.status_code} : {resp.text}")
    except Exception as e:
        st.error(f"API injoignable : {e}")
