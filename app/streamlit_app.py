"""Page d'accueil Streamlit — Telco Churn (Cas 13)."""
import streamlit as st

st.set_page_config(page_title="Telco Churn", page_icon="📉", layout="wide")

st.title("📉 Telco Customer Churn")
st.caption("IBM Telco dataset · OpenML 42178 · Data Mining ENSI 2026-2027")

st.markdown(
    """
Application ML interactive couvrant toute la chaîne de prédiction du churn :

1. **Données** — chargement, exploration, statistiques
2. **Préparation** — ColumnTransformer, split stratifié 80/20
3. **Modélisation** — 5 modèles, GridSearchCV (cv=3, roc_auc)
4. **Évaluation** — AUC-ROC, PR-AUC, seuil ajustable, matrice de confusion
5. **API** — test de l'API FastAPI (`/health`, `/models`, `/predict`)

Utilisez le menu latéral pour suivre les modules **dans l'ordre**.
"""
)

c1, c2, c3 = st.columns(3)
c1.metric("Clients", "7 043")
c2.metric("Variables", "19")
c3.metric("Taux de churn", "26,5 %")
