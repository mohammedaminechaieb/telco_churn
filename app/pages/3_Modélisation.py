"""Module 3 — Entraînement des 5 modèles avec GridSearchCV."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from data.training import MODEL_NAMES, train_all

st.set_page_config(page_title="Modélisation", page_icon="🧠", layout="wide")
st.title("🧠 Module 3 — Modélisation")

if "X_train" not in st.session_state:
    st.warning("Préparez d'abord les données (Module 2).")
    st.stop()

s = st.session_state
names = st.multiselect("Modèles", MODEL_NAMES, default=MODEL_NAMES)
save = st.checkbox("Sauvegarder dans models/*.pkl", value=True)

if st.button("Entraîner", type="primary") and names:
    bar = st.progress(0.0)
    status = st.empty()

    def progress(i, name):
        bar.progress(i / len(names))
        status.write(f"✓ {name}")

    fitted, best = train_all(
        s["X_train"], s["y_train"], s["num_cols"], s["cat_cols"],
        names=names, save=save, progress=progress,
    )
    s["fitted_models"] = fitted
    s["best_params"] = best
    s["metrics_df"] = pd.DataFrame(
        {n: {"CV AUC-ROC": b["cv_auc"], "Meilleurs paramètres": str(b["params"])} for n, b in best.items()}
    ).T
    s["best_model_name"] = max(best, key=lambda n: best[n]["cv_auc"])

if "metrics_df" in s:
    st.subheader("Résultats de la validation croisée")
    st.dataframe(s["metrics_df"], use_container_width=True)
    st.success(f"Meilleur modèle (CV AUC) : **{s['best_model_name']}**")
