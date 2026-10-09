"""Module 4 — Évaluation sur le jeu de test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

st.set_page_config(page_title="Évaluation", page_icon="📈", layout="wide")
st.title("📈 Module 4 — Évaluation")

if "fitted_models" not in st.session_state:
    st.warning("Entraînez d'abord les modèles (Module 3).")
    st.stop()

s = st.session_state
X_test, y_test = s["X_test"][s["num_cols"] + s["cat_cols"]], s["y_test"]
threshold = st.slider("Seuil de décision", 0.1, 0.9, 0.5, 0.05)

probas = {n: p.predict_proba(X_test)[:, 1] for n, p in s["fitted_models"].items()}
rows = []
for n, proba in probas.items():
    pred = (proba >= threshold).astype(int)
    rows.append({
        "Model": n,
        "AUC-ROC": roc_auc_score(y_test, proba),
        "PR-AUC": average_precision_score(y_test, proba),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
    })
res = pd.DataFrame(rows).set_index("Model").sort_values("AUC-ROC", ascending=False)
st.dataframe(res.style.format("{:.3f}"), use_container_width=True)
st.caption("Abaisser le seuil augmente le rappel (plus de churners détectés) au prix de plus de faux positifs.")

c1, c2 = st.columns(2)
roc, pr = go.Figure(), go.Figure()
for n, proba in probas.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    p, r, _ = precision_recall_curve(y_test, proba)
    roc.add_scatter(x=fpr, y=tpr, name=n, mode="lines")
    pr.add_scatter(x=r, y=p, name=n, mode="lines")
roc.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), showlegend=False)
roc.update_layout(title="Courbes ROC", xaxis_title="FPR", yaxis_title="TPR")
pr.update_layout(title="Courbes Précision/Rappel", xaxis_title="Rappel", yaxis_title="Précision")
c1.plotly_chart(roc, use_container_width=True)
c2.plotly_chart(pr, use_container_width=True)

st.subheader("Matrice de confusion")
options = list(probas)
default = s.get("best_model_name")
model = st.selectbox("Modèle", options, index=options.index(default) if default in options else 0)
cm = confusion_matrix(y_test, (probas[model] >= threshold).astype(int))
st.plotly_chart(
    px.imshow(
        cm, text_auto=True,
        x=["Prédit : reste", "Prédit : churn"], y=["Réel : reste", "Réel : churn"],
        color_continuous_scale="Blues",
    ),
    use_container_width=True,
)

st.subheader("Importance des variables")
pipe = s["fitted_models"][model]
clf, names = pipe.named_steps["clf"], list(pipe.named_steps["prep"].get_feature_names_out())
if hasattr(clf, "coef_"):
    imp = pd.Series(clf.coef_[0], index=names)
elif hasattr(clf, "feature_importances_"):
    imp = pd.Series(clf.feature_importances_, index=names)
else:
    imp = None
if imp is None:
    st.info("Pas d'importance native pour ce modèle (kNN).")
else:
    top = imp.reindex(imp.abs().sort_values(ascending=False).index[:15])[::-1]
    st.plotly_chart(px.bar(x=top.values, y=top.index, orientation="h", labels={"x": "", "y": ""}), use_container_width=True)
