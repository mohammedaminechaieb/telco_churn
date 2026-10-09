"""Module 2 — Préparation : ColumnTransformer + split stratifié."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from data.load import feature_columns
from data.preprocessing import build_preprocessor, get_feature_names, split

st.set_page_config(page_title="Préparation", page_icon="🔧", layout="wide")
st.title("🔧 Module 2 — Préparation")

if "df" not in st.session_state:
    st.warning("Chargez d'abord les données (Module 1).")
    st.stop()

df = st.session_state["df"]
num_cols, cat_cols = feature_columns(df)

test_size = st.slider("Taille du jeu de test", 0.1, 0.4, 0.2, 0.05)
seed = st.number_input("random_state", value=42, step=1)

if st.button("Préparer", type="primary"):
    X_train, X_test, y_train, y_test = split(df, test_size=test_size, random_state=int(seed))
    prep = build_preprocessor(num_cols, cat_cols)
    X_train_t = prep.fit_transform(X_train)
    X_test_t = prep.transform(X_test)
    st.session_state.update(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        X_train_t=X_train_t, X_test_t=X_test_t, preprocessor=prep,
        feat_names=get_feature_names(prep), num_cols=num_cols, cat_cols=cat_cols,
    )

if "X_train" not in st.session_state:
    st.stop()

s = st.session_state
a, b, c, d = st.columns(4)
a.metric("Train", len(s["X_train"]))
b.metric("Test", len(s["X_test"]))
c.metric("Churn train", f"{s['y_train'].mean():.2%}")
d.metric("Churn test", f"{s['y_test'].mean():.2%}")
st.caption("Split stratifié : le ratio de churn est conservé dans train et test.")

st.write(
    f"**Features après encodage :** {s['X_train_t'].shape[1]} "
    f"({len(num_cols)} numériques standardisées + {len(cat_cols)} catégorielles one-hot)"
)
with st.expander("Noms des features"):
    st.write(s["feat_names"])
