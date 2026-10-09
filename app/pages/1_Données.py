"""Module 1 — Chargement et exploration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from data.load import TARGET, clean, feature_columns, load_clean

st.set_page_config(page_title="Données", page_icon="📊", layout="wide")
st.title("📊 Module 1 — Données")


@st.cache_data(show_spinner="Téléchargement OpenML…")
def get_data():
    return load_clean()


source = st.radio("Source", ["OpenML (auto)", "CSV uploadé"], horizontal=True)
if source == "OpenML (auto)":
    df = get_data()
else:
    up = st.file_uploader("Fichier CSV Telco", type="csv")
    if up is None:
        st.info("Chargez un CSV au format Telco Churn.")
        st.stop()
    df = clean(pd.read_csv(up))

st.session_state["df"] = df
num_cols, cat_cols = feature_columns(df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Observations", f"{len(df):,}".replace(",", " "))
m2.metric("Variables", df.shape[1] - 1)
m3.metric("Taux de churn", f"{df[TARGET].mean():.1%}")
m4.metric("Valeurs manquantes", int(df.isna().sum().sum()))

st.subheader("Aperçu")
st.dataframe(df.head(), use_container_width=True)
st.dataframe(
    pd.DataFrame({"colonne": df.columns, "type": df.dtypes.astype(str).values}),
    use_container_width=True,
)

left, right = st.columns(2)
with left:
    st.subheader("Valeurs manquantes")
    miss = df.isna().sum()
    st.plotly_chart(px.bar(x=miss.index, y=miss.values, labels={"x": "", "y": "NaN"}), use_container_width=True)
with right:
    st.subheader("Équilibre des classes")
    counts = df[TARGET].map({0: "Reste", 1: "Résilie"}).value_counts()
    st.plotly_chart(px.pie(names=counts.index, values=counts.values, hole=0.4), use_container_width=True)

st.subheader("Distributions numériques par Churn")
cols = st.columns(len(num_cols))
for col, c in zip(cols, num_cols):
    col.plotly_chart(
        px.histogram(df, x=c, color=df[TARGET].astype(str), barmode="overlay", opacity=0.7),
        use_container_width=True,
    )

st.subheader("Taux de churn par variable catégorielle")
var = st.selectbox("Variable", cat_cols)
rate = df.groupby(var)[TARGET].mean().sort_values(ascending=False).reset_index()
st.plotly_chart(px.bar(rate, x=var, y=TARGET, labels={TARGET: "Taux de churn"}), use_container_width=True)

st.subheader("Corrélation (numériques + cible)")
corr = df[num_cols + [TARGET]].corr()
st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)
