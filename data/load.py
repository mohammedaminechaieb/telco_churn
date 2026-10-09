"""Chargement et nettoyage du dataset Telco Churn (OpenML 42178)."""
import pandas as pd

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie un DataFrame brut Telco (OpenML ou CSV uploadé)."""
    df = df.copy()
    df = df.drop(columns=["customerID"], errors="ignore")

    # TotalCharges contient des chaînes vides -> NaN -> 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    for c in ["tenure", "MonthlyCharges"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Cible : str -> int
    if df[TARGET].dtype == object or str(df[TARGET].dtype) == "category":
        df[TARGET] = df[TARGET].astype(str).str.strip().map({"Yes": 1, "No": 0})
    df[TARGET] = df[TARGET].astype(int)

    # Toutes les autres colonnes sont traitées comme catégorielles (str)
    for c in df.columns:
        if c not in NUM_COLS + [TARGET]:
            df[c] = df[c].astype(str)
    return df


def load_raw() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    return fetch_openml(data_id=42178, as_frame=True, parser="auto").frame


def load_clean() -> pd.DataFrame:
    return clean(load_raw())


def feature_columns(df: pd.DataFrame):
    """Retourne (num_cols, cat_cols) à partir du DataFrame nettoyé."""
    num_cols = [c for c in NUM_COLS if c in df.columns]
    cat_cols = [c for c in df.columns if c not in num_cols + [TARGET]]
    return num_cols, cat_cols
