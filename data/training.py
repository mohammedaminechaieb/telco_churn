"""Définition des 5 modèles, GridSearchCV, sauvegarde/chargement joblib."""
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from data.preprocessing import build_preprocessor

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "kNN", "Random Forest", "XGBoost"]


def get_model_specs(y_train):
    """name -> (estimateur, grille de paramètres préfixée 'clf__')."""
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=42),
            {"clf__max_depth": [3, 5, 7, None]},
        ),
        "kNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5, 7, 11]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10]},
        ),
        "XGBoost": (
            XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", random_state=42),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5]},
        ),
    }


def train_model(name, X_train, y_train, num_cols, cat_cols, spec=None):
    est, grid = spec or get_model_specs(y_train)[name]
    pipe = Pipeline([("prep", build_preprocessor(num_cols, cat_cols)), ("clf", est)])
    gs = GridSearchCV(pipe, grid, cv=3, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train[num_cols + cat_cols], y_train)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def train_all(X_train, y_train, num_cols, cat_cols, names=None, save=True, progress=None):
    specs = get_model_specs(y_train)
    fitted, best = {}, {}
    for i, name in enumerate(names or MODEL_NAMES):
        pipe, params, score = train_model(name, X_train, y_train, num_cols, cat_cols, specs[name])
        fitted[name] = pipe
        best[name] = {"params": params, "cv_auc": score}
        if save:
            save_model(name, pipe)
        if progress:
            progress(i + 1, name)
    return fitted, best


def model_path(name):
    return MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl"


def save_model(name, pipe):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, model_path(name))


def load_models():
    models = {}
    for name in MODEL_NAMES:
        p = model_path(name)
        if p.exists():
            models[name] = joblib.load(p)
    return models
