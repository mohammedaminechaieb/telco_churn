"""Entraîne les 5 modèles et écrit models/*.pkl (équivalent du notebook)."""
from sklearn.metrics import average_precision_score, roc_auc_score

from data.load import feature_columns, load_clean
from data.preprocessing import split
from data.training import train_all

df = load_clean()
X_train, X_test, y_train, y_test = split(df)
num_cols, cat_cols = feature_columns(df)
fitted, best = train_all(
    X_train, y_train, num_cols, cat_cols, progress=lambda i, n: print(f"[{i}/5] {n}")
)
for n, p in fitted.items():
    pr = p.predict_proba(X_test[num_cols + cat_cols])[:, 1]
    print(f"{n:20s} AUC={roc_auc_score(y_test, pr):.3f} PR-AUC={average_precision_score(y_test, pr):.3f} {best[n]['params']}")
