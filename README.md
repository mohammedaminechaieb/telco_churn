# Telco Churn — Cas 13 (ENSI 2026-2027)

Prédiction du churn (OpenML 42178) : Streamlit (5 modules) + FastAPI.

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py                                   # ou notebook.ipynb -> models/*.pkl
uvicorn api.main:app --port 8000                  # Terminal 1 (Swagger: /docs)
streamlit run app/streamlit_app.py                # Terminal 2
```

Ordre impératif : générer `models/*.pkl` avant de lancer l'API.

Structure : `data/` (load, preprocessing, training), `api/main.py`, `app/` (accueil + `pages/`).
