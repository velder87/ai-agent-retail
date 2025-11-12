import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ------------------ Config ------------------
AZ_SQL_SERVER   = os.getenv("AZ_SQL_SERVER")      # ex: myserver.database.windows.net
AZ_SQL_DB       = os.getenv("AZ_SQL_DB")          # ex: northwind
AZ_SQL_USER     = os.getenv("AZ_SQL_USER")        # ex: sqladmin
AZ_SQL_PASSWORD = os.getenv("AZ_SQL_PASSWORD")    # ex: ********
CORS_ALLOW      = os.getenv("CORS_ALLOW_ORIGINS", "*")

if not all([AZ_SQL_SERVER, AZ_SQL_DB, AZ_SQL_USER, AZ_SQL_PASSWORD]):
    raise RuntimeError("Missing one or more env vars: AZ_SQL_SERVER, AZ_SQL_DB, AZ_SQL_USER, AZ_SQL_PASSWORD")

SQL_URL = f"mssql+pytds://{AZ_SQL_USER}:{AZ_SQL_PASSWORD}@{AZ_SQL_SERVER}:1433/{AZ_SQL_DB}?encrypt=yes"

engine = create_engine(
    SQL_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    pool_size=int(os.getenv("SQL_POOL_SIZE", "5")),
    max_overflow=int(os.getenv("SQL_MAX_OVERFLOW", "5")),
)

# ------------------ FastAPI ------------------
app = FastAPI(title="Retail Agent (FastAPI on Render)", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ALLOW.split(",")] if CORS_ALLOW else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Schemas -------------------
class ChatIn(BaseModel):
    message: str

class SQLIn(BaseModel):
    query: str

class ForecastIn(BaseModel):
    product_id: Optional[int] = None
    horizon: int = 6

# ------------------ Utils SQL -----------------
def _guard_sql(q: str):
    bad = [" drop ", " delete ", " update ", " insert ", " alter ", " create ", " truncate ", " merge "]
    qlow = f" {q.lower()} "
    if any(x in qlow for x in bad):
        raise HTTPException(status_code=400, detail="DDL/DML not allowed in demo.")

def tool_sql(query: str) -> Dict[str, Any]:
    _guard_sql(query)
    try:
        with engine.begin() as conn:
            rows = conn.execute(text(query)).fetchall()
            cols = rows[0].keys() if rows else []
        return {"columns": list(cols), "rows": [list(r) for r in rows]}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"SQL error: {str(e)}") from e

# ------------------ Feature engineering -----------------
def _calendar_features(dts: pd.Series) -> pd.DataFrame:
    # dts est une série datetime (hebdomadaire)
    dfc = pd.DataFrame({"ds": pd.to_datetime(dts)})
    dfc["weekofyear"] = dfc["ds"].dt.isocalendar().week.astype(int)
    dfc["month"]      = dfc["ds"].dt.month.astype(int)
    dfc["quarter"]    = dfc["ds"].dt.quarter.astype(int)

    # Encodage sin/cos (saisonnalité annuelle sur semaine)
    # Période ~52 semaines
    w = dfc["weekofyear"].to_numpy()
    dfc["sin_w"] = np.sin(2 * np.pi * w / 52.0)
    dfc["cos_w"] = np.cos(2 * np.pi * w / 52.0)
    return dfc[["weekofyear", "month", "quarter", "sin_w", "cos_w"]]

def _make_supervised(series: pd.Series, max_lag: int = 6) -> pd.DataFrame:
    """
    Transforme une série y(t) en dataset supervisé avec lags y(t-1..max_lag).
    Retourne un DataFrame avec colonnes: y, lag1..lagK
    """
    df = pd.DataFrame({"y": series.values})
    for k in range(1, max_lag + 1):
        df[f"lag{k}"] = df["y"].shift(k)
    df = df.dropna().reset_index(drop=True)
    return df

def _train_and_forecast(df_hist: pd.DataFrame, horizon: int = 6) -> List[Dict[str, Any]]:
    """
    df_hist: colonnes ['ds','y'] (hebdo)
    Étapes:
      - construire features calendaires + lags
      - fit Ridge (scikit-learn) sur l'historique
      - prédire en rolling pour horizon semaines (multi-step)
      - intervalles +/-10% (simple); on peut estimer l'écart-type résiduel si besoin
    """
    if len(df_hist) < 12:
        # Trop peu de points => fallback moyenne
        return _fallback_avg(df_hist, horizon)

    df = df_hist.sort_values("ds").copy()
    df["ds"] = pd.to_datetime(df["ds"])

    # Supervised avec lags
    max_lag = 6  # tu peux ajuster (ex: ajouter lag52 si plusieurs années)
    sup = _make_supervised(df["y"], max_lag=max_lag)

    # Aligner les dates pour les lignes conservées
    sup["ds"] = df["ds"].iloc[max_lag:].reset_index(drop=True)

    # Features calendaires
    cal = _calendar_features(sup["ds"])
    X = pd.concat([cal, sup[[f"lag{k}" for k in range(1, max_lag+1)]]], axis=1)
    y = sup["y"].to_numpy()

    # Modèle (scaler + ridge)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ])
    model.fit(X, y)

    # Prévision multi-step récursive
    last_known_date = df["ds"].max()
    # On garde la "fenêtre" des derniers lags depuis la série d'origine
    last_vals = df["y"].tail(max_lag).to_numpy().astype(float).tolist()

    preds = []
    # pour construire les futures dates hebdo (lundi)
    future_dates = pd.date_range(last_known_date, periods=horizon + 1, freq="W-MON")[1:]

    for d in future_dates:
        # Features calendaires pour la date d
        cal_f = _calendar_features(pd.Series([d]))
        # Lags: on prend les derniers max_lag (y(t-1..t-max_lag))
        lag_feats = {f"lag{k}": last_vals[-k] for k in range(1, max_lag + 1)}
        Xf = pd.concat([cal_f.reset_index(drop=True), pd.DataFrame([lag_feats])], axis=1)
        yhat = float(model.predict(Xf)[0])

        # maj de la fenêtre de lags
        last_vals.append(yhat)
        if len(last_vals) > max_lag:
            last_vals = last_vals[-max_lag:]

        preds.append({
            "ds": d.date().isoformat(),
            "yhat": round(yhat, 4),
            "yhat_lower": round(yhat * 0.9, 4),   # bornes simples (±10%)
            "yhat_upper": round(yhat * 1.1, 4),
        })
    return preds

def _fallback_avg(df_hist: pd.DataFrame, horizon: int) -> List[Dict[str, Any]]:
    df_hist = df_hist.sort_values("ds")
    if df_hist.empty:
        return []
    window = min(6, max(1, len(df_hist) // 3))
    avg = float(df_hist["y"].tail(window).mean())
    last_date = pd.to_datetime(df_hist["ds"]).max()
    future_dates = pd.date_range(last_date, periods=horizon + 1, freq="W-MON")[1:]
    return [{
        "ds": d.date().isoformat(),
        "yhat": round(avg, 4),
        "yhat_lower": round(avg * 0.9, 4),
        "yhat_upper": round(avg * 1.1, 4),
    } for d in future_dates]

# ------------------ Forecast (Azure SQL -> sklearn) --------------
def tool_forecast(product_id: Optional[int], horizon: int = 6) -> List[Dict[str, Any]]:
    """
    Suppose une vue: dbo.vw_WeeklySalesByProduct(ProductID, ProductName, WeekStart, SalesAmount)
    Renvoie une prévision hebdo sur 'horizon' semaines.
    """
    horizon = max(1, min(int(horizon or 6), 52))

    base_sql = """
        SELECT ProductID, ProductName, CAST(WeekStart as date) as ds, CAST(SalesAmount as float) as y
        FROM dbo.vw_WeeklySalesByProduct
        {where_clause}
        ORDER BY ds
    """
    where_clause = ""
    params = {}
    if product_id is not None:
        where_clause = "WHERE ProductID = :pid"
        params = {"pid": product_id}

    try:
        with engine.begin() as conn:
            df = pd.read_sql(text(base_sql.format(where_clause=where_clause)), conn, params=params)
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"SQL error: {str(e)}") from e

    if df.empty:
        return []

    # Choix du produit: si non fourni, prendre celui avec la série la plus longue
    if product_id is None:
        # série la plus “présente”
        counts = df.groupby("ProductID")["ds"].count().sort_values(ascending=False)
        pid = int(counts.index[0])
        df = df[df["ProductID"] == pid].copy()

    df_focus = df[["ds", "y"]].copy()
    return _train_and_forecast(df_focus, horizon=horizon)

# ------------------ Intent routing -----------------
def route_intent(msg: str) -> str:
    m = msg.lower()
    if any(x in m for x in ["prévoi", "prevoi", "forecast", "prédis", "prevois", "semaines"]):
        return "forecast"
    if any(x in m for x in ["select ", " top ", " from ", " where ", " group by", " order by"]) or "sql:" in m:
        return "sql"
    if "top" in m and "produit" in m:
        return "builtin_top"
    return "help"

# ------------------ Endpoints ----------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/sql")
def run_sql(req: SQLIn):
    return tool_sql(req.query)

@app.post("/forecast")
def run_forecast(req: ForecastIn):
    return {"forecast": tool_forecast(req.product_id, req.horizon)}

@app.post("/chat")
def chat(req: ChatIn):
    intent = route_intent(req.message)
    if intent == "builtin_top":
        q = """
        SELECT TOP 5 ProductName, SUM(SalesAmount) AS SalesAmount_8w
        FROM dbo.vw_WeeklySalesByProduct
        WHERE WeekStart >= DATEADD(WEEK, -8, DATEADD(WEEK, DATEDIFF(WEEK, 0, GETDATE()), 0))
        GROUP BY ProductName
        ORDER BY SalesAmount_8w DESC
        """
        return {"intent": intent, "answer": tool_sql(q)}
    if intent == "sql":
        message = req.message.strip()
        q = message[4:].strip() if message.lower().startswith("sql:") else message
        return {"intent": intent, "answer": tool_sql(q)}
    if intent == "forecast":
        return {"intent": intent, "answer": tool_forecast(None, 6)}
    return {
        "intent": "help",
        "hint": [
            "Exemples:",
            "• \"Top 5 produits sur les 8 dernières semaines\"",
            "• \"sql: SELECT TOP 10 * FROM dbo.vw_WeeklySalesByProduct\"",
            "• \"Prévois les ventes pour les 6 prochaines semaines\""
        ]
    }
