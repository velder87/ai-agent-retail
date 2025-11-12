import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# ------------------ Config ------------------
AZ_SQL_SERVER   = os.getenv("AZ_SQL_SERVER")
AZ_SQL_DB       = os.getenv("AZ_SQL_DB")
AZ_SQL_USER     = os.getenv("AZ_SQL_USER")
AZ_SQL_PASSWORD = os.getenv("AZ_SQL_PASSWORD")
CORS_ALLOW      = os.getenv("CORS_ALLOW_ORIGINS", "*")

# ------------------ FastAPI ------------------
app = FastAPI(title="Retail Agent (FastAPI on Render)", version="1.3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ALLOW.split(",")] if CORS_ALLOW else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Lazy SQL engine ------------------
_engine = None
def get_engine():
    from sqlalchemy import create_engine  # safe import
    global _engine
    if _engine is not None:
        return _engine
    missing = [k for k,v in {
        "AZ_SQL_SERVER": AZ_SQL_SERVER,
        "AZ_SQL_DB": AZ_SQL_DB,
        "AZ_SQL_USER": AZ_SQL_USER,
        "AZ_SQL_PASSWORD": AZ_SQL_PASSWORD,
    }.items() if not v]
    if missing:
        raise HTTPException(status_code=500, detail=f"DB not configured. Missing env vars: {', '.join(missing)}")
    sql_url = f"mssql+pytds://{AZ_SQL_USER}:{AZ_SQL_PASSWORD}@{AZ_SQL_SERVER}:1433/{AZ_SQL_DB}?encrypt=yes"
    _engine = create_engine(
        sql_url,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=int(os.getenv("SQL_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("SQL_MAX_OVERFLOW", "5")),
    )
    return _engine

# ------------------ Schemas ------------------
class ChatIn(BaseModel):
    message: str

class SQLIn(BaseModel):
    query: str

class ForecastIn(BaseModel):
    product_id: Optional[int] = None
    horizon: int = 6

# ------------------ Utils SQL ------------------
def _guard_sql(q: str):
    bad = [" drop ", " delete ", " update ", " insert ", " alter ", " create ", " truncate ", " merge "]
    qlow = f" {q.lower()} "
    if any(x in qlow for x in bad):
        raise HTTPException(status_code=400, detail="DDL/DML not allowed in demo.")

def tool_sql(query: str) -> Dict[str, Any]:
    _guard_sql(query)
    try:
        eng = get_engine()
        with eng.begin() as conn:
            rows = conn.execute(text(query)).fetchall()
            cols = rows[0].keys() if rows else []
        return {"columns": list(cols), "rows": [list(r) for r in rows]}
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"SQL error: {str(e)}") from e

# ------------------ Forecast (sklearn en lazy) ------------------
def _calendar_features(dts):
    import numpy as np
    import pandas as pd
    dfc = pd.DataFrame({"ds": pd.to_datetime(dts)})
    dfc["weekofyear"] = dfc["ds"].dt.isocalendar().week.astype(int)
    dfc["month"]      = dfc["ds"].dt.month.astype(int)
    dfc["quarter"]    = dfc["ds"].dt.quarter.astype(int)
    w = dfc["weekofyear"].to_numpy()
    dfc["sin_w"] = np.sin(2 * np.pi * w / 52.0)
    dfc["cos_w"] = np.cos(2 * np.pi * w / 52.0)
    return dfc[["weekofyear", "month", "quarter", "sin_w", "cos_w"]]

def _make_supervised(series, max_lag: int = 6):
    import pandas as pd
    df = pd.DataFrame({"y": series.values})
    for k in range(1, max_lag + 1):
        df[f"lag{k}"] = df["y"].shift(k)
    return df.dropna().reset_index(drop=True)

def _fallback_avg(df_hist, horizon: int):
    import pandas as pd
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

def _train_and_forecast(df_hist, horizon: int = 6):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    if len(df_hist) < 12:
        return _fallback_avg(df_hist, horizon)
    df = df_hist.sort_values("ds").copy()
    df["ds"] = pd.to_datetime(df["ds"])

    max_lag = 6
    sup = _make_supervised(df["y"], max_lag=max_lag)
    sup["ds"] = df["ds"].iloc[max_lag:].reset_index(drop=True)
    cal = _calendar_features(sup["ds"])
    X = pd.concat([cal, sup[[f"lag{k}" for k in range(1, max_lag+1)]]], axis=1)
    y = sup["y"].to_numpy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ])
    model.fit(X, y)

    last_known_date = df["ds"].max()
    last_vals = df["y"].tail(max_lag).to_numpy().astype(float).tolist()
    preds = []
    future_dates = pd.date_range(last_known_date, periods=horizon + 1, freq="W-MON")[1:]

    for d in future_dates:
        cal_f = _calendar_features(pd.Series([d]))
        lag_feats = {f"lag{k}": last_vals[-k] for k in range(1, max_lag + 1)}
        Xf = pd.concat([cal_f.reset_index(drop=True), pd.DataFrame([lag_feats])], axis=1)
        yhat = float(model.predict(Xf)[0])
        last_vals.append(yhat)
        if len(last_vals) > max_lag:
            last_vals = last_vals[-max_lag:]
        preds.append({
            "ds": d.date().isoformat(),
            "yhat": round(yhat, 4),
            "yhat_lower": round(yhat * 0.9, 4),
            "yhat_upper": round(yhat * 1.1, 4),
        })
    return preds

def tool_forecast(product_id: Optional[int], horizon: int = 6):
    import pandas as pd
    horizon = max(1, min(int(horizon or 6), 52))
    base_sql = """
        SELECT ProductID, ProductName, CAST(WeekStart as date) as ds, CAST(SalesAmount as float) as y
        FROM dbo.vw_WeeklySalesByProduct
        {where_clause}
        ORDER BY ds
    """
    where_clause, params = "", {}
    if product_id is not None:
        where_clause = "WHERE ProductID = :pid"
        params = {"pid": product_id}
    try:
        eng = get_engine()
        with eng.begin() as conn:
            df = pd.read_sql(text(base_sql.format(where_clause=where_clause)), conn, params=params)
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"SQL error: {str(e)}") from e
    if df.empty:
        return []
    if product_id is None:
        pid = int(df.groupby("ProductID")["ds"].count().sort_values(ascending=False).index[0])
        df = df[df["ProductID"] == pid].copy()
    return _train_and_forecast(df[["ds","y"]], horizon=horizon)

# ------------------ Intent routing ------------------
def route_intent(msg: str) -> str:
    m = msg.lower()
    if any(x in m for x in ["prévoi", "prevoi", "forecast", "prédis", "prevois", "semaines"]):
        return "forecast"
    if any(x in m for x in ["select ", " top ", " from ", " where ", " group by", " order by"]) or "sql:" in m:
        return "sql"
    if "top" in m and "produit" in m:
        return "builtin_top"
    return "help"

# ------------------ Endpoints ------------------
@app.get("/health")
def health():
    db_ready = all([AZ_SQL_SERVER, AZ_SQL_DB, AZ_SQL_USER, AZ_SQL_PASSWORD])
    return {"status": "ok", "db_configured": db_ready}

@app.get("/diag/imports")
def diag_imports():
    """Vérifie les imports lourds sans faire tomber le serveur."""
    out = {}
    for mod in ["numpy", "pandas", "sklearn", "sqlalchemy"]:
        try:
            __import__(mod)
            out[mod] = "ok"
        except Exception as e:
            out[mod] = f"ERROR: {e.__class__.__name__}: {e}"
    return out

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
