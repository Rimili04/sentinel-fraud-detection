"""Sentinel web app.   python app.py   ->   http://127.0.0.1:5000"""
import io
import json
import math
import os
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

from sentinel.analysis import run_analysis
from sentinel.features import FEATURE_LABELS, FEATURES, SchemaError
from sentinel.model import (contributions, describe, fold, load_bundle,
                            risk_actions, score)

ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(ROOT, "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024
MAX_ROWS = 1_500_000

BUNDLE = load_bundle()
JOBS: "OrderedDict[str, object]" = OrderedDict()
LOCK = threading.Lock()


def clean(o):
    """Make numpy / NaN values JSON safe."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.strftime("%Y-%m-%d %H:%M:%S")
    return o


def ok(payload, status=200):
    return Response(json.dumps(clean(payload)), status=status, mimetype="application/json")


def err(msg, status=400):
    return ok({"error": msg}, status)


def store(job):
    with LOCK:
        JOBS[job.id] = job
        while len(JOBS) > 4:
            JOBS.popitem(last=False)


def get_job(job_id):
    with LOCK:
        return JOBS.get(job_id)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/model")
def model_info():
    m = dict(BUNDLE["metrics"])
    return ok(m)


@app.post("/api/analyze")
def analyze():
    f = request.files.get("file")
    if f is None or not f.filename:
        return err("Choose a CSV file to analyse.")
    if not f.filename.lower().endswith((".csv", ".txt")):
        return err("Only .csv files are supported.")
    try:
        raw = pd.read_csv(f.stream, low_memory=False)
    except Exception as e:  # noqa: BLE001
        return err(f"Could not read the file as CSV: {e}")
    if len(raw) == 0:
        return err("The file has no rows.")
    if len(raw) > MAX_ROWS:
        return err(f"File has {len(raw):,} rows; the limit is {MAX_ROWS:,}. Split it into batches.")
    try:
        job = run_analysis(raw, BUNDLE, f.filename)
    except SchemaError as e:
        return err(str(e))
    store(job)
    return ok({"job_id": job.id, "summary": job.summary})


@app.post("/api/analyze-sample")
def analyze_sample():
    path = os.path.join(ROOT, "data", "sample_transactions_labeled.csv")
    if not os.path.exists(path):
        return err("Sample file missing. Run: python train.py")
    job = run_analysis(pd.read_csv(path), BUNDLE, "sample_transactions_labeled.csv")
    store(job)
    return ok({"job_id": job.id, "summary": job.summary})


SORTABLE = {"risk_score", "amount", "timestamp", "user_id", "transaction_type", "merchant_category",
            "city", "device_type", "failed_attempts_24h", "transaction_id"}


@app.get("/api/job/<job_id>/transactions")
def transactions(job_id):
    job = get_job(job_id)
    if not job:
        return err("This analysis has expired. Upload the file again.", 404)
    df = job.df
    tier = request.args.get("tier", "suspicious")
    if tier == "suspicious":
        m = df["is_flagged"] == 1
    elif tier in ("High", "Medium", "Low"):
        m = df["risk_tier"] == tier
    else:
        m = pd.Series(True, index=df.index)
    q = request.args.get("q", "").strip().lower()
    sub = df[m]
    if q:
        cols = ["transaction_id", "user_id", "transaction_type", "merchant_category", "city", "country",
                "device_type", "top_reasons"]
        blob = sub[cols[0]].astype(str)
        for c in cols[1:]:
            blob = blob + " " + sub[c].astype(str)
        sub = sub[blob.str.lower().str.contains(q, regex=False)]
    sort = request.args.get("sort", "risk_score")
    if sort not in SORTABLE:
        sort = "risk_score"
    asc = request.args.get("dir", "desc") == "asc"
    sub = sub.sort_values(sort, ascending=asc, kind="mergesort")
    page = max(int(request.args.get("page", 1)), 1)
    size = min(max(int(request.args.get("size", 25)), 5), 200)
    total = len(sub)
    chunk = sub.iloc[(page - 1) * size: page * size]
    rows = []
    for i, r in chunk.iterrows():
        rows.append({
            "i": int(i), "transaction_id": r["transaction_id"], "timestamp": r["timestamp"],
            "user_id": r["user_id"], "amount": r["amount"], "transaction_type": r["transaction_type"],
            "merchant_category": r["merchant_category"], "city": r["city"], "country": r["country"],
            "device_type": r["device_type"], "failed_attempts_24h": r["failed_attempts_24h"],
            "risk_score": r["risk_score"], "risk_tier": r["risk_tier"], "top_reasons": r["top_reasons"],
        })
    return ok({"total": total, "page": page, "size": size, "pages": max(math.ceil(total / size), 1), "rows": rows})


@app.get("/api/job/<job_id>/transaction/<int:i>")
def drilldown(job_id, i):
    """Explain ONE selected transaction (analyst drill-down)."""
    job = get_job(job_id)
    if not job:
        return err("This analysis has expired. Upload the file again.", 404)
    if i < 0 or i >= len(job.df):
        return err("Transaction not found.", 404)
    row = job.df.iloc[i]
    x = job.X.iloc[[i]]
    C = contributions(BUNDLE["pipeline"], x, BUNDLE["background"], n_bg=100)[0]
    folded = fold(C)
    base = float(score(BUNDLE["pipeline"], BUNDLE["background"].iloc[:100]).mean())
    xr = x.iloc[0]
    factors = []
    for f, c in sorted(folded.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]:
        if abs(c) < 0.004:
            continue
        factors.append({"feature": f, "label": FEATURE_LABELS[f], "detail": describe(f, xr),
                        "points": round(c * 100, 1)})
    uid = row["user_id"]
    hist = job.df[job.df["user_id"] == uid].copy()
    hist["gap"] = (hist["timestamp"] - row["timestamp"]).abs()
    hist = hist.sort_values("gap").head(10).sort_values("timestamp")
    history = [{"i": int(j), "transaction_id": r["transaction_id"], "timestamp": r["timestamp"],
                "amount": r["amount"], "transaction_type": r["transaction_type"], "city": r["city"],
                "risk_score": r["risk_score"], "risk_tier": r["risk_tier"], "current": int(j) == i}
               for j, r in hist.iterrows()]
    skip = {"is_flagged", "is_high", "flagged_amount", "hour", "dow", "date", "top_reasons"}
    details = {k: v for k, v in row.items() if k not in skip}
    return ok({
        "i": i, "details": details, "risk_score": row["risk_score"], "risk_tier": row["risk_tier"],
        "baseline_score": round(base * 100, 1), "factors": factors, "history": history,
        "action": risk_actions(row["risk_tier"]),
        "user_summary": {"transactions": int((job.df["user_id"] == uid).sum()),
                         "flagged": int(job.df.loc[job.df["user_id"] == uid, "is_flagged"].sum())},
    })


def _csv_response(df, name):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={name}"})


def _export_frame(job, only_flagged):
    df = job.df
    drop = {"is_flagged", "is_high", "flagged_amount", "hour", "dow", "date"}
    lead = ["transaction_id", "timestamp", "user_id", "amount", "risk_score", "risk_tier", "top_reasons"]
    rest = [c for c in df.columns if c not in lead and c not in drop]
    out = df[lead + rest]
    if only_flagged:
        out = out[df["is_flagged"] == 1].sort_values("risk_score", ascending=False)
    return out


@app.get("/api/job/<job_id>/download/<kind>.csv")
def download(job_id, kind):
    job = get_job(job_id)
    if not job:
        return err("This analysis has expired. Upload the file again.", 404)
    if kind == "suspicious":
        return _csv_response(_export_frame(job, True), "sentinel_suspicious_transactions.csv")
    if kind == "full":
        return _csv_response(_export_frame(job, False), "sentinel_full_analysis.csv")
    return err("Unknown download.", 404)


@app.get("/api/template.csv")
def template():
    body = ("transaction_id,timestamp,user_id,amount,transaction_type,merchant_category,city,country,"
            "device_type,device_id,failed_attempts_24h,account_age_days\n"
            "TXN-000001,2026-10-01 09:14:22,U00042,38.50,POS,grocery,Mumbai,IN,POS_TERMINAL,T00112,0,812\n"
            "TXN-000002,2026-10-01 09:15:03,U00107,1249.00,ECOM,electronics,London,GB,Android,N0381920,2,95\n"
            "TXN-000003,2026-10-01 02:41:50,U00042,500.00,ATM,cash_withdrawal,Dubai,AE,ATM_KIOSK,T00871,1,812\n")
    return Response(body, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=sentinel_template.csv"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Sentinel running on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
