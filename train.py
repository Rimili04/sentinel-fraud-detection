"""Train the Sentinel fraud model end to end.

    python train.py

Steps: generate training data -> feature engineering -> train baseline
(logistic regression) and final model (gradient boosting) -> choose risk-tier
thresholds on a validation split -> evaluate on an independent holdout batch ->
permutation importance -> save model, metrics and sample CSVs.
"""
import datetime as dt
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sentinel.datagen import make_dataset
from sentinel.features import (CAT_FEATURES, FEATURE_LABELS, FEATURES,
                               NUM_FEATURES, build_features, validate_and_clean)
from sentinel.model import (METRICS_PATH, MODEL_PATH, assign_tiers,
                            build_pipeline, score)

ROOT = os.path.dirname(os.path.abspath(__file__))


def prepare(df):
    clean = validate_and_clean(df.drop(columns=["is_fraud", "fraud_type"], errors="ignore"))
    return clean, build_features(clean)


def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-9)}


def main():
    t0 = time.time()
    print("1/6 generating data ...")
    train_df = make_dataset(n_total=60000, n_users=7000, fraud_rate=0.035, seed=11)
    hold_df = make_dataset(n_total=15000, n_users=1800, fraud_rate=0.035, seed=7)
    _, Xtr_all = prepare(train_df)
    _, Xho = prepare(hold_df)
    ytr_all = train_df["is_fraud"].to_numpy()
    yho = hold_df["is_fraud"].to_numpy()

    print("2/6 splitting train / validation ...")
    Xtr, Xva, ytr, yva = train_test_split(Xtr_all, ytr_all, test_size=0.2, stratify=ytr_all, random_state=1)

    print("3/6 baseline logistic regression ...")
    lr = Pipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)])),
        ("clf", LogisticRegression(max_iter=800, class_weight="balanced")),
    ])
    lr.fit(Xtr[FEATURES], ytr)
    lr_p = lr.predict_proba(Xho[FEATURES])[:, 1]

    print("4/6 training gradient boosting model ...")
    pipe = build_pipeline()
    pipe.fit(Xtr[FEATURES], ytr)
    n_iter = int(pipe.named_steps["clf"].n_iter_)
    p_va = score(pipe, Xva)
    p_ho = score(pipe, Xho)

    # ---- tier thresholds from validation ----
    prec, rec, thr = precision_recall_curve(yva, p_va)
    prec, rec = prec[:-1], rec[:-1]
    ok_high = np.where(prec >= 0.85)[0]
    high = float(thr[ok_high[0]]) if len(ok_high) else 0.7
    ok_med = np.where(rec >= 0.90)[0]
    medium = float(thr[ok_med[-1]]) if len(ok_med) else 0.3
    high = float(np.clip(high, 0.5, 0.9))
    medium = float(np.clip(medium, 0.15, high - 0.15))
    thresholds = {"high": round(high, 3), "medium": round(medium, 3)}
    print("   thresholds:", thresholds)

    print("5/6 evaluating on independent holdout + permutation importance ...")
    tiers = assign_tiers(p_ho, thresholds)
    flagged = (tiers != "Low").astype(int)
    high_only = (tiers == "High").astype(int)
    per_type = {}
    for ft in sorted(set(hold_df["fraud_type"]) - {""}):
        m = (hold_df["fraud_type"] == ft).to_numpy()
        per_type[ft] = {"count": int(m.sum()), "caught": int(flagged[m].sum()),
                        "recall": float(flagged[m].mean())}
    metrics = {
        "model": "HistGradientBoostingClassifier",
        "baseline": "LogisticRegression (class-balanced)",
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "train_rows": int(len(Xtr)), "validation_rows": int(len(Xva)), "holdout_rows": int(len(Xho)),
        "train_fraud_rate": float(ytr.mean()), "holdout_fraud_rate": float(yho.mean()),
        "boosting_iterations": n_iter,
        "thresholds": thresholds,
        "holdout": {
            "roc_auc": float(roc_auc_score(yho, p_ho)),
            "pr_auc": float(average_precision_score(yho, p_ho)),
            "flagged_medium_plus": prf(yho, flagged),
            "flagged_high_only": prf(yho, high_only),
            "by_fraud_type": per_type,
        },
        "baseline_holdout": {
            "roc_auc": float(roc_auc_score(yho, lr_p)),
            "pr_auc": float(average_precision_score(yho, lr_p)),
        },
        "features": [{"name": f, "label": FEATURE_LABELS[f]} for f in FEATURES],
    }
    pi = permutation_importance(pipe, Xho[FEATURES], yho, scoring="average_precision",
                                n_repeats=3, random_state=3, n_jobs=1)
    imp = {}
    for f, v in zip(FEATURES, pi.importances_mean):
        key = {"log_amount": "amount", "is_night": "hour", "is_weekend": "dow"}.get(f, f)
        imp[key] = imp.get(key, 0.0) + max(float(v), 0.0)
    total = sum(imp.values()) or 1.0
    metrics["importance"] = sorted(
        [{"feature": k, "label": FEATURE_LABELS[k], "importance": v / total} for k, v in imp.items()],
        key=lambda d: d["importance"], reverse=True)

    print("6/6 saving artefacts ...")
    bg = Xtr_all.sample(200, random_state=5).reset_index(drop=True)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    joblib.dump({"pipeline": pipe, "thresholds": thresholds, "background": bg,
                 "features": FEATURES, "version": "sentinel-upgraded-1.0"}, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    hold_df.to_csv(os.path.join(ROOT, "data", "sample_transactions_labeled.csv"), index=False)
    hold_df.drop(columns=["is_fraud", "fraud_type"]).to_csv(
        os.path.join(ROOT, "data", "sample_transactions.csv"), index=False)

    h = metrics["holdout"]
    print(f"\nHoldout ROC-AUC {h['roc_auc']:.4f} | PR-AUC {h['pr_auc']:.4f}  (baseline LR: "
          f"{metrics['baseline_holdout']['roc_auc']:.4f} / {metrics['baseline_holdout']['pr_auc']:.4f})")
    m = h["flagged_medium_plus"]
    print(f"Medium+High flags: precision {m['precision']:.3f} recall {m['recall']:.3f}  "
          f"({m['tp'] + m['fp']} flagged of {len(yho)})")
    m = h["flagged_high_only"]
    print(f"High only        : precision {m['precision']:.3f} recall {m['recall']:.3f}")
    for k, v in per_type.items():
        print(f"  {k:15s} recall {v['recall']:.2f} ({v['caught']}/{v['count']})")
    print("Top features:", [(d['feature'], round(d['importance'], 3)) for d in metrics['importance'][:6]])
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
