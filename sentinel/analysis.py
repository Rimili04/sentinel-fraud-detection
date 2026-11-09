"""Score a whole batch and build every aggregate the dashboard needs."""
import time
import uuid

import numpy as np
import pandas as pd

from .features import OPTIONAL_DEFAULTS, build_features, validate_and_clean
from .model import assign_tiers, contributions, score, top_reasons

MAX_REASON_ROWS = 100
AMOUNT_EDGES = [0, 5, 20, 50, 100, 250, 500, 1000, 2500, np.inf]
AMOUNT_LABELS = ["<$5", "$5-20", "$20-50", "$50-100", "$100-250", "$250-500", "$500-1k", "$1k-2.5k", "$2.5k+"]


class Job:
    def __init__(self, df, X, summary, source_name):
        self.id = uuid.uuid4().hex[:12]
        self.df = df
        self.X = X
        self.summary = summary
        self.source_name = source_name
        self.created = time.time()


def _seg(df, col, min_total=1, top=None, sort="flagged"):
    g = df.groupby(col, observed=True).agg(
        total=("amount", "size"),
        flagged=("is_flagged", "sum"),
        high=("is_high", "sum"),
        amount=("amount", "sum"),
        flagged_amount=("flagged_amount", "sum"),
    ).reset_index()
    g = g[g["total"] >= min_total]
    g["rate"] = g["flagged"] / g["total"]
    g = g.sort_values(["flagged", "total"] if sort == "flagged" else sort, ascending=False)
    if top:
        g = g.head(top)
    g = g.rename(columns={col: "label"})
    g["label"] = g["label"].astype(str)
    return g.to_dict("records")


def _evaluation(df):
    if "is_fraud" not in df.columns:
        return None
    y = pd.to_numeric(df["is_fraud"], errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        return None
    y = y.astype(int).to_numpy()
    if y.sum() == 0:
        return None

    def block(pred):
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}

    ev = {
        "actual_fraud": int(y.sum()),
        "medium_plus": block(df["is_flagged"].to_numpy()),
        "high_only": block(df["is_high"].to_numpy()),
        "fraud_amount": float(df.loc[y == 1, "amount"].sum()),
        "caught_amount": float(df.loc[(y == 1) & (df["is_flagged"] == 1), "amount"].sum()),
    }
    if "fraud_type" in df.columns:
        rows = []
        for ft, g in df[(y == 1)].groupby("fraud_type"):
            if str(ft).strip() in ("", "nan"):
                continue
            rows.append({"label": str(ft), "count": int(len(g)), "caught": int(g["is_flagged"].sum()),
                         "recall": float(g["is_flagged"].mean())})
        ev["by_type"] = sorted(rows, key=lambda r: r["count"], reverse=True)
    return ev


def _insights(df, s, hours, types, cats, devs, cities, users_top, fail_rows):
    out = []
    n, f = s["n_transactions"], s["n_flagged"]
    if f == 0:
        return ["No transactions crossed the medium-risk threshold in this batch."]
    out.append(f"{f:,} of {n:,} transactions ({f / n:.1%}) are flagged for review, putting "
               f"${s['flagged_amount']:,.0f} ({s['flagged_amount'] / max(s['total_amount'], 1):.1%} of volume) at risk.")
    out.append(f"{s['n_high']:,} are high risk (score {s['thresholds']['high'] * 100:.0f}+) and should be held now; "
               f"{s['n_medium']:,} are medium and belong in the analyst queue.")
    night = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5])
    if night.mean() > 0:
        out.append(f"Night-time (22:00-05:59) is {night.mean():.0%} of traffic but "
                   f"{df.loc[night, 'is_flagged'].sum() / f:.0%} of flagged transactions.")
    hv = [h for h in hours if h["total"] >= 30]
    if hv:
        h = max(hv, key=lambda r: r["rate"])
        out.append(f"Riskiest hour is {h['hour']:02d}:00, where {h['rate']:.0%} of transactions are flagged.")
    tv = [t for t in types if t["total"] >= 30]
    if tv:
        t = max(tv, key=lambda r: r["rate"])
        out.append(f"{t['label']} is the riskiest channel at a {t['rate']:.1%} flag rate.")
    cv = [c for c in cats if c["total"] >= 30]
    if cv:
        c = max(cv, key=lambda r: r["rate"])
        out.append(f"Merchant category '{c['label'].replace('_', ' ')}' has the highest flag rate ({c['rate']:.1%}).")
    if cities:
        c = cities[0]
        out.append(f"{c['label']} accounts for the most flagged transactions ({c['flagged']:,}).")
    hi_fail = [r for r in fail_rows if r["label"] in ("3", "4+")]
    base = next((r for r in fail_rows if r["label"] == "0"), None)
    if hi_fail and base and sum(r["total"] for r in hi_fail) >= 10:
        tot = sum(r["total"] for r in hi_fail); fl = sum(r["flagged"] for r in hi_fail)
        out.append(f"Accounts with 3+ failed attempts are flagged {fl / tot:.0%} of the time vs {base['rate']:.1%} with none.")
    if users_top:
        top10 = sum(u["flagged_amount"] for u in users_top[:10])
        out.append(f"The 10 most exposed users hold ${top10:,.0f} of flagged value "
                   f"({top10 / max(s['flagged_amount'], 1):.0%} of the total).")
    return out


def run_analysis(raw: pd.DataFrame, bundle: dict, source_name: str = "upload.csv") -> Job:
    t0 = time.time()
    present = {c.strip().lower().replace(" ", "_") for c in raw.columns}
    df = validate_and_clean(raw)
    dropped = df.attrs.get("dropped_rows", 0)
    X = build_features(df)
    pipe = bundle["pipeline"]
    th = bundle["thresholds"]

    # Predict in smaller batches to reduce peak prediction memory.
    p = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), 5000):
        end = min(start + 5000, len(X))
        p[start:end] = score(pipe, X.iloc[start:end])
    df["risk_score"] = np.round(p * 100, 1)
    df["risk_tier"] = assign_tiers(p, th)
    df["is_flagged"] = (df["risk_tier"] != "Low").astype(int)
    df["is_high"] = (df["risk_tier"] == "High").astype(int)
    df["flagged_amount"] = df["amount"] * df["is_flagged"]
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["txn_count_1h"] = X["txn_count_1h"].astype(int).to_numpy()
    df["amount_to_user_mean"] = X["amount_to_user_mean"].round(2).to_numpy()
    df["new_device"] = X["new_device"].astype(int).to_numpy()
    df["city_mismatch"] = X["city_mismatch"].astype(int).to_numpy()

    # ---- reason codes for flagged rows (highest scores first) ----
    df["top_reasons"] = ""
    idx = np.flatnonzero(df["is_flagged"].to_numpy() == 1)
    if len(idx) > MAX_REASON_ROWS:
        idx = idx[np.argsort(-p[idx])[:MAX_REASON_ROWS]]
    if len(idx):
        n_bg = int(max(4, min(24, 160000 // max(len(idx), 1))))
        C = contributions(pipe, X.iloc[idx], bundle["background"], n_bg=n_bg)
        reasons = []
        for r, i in enumerate(idx):
            reasons.append("; ".join(d for _, _, d in top_reasons(C[r], X.iloc[i], k=3)))
        df.loc[idx, "top_reasons"] = reasons

    n = len(df)
    flagged = int(df["is_flagged"].sum())
    notes = []
    missing_cols = [c for c in OPTIONAL_DEFAULTS if c not in present]
    if missing_cols:
        notes.append("Optional columns not found, defaults used: " + ", ".join(missing_cols) +
                     ". Adding them improves accuracy.")
    if dropped:
        notes.append(f"{dropped:,} rows were skipped because the timestamp or amount could not be read.")
    summary = {
        "source": source_name,
        "n_transactions": n,
        "n_users": int(df["user_id"].nunique()),
        "total_amount": float(df["amount"].sum()),
        "n_flagged": flagged,
        "n_high": int(df["is_high"].sum()),
        "n_medium": int(((df["risk_tier"] == "Medium")).sum()),
        "n_low": int((df["risk_tier"] == "Low").sum()),
        "flagged_amount": float(df["flagged_amount"].sum()),
        "high_amount": float(df.loc[df["is_high"] == 1, "amount"].sum()),
        "flag_rate": flagged / max(n, 1),
        "avg_score": float(df["risk_score"].mean()),
        "date_from": df["timestamp"].min().strftime("%Y-%m-%d %H:%M"),
        "date_to": df["timestamp"].max().strftime("%Y-%m-%d %H:%M"),
        "thresholds": th,
        "notes": notes,
        "seconds": round(time.time() - t0, 2),
    }

    # ---- aggregates ----
    hours = (df.groupby("hour").agg(total=("amount", "size"), flagged=("is_flagged", "sum"), high=("is_high", "sum"))
             .reindex(range(24), fill_value=0).reset_index())
    hours["rate"] = hours["flagged"] / hours["total"].replace(0, np.nan)
    hours = hours.fillna(0).to_dict("records")

    daily = df.groupby("date").agg(total=("amount", "size"), flagged=("is_flagged", "sum"),
                                   high=("is_high", "sum")).reset_index()
    daily = daily.rename(columns={"date": "label"}).to_dict("records")

    heat = np.zeros((7, 24), int); heat_tot = np.zeros((7, 24), int)
    np.add.at(heat, (df["dow"].to_numpy(), df["hour"].to_numpy()), df["is_flagged"].to_numpy())
    np.add.at(heat_tot, (df["dow"].to_numpy(), df["hour"].to_numpy()), 1)

    bins = np.arange(0, 105, 5)
    hist = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (df["risk_score"] >= lo) & ((df["risk_score"] < hi) if hi < 100 else (df["risk_score"] <= hi))
        sub = df.loc[m, "risk_tier"]
        hist.append({"lo": int(lo), "hi": int(hi), "Low": int((sub == "Low").sum()),
                     "Medium": int((sub == "Medium").sum()), "High": int((sub == "High").sum())})

    df["amount_bucket"] = pd.cut(df["amount"], AMOUNT_EDGES, labels=AMOUNT_LABELS, right=False)
    amount_rows = _seg(df, "amount_bucket", sort="flagged")
    order = {l: i for i, l in enumerate(AMOUNT_LABELS)}
    amount_rows.sort(key=lambda r: order[r["label"]])
    for r in amount_rows:
        r["normal"] = r["total"] - r["flagged"]

    df["failed_bucket"] = df["failed_attempts_24h"].clip(upper=4).astype(int).astype(str).replace({"4": "4+"})
    fail_rows = sorted(_seg(df, "failed_bucket"), key=lambda r: r["label"])
    df["velocity_bucket"] = df["txn_count_1h"].clip(upper=5).astype(int).astype(str).replace({"5": "5+"})
    vel_rows = sorted(_seg(df, "velocity_bucket"), key=lambda r: r["label"])

    types = _seg(df, "transaction_type")
    cats = _seg(df, "merchant_category", top=12)
    devs = _seg(df, "device_type")
    cities = _seg(df, "city", top=12)
    countries = _seg(df, "country", top=10)

    fl = df[df["is_flagged"] == 1]
    ug = fl.groupby("user_id").agg(flagged=("is_flagged", "sum"), flagged_amount=("amount", "sum"),
                                   max_score=("risk_score", "max")).reset_index()
    tot_u = df.groupby("user_id").size().rename("total")
    ug = ug.join(tot_u, on="user_id").sort_values("flagged_amount", ascending=False).head(12)
    users_top = ug.to_dict("records")

    full_cat_rows = _seg(df, "merchant_category")
    summary["insights"] = _insights(df, summary, hours, types, full_cat_rows, devs, cities, users_top, fail_rows)

    summary["charts"] = {
        "hours": hours, "daily": daily, "heatmap": heat.tolist(), "heatmap_total": heat_tot.tolist(),
        "histogram": hist, "amount": amount_rows, "failed": fail_rows, "velocity": vel_rows,
        "types": types, "categories": cats, "devices": devs, "cities": cities, "countries": countries,
        "users": users_top,
    }
    summary["evaluation"] = _evaluation(df)
    df.drop(columns=["amount_bucket", "failed_bucket", "velocity_bucket"], inplace=True)
    return Job(df.reset_index(drop=True), X.reset_index(drop=True), summary, source_name)
