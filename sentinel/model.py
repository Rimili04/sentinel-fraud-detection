"""Model pipeline, batch scoring, risk tiers and explanations."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .features import CAT_FEATURES, FEATURES, FEATURE_LABELS, NUM_FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "sentinel_model.joblib")
METRICS_PATH = os.path.join(ROOT, "models", "metrics.json")

# contributions of these are folded into a more readable parent feature
MERGE = {"log_amount": "amount", "is_night": "hour", "is_weekend": "dow"}


def build_pipeline(seed=42):
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CAT_FEATURES)],
        remainder="passthrough",
    )
    clf = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=500, max_leaf_nodes=31, min_samples_leaf=25,
        l2_regularization=1.0, categorical_features=list(range(len(CAT_FEATURES))),
        early_stopping=True, validation_fraction=0.12, n_iter_no_change=25, random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _ordered(X: pd.DataFrame) -> pd.DataFrame:
    return X[FEATURES]


def score(pipe, X: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(_ordered(X))[:, 1]


def assign_tiers(p: np.ndarray, thresholds: dict) -> np.ndarray:
    return np.where(p >= thresholds["high"], "High",
                    np.where(p >= thresholds["medium"], "Medium", "Low"))


def load_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not trained yet. Run:  python train.py")
    bundle = joblib.load(MODEL_PATH)
    with open(METRICS_PATH) as f:
        bundle["metrics"] = json.load(f)
    return bundle


def contributions(pipe, X: pd.DataFrame, background: pd.DataFrame, n_bg: int = 24, seed: int = 0) -> np.ndarray:
    """Shapley-style attribution by permutation sampling.

    For each row and each sampled background transaction we add the row's
    feature values one at a time in a random order and record how the score
    changes. Averaging gives each feature's fair share of (score - typical
    score), which stays correct when several signals overlap (new device +
    foreign city + failed logins) - a plain one-feature swap would not.
    Positive = pushed the risk score up.
    """
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    enc_cols = CAT_FEATURES + [f for f in FEATURES if f not in CAT_FEATURES]
    Z = pre.transform(_ordered(X))
    Zb = pre.transform(_ordered(background))
    n, F = Z.shape
    rng = np.random.default_rng(seed)
    rows = n * n_bg
    Xr = np.repeat(Z, n_bg, axis=0)
    Br = Zb[rng.integers(0, len(Zb), size=rows)]
    perm = np.argsort(rng.random((rows, F)), axis=1)      # feature order per sample
    rank = np.argsort(perm, axis=1)                        # position of each feature
    prev = clf.predict_proba(Br)[:, 1]
    contrib = np.zeros((rows, F))
    ar = np.arange(rows)
    for k in range(1, F + 1):
        cur = clf.predict_proba(np.where(rank < k, Xr, Br))[:, 1]
        contrib[ar, perm[:, k - 1]] = cur - prev
        prev = cur
    contrib = contrib.reshape(n, n_bg, F).mean(axis=1)
    out = np.zeros((n, F))
    for j, f in enumerate(enc_cols):
        out[:, FEATURES.index(f)] = contrib[:, j]
    return out


def fold(contrib_row: np.ndarray) -> dict:
    d = {f: float(c) for f, c in zip(FEATURES, contrib_row)}
    for src, dst in MERGE.items():
        d[dst] = d.get(dst, 0.0) + d.pop(src, 0.0)
    return d


def describe(feature: str, row: pd.Series) -> str:
    v = row.get(feature)
    try:
        if feature == "amount":
            a = float(v)
            if a < 10:
                return f"Micro-amount (${a:,.2f}), a card-testing pattern"
            return f"Large amount (${a:,.2f})"
        if feature == "amount_to_user_mean":
            return f"Amount is {float(v):.1f}x this user's usual spend"
        if feature == "amount_zscore_user":
            return f"Amount is {float(v):.1f} std devs above the user's norm"
        if feature == "txn_count_1h":
            return f"{int(v)} transactions from this user within an hour"
        if feature == "txn_count_24h":
            return f"{int(v)} transactions from this user in 24h"
        if feature == "mins_since_prev":
            m = float(v)
            if m < 1:
                return "Within seconds of the user's previous transaction"
            return f"Only {m:.0f} min after the user's previous transaction"
        if feature == "failed_attempts_24h":
            return f"{int(v)} failed attempts in the last 24h"
        if feature == "new_device":
            return "Device not previously seen for this user"
        if feature == "city_mismatch":
            return "City differs from the user's usual location"
        if feature == "country_mismatch":
            return "Country differs from the user's usual location"
        if feature == "cities_24h":
            return f"{int(v)} different cities within 24h"
        if feature == "hour":
            return f"Unusual time of day ({int(v):02d}:00)"
        if feature == "dow":
            return "Unusual day of week"
        if feature == "round_amount":
            return "Round-number amount (cash-out pattern)"
        if feature == "merchant_category":
            return f"Merchant category: {str(v).replace('_', ' ')}"
        if feature == "device_type":
            return f"Device type: {str(v).replace('_', ' ')}"
        if feature == "transaction_type":
            return {"ECOM": "Card-not-present online purchase", "TRANSFER": "Account-to-account transfer",
                    "ATM": "ATM cash withdrawal", "POS": "In-store card payment",
                    "BILL_PAY": "Bill payment"}.get(str(v), f"Transaction type: {v}")
        if feature == "country":
            return f"Country: {v}"
        if feature == "account_age_days":
            return f"Account age: {int(v)} days"
    except (TypeError, ValueError):
        pass
    return FEATURE_LABELS.get(feature, feature)


def top_reasons(contrib_row: np.ndarray, row: pd.Series, k: int = 3, min_c: float = 0.015):
    folded = fold(contrib_row)
    items = sorted(folded.items(), key=lambda kv: kv[1], reverse=True)
    return [(f, c, describe(f, row)) for f, c in items[:k] if c >= min_c]


def risk_actions(tier: str) -> str:
    return {
        "High": "Hold the transaction and send to an analyst now; trigger step-up verification with the customer.",
        "Medium": "Queue for analyst review and request step-up authentication if the user is active.",
        "Low": "No action. Transaction looks consistent with normal behaviour.",
    }[tier]
