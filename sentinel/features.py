"""Feature engineering - identical code path for training and batch scoring.

Behavioural features (velocity, user baselines, new device, location change)
are computed *within the uploaded batch*, per user, so a CSV of any size is
scored with full history context and no one-row-at-a-time lookups.
"""
import numpy as np
import pandas as pd

REQUIRED = ["transaction_id", "timestamp", "user_id", "amount"]
OPTIONAL_DEFAULTS = {
    "transaction_type": "UNKNOWN",
    "merchant_category": "unknown",
    "city": "unknown",
    "country": "unknown",
    "device_type": "unknown",
    "device_id": "",
    "failed_attempts_24h": 0,
    "account_age_days": 365,
}
CAT_FEATURES = ["transaction_type", "merchant_category", "device_type", "country"]
NUM_FEATURES = [
    "amount", "log_amount", "hour", "dow", "is_night", "is_weekend", "round_amount",
    "failed_attempts_24h", "account_age_days",
    "txn_count_1h", "txn_count_24h", "mins_since_prev",
    "amount_to_user_mean", "amount_zscore_user",
    "new_device", "city_mismatch", "country_mismatch", "cities_24h",
]
FEATURES = NUM_FEATURES + CAT_FEATURES

FEATURE_LABELS = {
    "amount": "Transaction amount",
    "log_amount": "Transaction amount (log scale)",
    "hour": "Hour of day",
    "dow": "Day of week",
    "is_night": "Night-time activity",
    "is_weekend": "Weekend activity",
    "round_amount": "Round-number amount",
    "failed_attempts_24h": "Failed attempts (24h)",
    "account_age_days": "Account age",
    "txn_count_1h": "Transactions in last hour",
    "txn_count_24h": "Transactions in last 24h",
    "mins_since_prev": "Time since previous transaction",
    "amount_to_user_mean": "Amount vs user's usual spend",
    "amount_zscore_user": "Amount deviation from user norm",
    "new_device": "Unrecognised device",
    "city_mismatch": "Unusual city for user",
    "country_mismatch": "Unusual country for user",
    "cities_24h": "Distinct cities in 24h",
    "transaction_type": "Transaction type",
    "merchant_category": "Merchant category",
    "device_type": "Device type",
    "country": "Country",
}


class SchemaError(ValueError):
    pass


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, check required columns, fill optional ones."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(
            "Missing required column(s): " + ", ".join(missing) +
            ". Required: " + ", ".join(REQUIRED) + "."
        )
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    bad = df["timestamp"].isna() | df["amount"].isna() | df["user_id"].isna()
    if bad.all():
        raise SchemaError("No valid rows: check the timestamp format (e.g. 2026-10-01 14:32:10) and numeric amount.")
    dropped = int(bad.sum())
    df = df[~bad].copy()
    df["failed_attempts_24h"] = pd.to_numeric(df["failed_attempts_24h"], errors="coerce").fillna(0).clip(lower=0)
    df["account_age_days"] = pd.to_numeric(df["account_age_days"], errors="coerce").fillna(365).clip(lower=0)
    for c in ["transaction_type", "merchant_category", "city", "country", "device_type", "device_id"]:
        df[c] = df[c].fillna(OPTIONAL_DEFAULTS[c]).astype(str).str.strip()
    df["user_id"] = df["user_id"].astype(str)
    df["transaction_id"] = df["transaction_id"].astype(str)
    df.attrs["dropped_rows"] = dropped
    return df.reset_index(drop=True)


def _window_counts(ts_sec: np.ndarray, window: int) -> np.ndarray:
    """Count of events in (t-window, t] for a sorted array of timestamps."""
    left = np.searchsorted(ts_sec, ts_sec - window, side="right")
    return np.arange(len(ts_sec)) - left + 1


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame aligned with df.index containing all model features."""
    n = len(df)
    ts = df["timestamp"]
    sec = ((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy().astype("int64")
    out = pd.DataFrame(index=df.index)
    out["amount"] = df["amount"].astype(float)
    out["log_amount"] = np.log1p(out["amount"].clip(lower=0))
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek
    out["is_night"] = ((out["hour"] >= 22) | (out["hour"] <= 5)).astype(int)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["round_amount"] = ((out["amount"] % 100 == 0) & (out["amount"] >= 100)).astype(int)
    out["failed_attempts_24h"] = df["failed_attempts_24h"].astype(float)
    out["account_age_days"] = df["account_age_days"].astype(float)

    # --- per-user behavioural features ---
    order = np.lexsort((sec, df["user_id"].to_numpy()))
    users = df["user_id"].to_numpy()[order]
    s_sorted = sec[order]
    amt_sorted = out["amount"].to_numpy()[order]
    city_sorted = df["city"].to_numpy()[order]
    ctry_sorted = df["country"].to_numpy()[order]
    dev_sorted = df["device_id"].to_numpy()[order]
    typ_sorted = df["transaction_type"].to_numpy()[order]

    bounds = np.flatnonzero(np.r_[True, users[1:] != users[:-1], True])
    c1h = np.zeros(n); c24h = np.zeros(n); since = np.full(n, 10080.0)
    ratio = np.ones(n); zsc = np.zeros(n); utot = np.zeros(n)
    newdev = np.zeros(n); citym = np.zeros(n); ctrym = np.zeros(n); cities24 = np.ones(n)

    for a, b in zip(bounds[:-1], bounds[1:]):
        k = b - a
        s = s_sorted[a:b]
        c1h[a:b] = _window_counts(s, 3600)
        c24h[a:b] = _window_counts(s, 86400)
        if k > 1:
            since[a + 1:b] = np.minimum(np.diff(s) / 60.0, 10080.0)
        utot[a:b] = k
        amt = amt_sorted[a:b]
        if k > 1:
            tot = amt.sum()
            loo_mean = (tot - amt) / (k - 1)
            ratio[a:b] = amt / np.maximum(loo_mean, 1.0)
            # leave-one-out std
            sq = (amt ** 2).sum()
            loo_var = np.maximum((sq - amt ** 2) / (k - 1) - loo_mean ** 2, 0)
            zsc[a:b] = (amt - loo_mean) / np.maximum(np.sqrt(loo_var), 0.25 * np.maximum(loo_mean, 1.0))
        # modal city / country / device (digital only) within the batch
        cs = pd.Series(city_sorted[a:b]); ms = cs.mode().iloc[0]
        citym[a:b] = (cs.to_numpy() != ms) & (k > 1)
        cts = pd.Series(ctry_sorted[a:b]); mc = cts.mode().iloc[0]
        ctrym[a:b] = (cts.to_numpy() != mc) & (k > 1)
        digital = np.isin(typ_sorted[a:b], ["ECOM", "TRANSFER", "BILL_PAY"])
        dv = pd.Series(dev_sorted[a:b])
        if digital.any() and k > 1:
            md = dv[digital].astype(str).value_counts().index[0]
            newdev[a:b] = digital & (dv.to_numpy() != md) & (dv.to_numpy() != "")
        # distinct cities within trailing 24h (small groups -> simple loop)
        if k > 1:
            lefts = np.searchsorted(s, s - 86400, side="right")
            for i in range(k):
                if i - lefts[i] > 0:
                    cities24[a + i] = len(set(city_sorted[a + lefts[i]:a + i + 1]))

    def unsort(x):
        r = np.empty(n); r[order] = x; return r

    out["txn_count_1h"] = unsort(c1h)
    out["txn_count_24h"] = unsort(c24h)
    out["mins_since_prev"] = unsort(since)
    out["amount_to_user_mean"] = unsort(ratio)
    out["amount_zscore_user"] = np.clip(unsort(zsc), -10, 25)
    out["new_device"] = unsort(newdev)
    out["city_mismatch"] = unsort(citym)
    out["country_mismatch"] = unsort(ctrym)
    out["cities_24h"] = unsort(cities24)

    for c in CAT_FEATURES:
        out[c] = df[c].astype(str)
    return out[FEATURES]
