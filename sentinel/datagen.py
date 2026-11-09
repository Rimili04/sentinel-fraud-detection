"""Synthetic transaction generator with realistic fraud scenarios.

Fraud scenarios simulated:
  card_testing   - bursts of tiny e-commerce charges, new device, failed attempts
  takeover       - account takeover: new device/location, large transfers, night
  cash_out       - round-amount ATM / transfer cash-outs in a short window
  risky_merchant - purchase at a high-risk merchant category
  stealth        - fraud that mimics normal behaviour (hard cases, limits recall)

Legitimate traffic also contains noise (travel, big purchases, night usage,
failed PINs) so the model cannot reach a perfect score.
"""
import numpy as np
import pandas as pd

CITIES = [
    ("Mumbai", "IN"), ("Delhi", "IN"), ("Bengaluru", "IN"), ("Kolkata", "IN"),
    ("Hyderabad", "IN"), ("Chennai", "IN"), ("New York", "US"),
    ("San Francisco", "US"), ("Chicago", "US"), ("London", "GB"),
    ("Manchester", "GB"), ("Singapore", "SG"), ("Dubai", "AE"),
    ("Berlin", "DE"), ("Frankfurt", "DE"), ("Sydney", "AU"),
    ("Toronto", "CA"), ("Sao Paulo", "BR"),
]
HOME_W = np.array([14, 12, 12, 8, 8, 8, 9, 6, 5, 6, 3, 4, 3, 3, 2, 2, 2, 1], float)
HOME_W /= HOME_W.sum()

POS_CATS = ["grocery", "restaurants", "fuel", "fashion", "healthcare", "entertainment", "travel"]
POS_P = np.array([0.30, 0.22, 0.14, 0.12, 0.08, 0.07, 0.07])
POS_P = POS_P / POS_P.sum()
ECOM_CATS = ["electronics", "fashion", "entertainment", "travel", "grocery", "gift_cards",
             "crypto_exchange", "online_gambling", "jewelry"]
ECOM_P = np.array([0.17, 0.20, 0.16, 0.14, 0.22, 0.04, 0.02, 0.025, 0.025])
ECOM_P = ECOM_P / ECOM_P.sum()
RISKY = ["gift_cards", "crypto_exchange", "online_gambling", "jewelry", "electronics"]
DIGITAL_DEVICES = ["Android", "iOS", "Windows", "macOS"]
HOUR_W = np.array([1, .6, .4, .3, .4, 1, 2.5, 4, 5.5, 6, 6.5, 7, 7.5, 7, 6.5, 6.5, 7, 7.5, 8, 8, 7, 5, 3.5, 2], float)
HOUR_W /= HOUR_W.sum()

START = pd.Timestamp("2026-10-01")
DAYS = 31


def _ts(rng, n, night=False):
    day = rng.integers(0, DAYS, n)
    if night:
        hour = rng.choice([0, 1, 2, 3, 4, 23, 22], n, p=[.2, .2, .2, .15, .1, .1, .05])
    else:
        hour = rng.choice(24, n, p=HOUR_W)
    sec = rng.integers(0, 3600, n)
    return START + pd.to_timedelta(day, "D") + pd.to_timedelta(hour, "h") + pd.to_timedelta(sec, "s")


def make_dataset(n_total=15000, n_users=1800, fraud_rate=0.035, seed=7):
    rng = np.random.default_rng(seed)
    n_fraud_target = int(n_total * fraud_rate)
    n_legit = n_total - n_fraud_target

    # ---- users ----
    home_idx = rng.choice(len(CITIES), n_users, p=HOME_W)
    typical = np.exp(rng.normal(np.log(42), 0.55, n_users))
    age = np.clip(rng.gamma(2.0, 500, n_users), 20, 4500).astype(int)
    dev_ids = [f"D{u:05d}-{k}" for u in range(n_users) for k in range(2)]
    activity = rng.pareto(2.2, n_users) + 0.4
    activity /= activity.sum()
    user_ids = np.array([f"U{u+1:05d}" for u in range(n_users)])

    # ---- legitimate traffic ----
    u = rng.choice(n_users, n_legit, p=activity)
    ts = _ts(rng, n_legit)
    ttype = rng.choice(["POS", "ECOM", "ATM", "TRANSFER", "BILL_PAY"], n_legit, p=[.40, .30, .08, .12, .10])
    legit = pd.DataFrame({"u": u, "timestamp": ts, "transaction_type": ttype})
    cat = np.empty(n_legit, object)
    m = ttype == "POS"; cat[m] = rng.choice(POS_CATS, m.sum(), p=POS_P)
    m = ttype == "ECOM"; cat[m] = rng.choice(ECOM_CATS, m.sum(), p=ECOM_P)
    cat[ttype == "ATM"] = "cash_withdrawal"
    cat[ttype == "TRANSFER"] = "money_transfer"
    cat[ttype == "BILL_PAY"] = "utilities"
    legit["merchant_category"] = cat
    mult = np.exp(rng.normal(0, 0.7, n_legit))
    mult[ttype == "ATM"] *= 3
    mult[ttype == "TRANSFER"] *= 2.2
    mult[ttype == "BILL_PAY"] *= 1.6
    big = rng.random(n_legit) < 0.02
    mult[big] *= rng.uniform(5, 18, big.sum())
    legit["amount"] = np.round(typical[u] * mult, 2)
    city = home_idx[u].copy()
    trav = rng.random(n_legit) < 0.03
    city[trav] = rng.integers(0, len(CITIES), trav.sum())
    legit["city_i"] = city
    dtype = np.empty(n_legit, object)
    dig = np.isin(ttype, ["ECOM", "TRANSFER", "BILL_PAY"])
    dtype[dig] = rng.choice(DIGITAL_DEVICES, dig.sum(), p=[.45, .30, .17, .08])
    dtype[ttype == "POS"] = "POS_TERMINAL"
    dtype[ttype == "ATM"] = "ATM_KIOSK"
    emu = dig & (rng.random(n_legit) < 0.004)
    dtype[emu] = "Emulator"
    legit["device_type"] = dtype
    did = np.array([dev_ids[2 * x + rng.integers(0, 2)] for x in u], object)
    newdev = dig & (rng.random(n_legit) < 0.03)
    did[newdev] = [f"N{rng.integers(0, 10**7):07d}" for _ in range(newdev.sum())]
    did[~dig] = [f"T{rng.integers(0, 4000):05d}" for _ in range((~dig).sum())]
    legit["device_id"] = did
    fa = np.zeros(n_legit, int)
    r = rng.random(n_legit)
    fa[r < 0.04] = 1
    fa[r < 0.008] = 2
    fa[r < 0.002] = 3
    legit["failed_attempts_24h"] = fa
    legit["is_fraud"] = 0
    legit["fraud_type"] = ""
    rows = [legit]

    # ---- fraud scenarios ----
    made = 0
    # event weights = target txn share / avg txns per event, so the txn-level mix is balanced
    scen_p = {"card_testing": .038, "takeover": .12, "cash_out": .043, "risky_merchant": .131, "stealth": .13}
    names = list(scen_p)
    probs = np.array(list(scen_p.values()))
    probs = probs / probs.sum()
    while made < n_fraud_target:
        sc = rng.choice(names, p=probs)
        v = int(rng.integers(0, n_users))
        home = int(home_idx[v])
        base = _ts(rng, 1, night=(sc in ("takeover", "cash_out") and rng.random() < .7))[0]
        if sc == "card_testing":
            k = int(rng.integers(4, 10))
            t = base + pd.to_timedelta(np.sort(rng.integers(0, 1500, k)), "s")
            amt = np.round(rng.choice([0.5, 1, 1.5, 2, 3, 5, 9.99, 12], k) * rng.uniform(.8, 1.2, k), 2)
            c = int(rng.integers(0, len(CITIES))) if rng.random() < .6 else home
            fa_ = rng.integers(0, 7, k) * (rng.random(k) < .75)
            df = pd.DataFrame({"u": v, "timestamp": t, "transaction_type": "ECOM",
                               "merchant_category": rng.choice(["entertainment", "gift_cards", "electronics", "grocery", "fashion"], k),
                               "amount": amt, "city_i": c,
                               "device_type": rng.choice(["Emulator", "Android", "Windows"], k, p=[.4, .4, .2]),
                               "device_id": f"N{rng.integers(0, 10**7):07d}",
                               "failed_attempts_24h": fa_})
        elif sc == "takeover":
            k = int(rng.integers(1, 5))
            t = base + pd.to_timedelta(np.sort(rng.integers(0, 5400, k)), "s")
            c = int(rng.integers(0, len(CITIES))) if rng.random() < .75 else home
            ttp = rng.choice(["TRANSFER", "ECOM"], k, p=[.55, .45])
            mc = np.where(ttp == "TRANSFER", "money_transfer",
                          rng.choice(["electronics", "crypto_exchange", "gift_cards", "jewelry"], k))
            df = pd.DataFrame({"u": v, "timestamp": t, "transaction_type": ttp, "merchant_category": mc,
                               "amount": np.round(typical[v] * rng.uniform(3, 16, k), 2), "city_i": c,
                               "device_type": rng.choice(["Android", "Windows", "Emulator", "iOS"], k, p=[.4, .3, .2, .1]),
                               "device_id": f"N{rng.integers(0, 10**7):07d}",
                               "failed_attempts_24h": rng.choice([0, 1, 2, 3, 4, 5], k, p=[.2, .2, .2, .15, .15, .1])})
        elif sc == "cash_out":
            k = int(rng.integers(2, 6))
            t = base + pd.to_timedelta(np.sort(rng.integers(0, 7200, k)), "s")
            c = int(rng.integers(0, len(CITIES))) if rng.random() < .5 else home
            is_atm = rng.random(k) < .6
            df = pd.DataFrame({"u": v, "timestamp": t,
                               "transaction_type": np.where(is_atm, "ATM", "TRANSFER"),
                               "merchant_category": np.where(is_atm, "cash_withdrawal", "money_transfer"),
                               "amount": rng.choice([200, 300, 500, 800, 1000, 1500, 2000, 2500], k).astype(float),
                               "city_i": c,
                               "device_type": np.where(is_atm, "ATM_KIOSK", rng.choice(["Android", "Windows"], k)),
                               "device_id": [f"T{rng.integers(0, 4000):05d}" if a else f"N{rng.integers(0, 10**7):07d}" for a in is_atm],
                               "failed_attempts_24h": rng.choice([0, 1, 2, 3], k, p=[.4, .3, .2, .1])})
        elif sc == "risky_merchant":
            k = 1 if rng.random() < .7 else 2
            t = base + pd.to_timedelta(np.sort(rng.integers(0, 1800, k)), "s")
            df = pd.DataFrame({"u": v, "timestamp": t, "transaction_type": "ECOM",
                               "merchant_category": rng.choice(RISKY, k),
                               "amount": np.round(typical[v] * rng.uniform(3, 12, k), 2),
                               "city_i": home if rng.random() < .65 else int(rng.integers(0, len(CITIES))),
                               "device_type": rng.choice(DIGITAL_DEVICES + ["Emulator"], k, p=[.35, .2, .2, .1, .15]),
                               "device_id": f"N{rng.integers(0, 10**7):07d}" if rng.random() < .55 else dev_ids[2 * v],
                               "failed_attempts_24h": rng.choice([0, 1, 2], k, p=[.65, .25, .1])})
        else:  # stealth
            ttp = str(rng.choice(["ECOM", "POS", "TRANSFER"], p=[.6, .2, .2]))
            pool = ECOM_CATS if ttp == "ECOM" else (POS_CATS if ttp == "POS" else ["money_transfer"])
            df = pd.DataFrame({"u": [v], "timestamp": [base], "transaction_type": [ttp],
                               "merchant_category": [str(rng.choice(pool))],
                               "amount": np.round(typical[v] * rng.uniform(1.2, 4.5, 1), 2), "city_i": [home],
                               "device_type": ["POS_TERMINAL" if ttp == "POS" else str(rng.choice(DIGITAL_DEVICES))],
                               "device_id": [f"T{rng.integers(0, 4000):05d}" if ttp == "POS" else dev_ids[2 * v]],
                               "failed_attempts_24h": [0]})
        df["is_fraud"] = 1
        df["fraud_type"] = sc
        rows.append(df)
        made += len(df)

    d = pd.concat(rows, ignore_index=True)
    d = d[d["timestamp"] < START + pd.Timedelta(days=DAYS)]
    d = d.sort_values("timestamp").reset_index(drop=True)
    d["transaction_id"] = [f"TXN-{seed % 100:02d}{i+1:06d}" for i in range(len(d))]
    d["user_id"] = user_ids[d["u"].values]
    d["city"] = [CITIES[i][0] for i in d["city_i"]]
    d["country"] = [CITIES[i][1] for i in d["city_i"]]
    d["account_age_days"] = age[d["u"].values]
    d["amount"] = d["amount"].clip(lower=0.5).round(2)
    d["timestamp"] = d["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cols = ["transaction_id", "timestamp", "user_id", "amount", "transaction_type", "merchant_category",
            "city", "country", "device_type", "device_id", "failed_attempts_24h", "account_age_days"]
    return d[cols + ["is_fraud", "fraud_type"]]


if __name__ == "__main__":
    df = make_dataset()
    print(len(df), round(df.is_fraud.mean(), 4))
    print(df.fraud_type.value_counts())
