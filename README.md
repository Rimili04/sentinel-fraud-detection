# Sentinel Upgraded Version

Batch payment-fraud analytics. Upload a CSV with thousands (or hundreds of thousands) of
transactions, every row is scored by a trained ML model, and analysts get a full
investigation dashboard. There is no one-transaction-at-a-time entry; drill-down only
opens when an analyst clicks a flagged transaction.

## Run it

```bash
pip install -r requirements.txt
python train.py        # ~15 s: generates data, trains, evaluates, saves models/ + data/
python app.py          # open http://127.0.0.1:5000
```

Click **Run the 15,000-transaction demo**, or upload `data/sample_transactions.csv`
(no labels) / `data/sample_transactions_labeled.csv` (adds precision/recall panel).

## CSV format

Required: `transaction_id, timestamp, user_id, amount`
Optional (improves accuracy): `transaction_type` (POS/ECOM/ATM/TRANSFER/BILL_PAY),
`merchant_category, city, country, device_type, device_id, failed_attempts_24h,
account_age_days`. Add `is_fraud` (0/1) to grade the model against your labels.
Template: `/api/template.csv`.

## What you get

- Risk score 0-100 for every transaction, tiered High / Medium / Low
- Executive overview: exposure value, tier split, score distribution, daily trend,
  fraud vs normal, auto-written key findings, most exposed users
- Patterns: hourly trend, weekday x hour heatmap, city and country, transaction type,
  merchant category, device, amount bands, failed attempts, transaction frequency
- Searchable, sortable suspicious-transaction table with reason codes
- Analyst drill-down: Shapley-style factor breakdown, details, same-user history
- Downloads: suspicious transactions CSV, full analysed dataset CSV
- Model page: holdout ROC-AUC / PR-AUC, precision / recall, feature importance,
  recall by fraud scenario, baseline comparison

## How it works

1. `sentinel/features.py` validates the file and builds per-user behaviour features
   inside the batch: velocity (1h / 24h), time since last transaction, spend vs the
   user's norm, new device, unusual city / country, night, round amounts, failed attempts.
2. `sentinel/model.py` scores with a HistGradientBoosting classifier. Tier thresholds
   (High 50+, Medium 15+) were tuned on a validation split in `train.py`.
3. `sentinel/analysis.py` builds every aggregate and the written insights.
4. `app.py` (Flask) serves the API; `static/` is the dashboard (vanilla JS, custom
   SVG charts, no CDN libraries; only Google Fonts are optional).

## Results (independent 15,002-row holdout, 3.5% fraud)

ROC-AUC 0.981, PR-AUC 0.892 (logistic regression baseline: 0.962 / 0.835).
Flagging Medium+High catches 84% of fraud at 78% precision; High only is 95% precision.
Stealth fraud that mimics normal behaviour is caught only ~23% of the time.

## Honest limits

- The bundled model is trained on **simulated** transactions (`sentinel/datagen.py`).
  Metrics above describe that simulation, not your real data. For production, replace the
  data step in `train.py` with your labelled history, retrain, and re-check thresholds.
- Behaviour features come from the uploaded batch itself, so very small files or users
  with a single transaction get less context.
- Reason codes are generated for up to 6,000 top-scored flagged rows per batch; any
  transaction can still be explained on demand in the drill-down.
- Speed: about 15k rows in 3 s, 150k rows in 30 s on one CPU core. Limit is 1.5M rows
  per file; keep the last 4 analyses in memory. Use gunicorn or waitress for deployment.
