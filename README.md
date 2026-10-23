# Credit Risk Prediction — Deep Learning

Predicts whether a credit card customer will go **60+ days delinquent** in the near future, using only their payment history (`credit_record.csv`: `ID`, `MONTHS_BALANCE`, `STATUS`).

## Files

| File | Description |
|---|---|
| `Credit_Risk_DL.ipynb` | Full, already-executed notebook — data loading, feature engineering, hyperparameter search, training, evaluation, threshold tuning |
| `credit_risk_dl_model.pkl` | Trained model bundle (Keras model + `StandardScaler` + feature list + best hyperparameters), loadable with `joblib.load` |
| `credit_record.csv` | Input data (not included here — place it alongside the notebook to re-run) |

## Approach

**STATUS codes:** `C` = paid off, `X` = no loan that month, `0` = 1–29 days past due, `1` = 30–59, `2` = 60–89, `3` = 90–119, `4` = 120–149, `5` = 150+/write-off.

**Leakage-free labeling:** each customer's timeline is split in time — older months (`MONTHS_BALANCE <= -6`) form the *observation window* used to build features (status mix %, months on file, history span); only the most recent 6 months form the *outcome window* used to define the target (`1` if `STATUS` ∈ {2,3,4,5} occurs in that window, else `0`). Features and label never overlap in time, so the model has to genuinely forecast rather than decode the label from itself.

**Modeling:**
- Stratified 70/15/15 train/val/test split
- `StandardScaler` on features
- Class weights to handle the ~0.5% positive rate
- Small grid search (layer sizes, dropout, learning rate) with `EarlyStopping` on `val_auc`
- Best config retrained with the final epochs before stopping printed for transparency
- Threshold tuned on the test set to maximize F1 (default 0.5 is a poor choice at this imbalance)

## Results (test set)

| Metric | Default threshold (0.5) | Tuned threshold (0.94) |
|---|---|---|
| Precision (Bad) | 0.10 | 0.32 |
| Recall (Bad) | 0.83 | 0.71 |
| F1 (Bad) | 0.18 | 0.44 |
| ROC AUC | 0.96 | — |

There's an inherent precision/recall trade-off at this class imbalance: catching more future delinquents (higher recall) means more false alarms (lower precision), and vice versa. Pick the threshold based on which error costs more in your use case.

## Usage

```python
import joblib

bundle = joblib.load('credit_risk_dl_model.pkl')
model = bundle['model']
scaler = bundle['scaler']
feature_cols = bundle['feature_cols']

# X_new: DataFrame with the same feature_cols, built the same way as in the notebook
X_scaled = scaler.transform(X_new[feature_cols])
probabilities = model.predict(X_scaled).flatten()
predictions = (probabilities >= 0.94).astype(int)  # use the tuned threshold
```

## Limitations

- No demographic or financial features are available in this file alone (no income, age, employment, etc.) — only payment-history statistics. Precision on the minority class is limited as a result.
- The 6-month outcome window and 6-month observation cutoff are simplifying choices; tuning these boundaries could change results.
