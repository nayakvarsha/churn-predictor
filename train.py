import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from features import add_features

import glob
csv_path = glob.glob("data/*.csv*")[0]            # first CSV-like file inside data/
df = pd.read_csv(csv_path)
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df = df.drop(columns=["customerID"])
df = add_features(df)

X = df.drop(columns="Churn")
y = df["Churn"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

num = X.select_dtypes("number").columns
cat = X.select_dtypes(exclude="number").columns
pre = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
])

spw = (y_tr == 0).sum() / (y_tr == 1).sum()
best = Pipeline([("pre", pre), ("model", XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    scale_pos_weight=spw, eval_metric="logloss", random_state=42))])
best.fit(X_tr, y_tr)

proba = best.predict_proba(X_te)[:, 1]
print("Test ROC-AUC:", round(roc_auc_score(y_te, proba), 3))
print("Test PR-AUC:", round(average_precision_score(y_te, proba), 3))

COST_MISS, COST_OFFER = 15000, 1500
def total_cost(t):
    pred = proba >= t
    fn = ((~pred) & (y_te == 1)).sum()
    return fn * COST_MISS + pred.sum() * COST_OFFER

ts = np.linspace(0.05, 0.95, 91)
best_t = ts[int(np.argmin([total_cost(t) for t in ts]))]
print("Best threshold:", round(best_t, 2))

joblib.dump({"model": best, "threshold": best_t}, "model.pkl")
print("Saved model.pkl")