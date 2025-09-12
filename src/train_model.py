import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

# paths
DATA_PATH = "data/synthetic_ehr.csv"
MODEL_PATH = "artifacts/model_xgb.joblib"
PLOTS_DIR = "plots"

os.makedirs("artifacts", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# 1) load data
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["readmitted_30d"])
y = df["readmitted_30d"]

num_features = ["age","has_diabetes","has_hypertension","prior_admissions","avg_glucose","meds_count","length_of_stay"]
cat_features = ["sex"]

# 2) preprocessing + model
pre = ColumnTransformer(
    [("ohe", OneHotEncoder(handle_unknown="ignore"), cat_features)],
    remainder="passthrough"
)

model = XGBClassifier(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
)

pipe = Pipeline([("pre", pre), ("xgb", model)])

# 3) train/test split + train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
pipe.fit(X_train, y_train)

# 4) evaluate
y_prob = pipe.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)
auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", round(auc, 3))
print(classification_report(y_test, y_pred, digits=3))

# 5) save model
joblib.dump(pipe, MODEL_PATH)
print(f"✅ saved model to {MODEL_PATH}")

# 6) plots: ROC + permutation importance
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve (XGBoost)")
plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), bbox_inches="tight")
plt.close()

# permutation importance on test set
result = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
imp = result.importances_mean

# feature names after preprocessing
ohe = pipe.named_steps["pre"].named_transformers_["ohe"]
ohe_names = list(ohe.get_feature_names_out(["sex"]))
feature_names = ohe_names + num_features

# sort and plot top 10
idx = np.argsort(imp)[::-1][:10]
labels = [feature_names[i] for i in idx][::-1]
vals = imp[idx][::-1]

plt.barh(labels, vals)
plt.xlabel("Mean decrease in accuracy")
plt.title("Permutation Importance (Top 10)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "perm_importance.png"), bbox_inches="tight")
plt.close()

print("✅ saved plots to plots/roc_curve.png and plots/perm_importance.png")
