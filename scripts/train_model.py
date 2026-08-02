"""
Trains a RandomForestClassifier on the Telco churn dataset and saves:
- model.pkl            (trained model)
- encoders.pkl         (LabelEncoders for each categorical column)
- feature_columns.pkl  (ordered list of feature column names)
- metrics.json         (accuracy, roc_auc, precision, recall, feature importances)
"""
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

df = pd.read_csv("telco_churn.csv")

target = "Churn"
categorical_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]
numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
feature_cols = categorical_cols + numeric_cols

encoders = {}
df_enc = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    encoders[col] = le

target_le = LabelEncoder()
y = target_le.fit_transform(df_enc[target])  # Yes -> 1, No -> 0 (alphabetical check below)
# ensure "Yes" maps to 1
if list(target_le.classes_) == ["No", "Yes"]:
    pass
else:
    y = 1 - y
encoders["Churn"] = target_le

X = df_enc[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(
    n_estimators=400, max_depth=10, min_samples_leaf=4, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "model_name": "Random Forest",
    "accuracy": round(accuracy_score(y_test, y_pred) * 100, 1),
    "roc_auc": round(roc_auc_score(y_test, y_proba), 3),
    "precision": round(precision_score(y_test, y_pred) * 100, 1),
    "recall": round(recall_score(y_test, y_pred) * 100, 1),
    "n_train": len(X_train),
    "n_test": len(X_test),
}

importances = dict(zip(feature_cols, model.feature_importances_.round(4).tolist()))
metrics["feature_importances"] = dict(sorted(importances.items(), key=lambda x: -x[1]))

print(json.dumps(metrics, indent=2))

joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(feature_cols, "feature_columns.pkl")
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# also save churn distribution for the pie chart on the app's home/prediction page
dist = df["Churn"].value_counts(normalize=True).round(4).to_dict()
with open("churn_distribution.json", "w") as f:
    json.dump(dist, f, indent=2)

print("\nSaved: model.pkl, encoders.pkl, feature_columns.pkl, metrics.json, churn_distribution.json")
