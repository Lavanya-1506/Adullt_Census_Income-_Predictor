import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# Load Dataset

df = pd.read_csv("data/raw/adult.csv")

# Remove Missing Values

df.replace(" ?", pd.NA, inplace=True)
df.dropna(inplace=True)

# Encode Target

df["income"] = df["income"].str.strip()

df["income"] = df["income"].map({
    "<=50K": 0,
    ">50K": 1
})

# Encode Categorical Columns

encoders = {}

categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    encoders[col] = le

# Features & Target

X = df.drop("income", axis=1)
y = df["income"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# Precision/Recall for positive class (income > 50K => 1)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    y_pred,
    labels=[1],
    average=None,
    zero_division=0,
)
precision = float(precision[0])
recall = float(recall[0])
f1 = float(f1[0])

cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()  # [[tn, fp],[fn,tp]]

# Feature importance from RandomForest
# (categorical encoding is label-encoded; importances are still meaningful per encoded feature.)
try:
    feature_importances = model.feature_importances_
except Exception:
    feature_importances = None

feature_names = X.columns.tolist()

if feature_importances is not None:
    feature_importance_pairs = [
        {"feature": feature_names[i], "importance": float(feature_importances[i])}
        for i in range(len(feature_names))
    ]
    feature_importance_pairs.sort(key=lambda x: x["importance"], reverse=True)
    top_features = feature_importance_pairs[:15]
else:
    feature_importance_pairs = []
    top_features = []

analytics = {
    "model_name": "RandomForestClassifier",
    "model_type": "Random Forest",
    "accuracy": float(accuracy),
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "confusion_matrix": cm,
    "feature_importances": feature_importance_pairs,
    "top_features": top_features,
}


print(f"Accuracy : {accuracy:.4f}")
print(f"Precision (class=1): {precision:.4f}")
print(f"Recall (class=1): {recall:.4f}")
print("Model Saved Successfully! (training complete)")


# Save Model + Analytics


joblib.dump(model, "models/income_model.pkl")
joblib.dump(encoders, "models/encoder.pkl")

import json

with open("models/analytics_metrics.json", "w", encoding="utf-8") as f:
    json.dump(analytics, f, ensure_ascii=False, indent=2)

