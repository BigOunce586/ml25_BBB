import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib

train_df = pd.read_csv("train_final.csv")
test_df = pd.read_csv("test_final.csv")

if "customer_id" in train_df.columns:
    mean_purchase = train_df.groupby("customer_id")["label"].mean().to_dict()
    train_df["customer_score"] = train_df["customer_id"].map(mean_purchase)
    test_df["customer_score"] = test_df["customer_id"].map(mean_purchase).fillna(train_df["customer_score"].mean())

y = train_df["label"]
X = train_df.drop(columns=["label"])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=4.0,
    reg_alpha=2.0,
    min_child_weight=10,
    gamma=2.0,
    scale_pos_weight=2.0,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
joblib.dump(model, "xgb_by_customer.pkl")

y_pred_val = model.predict(X_val)
acc_val = accuracy_score(y_val, y_pred_val)
auc_val = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

print("Validación Accuracy:", round(acc_val, 4))
print("Validación AUC:", round(auc_val, 4))
print("Matriz de confusión:\n", confusion_matrix(y_val, y_pred_val))
print("Reporte de clasificación:\n", classification_report(y_val, y_pred_val))

probs = model.predict_proba(test_df)[:, 1]
preds = (probs > 0.5).astype(int)

submission = pd.DataFrame({
    "ID": range(len(test_df)),
    "pred": preds
})
submission.to_csv("predictions_xgb_by_customer.csv", index=False)
