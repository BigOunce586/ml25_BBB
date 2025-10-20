import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

train_df = pd.read_csv("train_final.csv")
test_df = pd.read_csv("test_final.csv")

y = train_df["label"]
X = train_df.drop(columns=["label"])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "auc",
    "num_leaves": 64,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_estimators": 1200,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_samples": 50,
    "reg_alpha": 1.5,
    "reg_lambda": 2.0,
    "scale_pos_weight": 2.0,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1
}

callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    callbacks=callbacks
)

joblib.dump(model, "lightgbm_model.pkl")

y_prob_val = model.predict(X_val, num_iteration=model.best_iteration)
y_pred_val = (y_prob_val > 0.5).astype(int)

acc = accuracy_score(y_val, y_pred_val)
auc = roc_auc_score(y_val, y_prob_val)
print("Validación Accuracy:", round(acc, 4))
print("Validación AUC:", round(auc, 4))
print("Matriz de confusión:\n", confusion_matrix(y_val, y_pred_val))
print("Reporte de clasificación:\n", classification_report(y_val, y_pred_val))

probs = model.predict(test_df, num_iteration=model.best_iteration)
threshold = np.percentile(probs, 85)
preds = (probs > threshold).astype(int)

submission = pd.DataFrame({
    "ID": range(len(test_df)),
    "pred": preds
})
submission.to_csv("predictions_lgbm.csv", index=False)
