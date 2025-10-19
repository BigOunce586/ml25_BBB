import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib

train_df = pd.read_csv("train_final.csv")
test_df = pd.read_csv("test_final.csv")

y = train_df["label"]
X = train_df.drop(columns=["label"])

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # reemplaza NaN por la mediana de cada columna
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X, y)
joblib.dump(model, "logistic_model_imputed.pkl")

y_pred_train = model.predict(X)
y_proba_train = model.predict_proba(X)[:, 1]
acc = accuracy_score(y, y_pred_train)
auc = roc_auc_score(y, y_proba_train)
print("Entrenamiento Accuracy:", round(acc, 4))
print("Entrenamiento AUC:", round(auc, 4))
print("Matriz de confusión:\n", confusion_matrix(y, y_pred_train))
print("Reporte de clasificación:\n", classification_report(y, y_pred_train))

probs = model.predict_proba(test_df)[:, 1]
threshold = 0.5
preds = (probs > threshold).astype(int)

submission = pd.DataFrame({
    "ID": range(len(test_df)),
    "pred": preds
})
submission.to_csv("predictions_logistic.csv", index=False)
