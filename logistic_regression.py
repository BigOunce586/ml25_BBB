import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# --- 1. Cargar datos ---
train = pd.read_csv("customer_purchase_train_ready.csv")
test = pd.read_csv("customer_purchase_test_ready.csv")

X = train.drop(columns=["label", "customer_id"])
y = train["label"]

# --- 2. División ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. Modelo ---
model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
model.fit(X_train, y_train)

# --- 4. Evaluación ---
pred_val = model.predict(X_val)
acc = accuracy_score(y_val, pred_val)
f1 = f1_score(y_val, pred_val)
cm = confusion_matrix(y_val, pred_val)
print("\n✅ Evaluación Regresión Logística:")
print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
print("Matriz de confusión:\n", cm)

# --- 5. Cross-validation ---
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_cv = cross_val_score(model, X, y, cv=kfold, scoring="accuracy").mean()
f1_cv = cross_val_score(model, X, y, cv=kfold, scoring="f1").mean()
print(f"\n🔁 Cross-validation | Accuracy: {acc_cv:.4f} | F1: {f1_cv:.4f}")

# --- 6. Entrenar completo y predecir test ---
model.fit(X, y)
preds = model.predict(test.drop(columns=["customer_id"]))

pred_df = pd.DataFrame({"id": range(len(preds)), "prediction": preds})
pred_df.to_csv("predicciones_logistic_ready.csv", index=False)

joblib.dump(model, "best_model_logistic_ready.pkl")
print("\n Archivo 'predicciones_logistic_ready.csv' generado correctamente.")
print(pred_df.head())
