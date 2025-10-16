import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# --- 1. Cargar datasets ---
train = pd.read_csv("customer_purchase_train_modify.csv")
test = pd.read_csv("customer_purchase_test_modify.csv")

# --- 2. Crear etiqueta ---
train["label"] = train["purchases_in_category"].apply(lambda x: 1 if x > 0 else 0)

# --- 3. Eliminar columnas con fuga ---
cols_to_drop = [
    "purchases_in_category",
    "avg_spent_in_category",
    "total_spent_in_category"
]

feature_cols = [c for c in train.columns if c not in ["customer_id", "label"] + cols_to_drop]
X = train[feature_cols]
y = train["label"]

# --- 4. Eliminar columnas sin variabilidad ---
low_var = [col for col in X.columns if X[col].nunique() <= 1]
if low_var:
    print(f"⚠️ Eliminando columnas con un solo valor: {low_var}")
    X = X.drop(columns=low_var)

# --- 5. División train/validación ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 6. Escalado ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# --- 7. Preparar test ---
test_ids = test["customer_id"].copy()
missing_cols = [c for c in feature_cols if c not in test.columns]
for col in missing_cols:
    test[col] = 0
test_features = test[X.columns]
test_scaled = scaler.transform(test_features)

# --- 8. Modelo LightGBM regularizado ---
model = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    reg_alpha=0.5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# --- 9. Evaluación ---
pred_val = model.predict(X_val_scaled)
acc = accuracy_score(y_val, pred_val)
f1 = f1_score(y_val, pred_val)
cm = confusion_matrix(y_val, pred_val)

print("\n✅ Evaluación en validación simple (LightGBM regularizado):")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("Matriz de confusión:\n", cm)

# --- 10. Validación cruzada ---
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_scaled_full = scaler.fit_transform(X)
acc_cv = cross_val_score(model, X_scaled_full, y, cv=kfold, scoring="accuracy").mean()
f1_cv = cross_val_score(model, X_scaled_full, y, cv=kfold, scoring="f1").mean()

print(f"\n🔁 Cross-validation Accuracy promedio: {acc_cv:.4f}")
print(f"🔁 Cross-validation F1 promedio:       {f1_cv:.4f}")

# --- 11. Importancia de variables ---
importancia = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n🌿 Variables más importantes (LightGBM):")
print(importancia.head(15))

plt.figure(figsize=(10,6))
top_features = importancia.head(15)
plt.barh(top_features["feature"], top_features["importance"], color="limegreen")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.title("Importancia de Variables - LightGBM (Regularizado)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# --- 12. Reentrenar con todo el dataset ---
model.fit(X_scaled_full, y)

# --- 13. Predicciones finales ---
preds_test = model.predict(test_scaled)

pred_df = pd.DataFrame({
    "customer_id": test_ids,
    "prediction": preds_test
})
pred_df.to_csv("predicciones_LightGBM_final.csv", index=False)


joblib.dump(model, "best_model_LightGBM_final.pkl")
joblib.dump(scaler, "scaler_LightGBM_final.pkl")

print("\n Archivo 'predicciones_LightGBM_final.csv' generado correctamente.")
print(f"Filas del test: {pred_df.shape[0]}")
print(pred_df.head())