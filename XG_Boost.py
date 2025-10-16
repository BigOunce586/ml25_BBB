import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# --- 1. Cargar datasets ---
train = pd.read_csv("customer_purchase_train_modify.csv")
test = pd.read_csv("customer_purchase_test_modify.csv")

# --- 2. Crear etiqueta ---
train["label"] = train["purchases_in_category"].apply(lambda x: 1 if x > 0 else 0)

# --- 3. Eliminar columnas con fuga de información ---
cols_to_drop = [
    "purchases_in_category",
    "avg_spent_in_category",
    "total_spent_in_category"
]

feature_cols = [c for c in train.columns if c not in ["customer_id", "label"] + cols_to_drop]
X = train[feature_cols]
y = train["label"]

# Eliminar columnas sin variabilidad
low_var = [col for col in X.columns if X[col].nunique() <= 1]
if low_var:
    print(f"⚠️ Eliminando columnas con un solo valor: {low_var}")
    X = X.drop(columns=low_var)

# --- 4. División train/validación ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 5. Escalado ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# --- 6. Preparar test ---
test_ids = test["customer_id"].copy()
missing_cols = [c for c in feature_cols if c not in test.columns]
for col in missing_cols:
    test[col] = 0
test_features = test[X.columns]
test_scaled = scaler.transform(test_features)

# --- 7. Modelo XGBoost (regularizado) ---
model = XGBClassifier(
    n_estimators=300,          # número de árboles
    learning_rate=0.05,        # tasa de aprendizaje baja
    max_depth=4,               # profundidad moderada
    subsample=0.8,             # fracción de muestras por árbol
    colsample_bytree=0.8,      # fracción de columnas por árbol
    min_child_weight=5,        # controla sobreajuste en hojas pequeñas
    reg_lambda=2.0,            # regularización L2
    reg_alpha=0.5,             # regularización L1
    scale_pos_weight=1,        # balancear clases si es necesario
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train_scaled, y_train)

# --- 8. Evaluación ---
pred_val = model.predict(X_val_scaled)
acc = accuracy_score(y_val, pred_val)
f1 = f1_score(y_val, pred_val)
cm = confusion_matrix(y_val, pred_val)

print("\n✅ Evaluación en validación simple (XGBoost regularizado):")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("Matriz de confusión:\n", cm)

# --- 9. Validación cruzada ---
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_scaled_full = scaler.fit_transform(X)
acc_cv = cross_val_score(model, X_scaled_full, y, cv=kfold, scoring="accuracy").mean()
f1_cv = cross_val_score(model, X_scaled_full, y, cv=kfold, scoring="f1").mean()

print(f"\n🔁 Cross-validation Accuracy promedio: {acc_cv:.4f}")
print(f"🔁 Cross-validation F1 promedio:       {f1_cv:.4f}")

# --- 10. Importancia de variables ---
importancia = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n⚡ Variables más importantes (XGBoost):")
print(importancia.head(15))

plt.figure(figsize=(10,6))
top_features = importancia.head(15)
plt.barh(top_features["feature"], top_features["importance"], color="darkorange")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.title("Importancia de Variables - XGBoost (Regularizado)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# --- 11. Reentrenar con todo el dataset ---
model.fit(X_scaled_full, y)

# --- 12. Predicciones finales ---
preds_test = model.predict(test_scaled)

pred_df = pd.DataFrame({
    "customer_id": test_ids,
    "prediction": preds_test
})
pred_df.to_csv("predicciones_XGBoost_final.csv", index=False)

# --- 13. Guardar modelo y escalador ---
joblib.dump(model, "best_model_XGBoost_final.pkl")
joblib.dump(scaler, "scaler_XGBoost_final.pkl")

print("\n✅ Archivo 'predicciones_XGBoost_final.csv' generado correctamente.")
print(f"Filas del test: {pred_df.shape[0]}")
print(pred_df.head())
