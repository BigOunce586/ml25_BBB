import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# --- 1. Cargar datos ---
train = pd.read_csv("customer_purchase_train_modify.csv")
test = pd.read_csv("customer_purchase_test_modify.csv")

train["label"] = train["purchases_in_category"].apply(lambda x: 1 if x > 0 else 0)

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

# --- 2. División ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Preparar test ---
test_ids = test["customer_id"].copy()
missing_cols = [c for c in feature_cols if c not in test.columns]
for col in missing_cols:
    test[col] = 0
test_features = test[X.columns]

# --- 4. Modelo Random Forest con regularización ---
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=8,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# --- 5. Evaluar ---
pred_val = model.predict(X_val)
acc = accuracy_score(y_val, pred_val)
f1 = f1_score(y_val, pred_val)
cm = confusion_matrix(y_val, pred_val)

print("\n✅ Evaluación en validación simple (ajustada):")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("Matriz de confusión:\n", cm)

# --- 6. Cross-validation ---
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_cv = cross_val_score(model, X, y, cv=kfold, scoring="accuracy").mean()
f1_cv = cross_val_score(model, X, y, cv=kfold, scoring="f1").mean()

print(f"\n🔁 Cross-validation Accuracy promedio: {acc_cv:.4f}")
print(f"🔁 Cross-validation F1 promedio:       {f1_cv:.4f}")

# --- 7. Importancia de variables ---
importancia = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n🌿 Variables más importantes:")
print(importancia.head(15))

plt.figure(figsize=(10,6))
top = importancia.head(15)
plt.barh(top["feature"], top["importance"], color="seagreen")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.title("Importancia de Variables - Random Forest (Regularizado)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# --- 8. Reentrenar y predecir ---
model.fit(X, y)
preds_test = model.predict(test_features)

pred_df = pd.DataFrame({"customer_id": test_ids, "prediction": preds_test})
pred_df.to_csv("predicciones_RandomForest_fixed.csv", index=False)
joblib.dump(model, "best_model_RandomForest_fixed.pkl")

print("\n✅ Archivo 'predicciones_RandomForest_fixed.csv' generado correctamente.")
print(f"Filas del test: {pred_df.shape[0]}")
print(pred_df.head())
