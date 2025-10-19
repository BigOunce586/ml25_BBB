import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 1. Cargar datasets originales ---
train_path = "customer_purchase_train_modify.csv"
test_path = "customer_purchase_test_modify.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"📦 Train original: {train.shape[0]} filas, {train.shape[1]} columnas")
print(f"📦 Test original:  {test.shape[0]} filas, {test.shape[1]} columnas")

# --- 2. Crear etiqueta binaria en el train ---
train["label"] = train["purchases_in_category"].apply(lambda x: 1 if x > 0 else 0)

# --- 3. Eliminar columnas que causan fuga o no aportan ---
cols_to_drop = [
    "purchases_in_category",
    "total_spent_in_category",
    "avg_spent_in_category",
    "color_count_red"
]
train = train.drop(columns=[c for c in cols_to_drop if c in train.columns], errors="ignore")
test = test.drop(columns=[c for c in cols_to_drop if c in test.columns], errors="ignore")

# --- 4. Asegurar consistencia entre columnas ---
common_cols = [col for col in train.columns if col in test.columns]
print(f"✅ Columnas en común (sin label): {len(common_cols)}")

# --- 5. Eliminar identificadores que no se usan para entrenamiento ---
feature_cols = [c for c in common_cols if c not in ["customer_id"]]

# --- 6. Codificar variables categóricas si existen ---
# (no siempre es necesario, pero por seguridad)
for col in feature_cols:
    if train[col].dtype == "object":
        le = LabelEncoder()
        all_values = list(train[col].astype(str)) + list(test[col].astype(str))
        le.fit(all_values)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

# --- 7. Eliminar columnas sin variabilidad ---
low_var = [col for col in feature_cols if train[col].nunique() <= 1]
if low_var:
    print(f"⚠️ Columnas sin variabilidad eliminadas: {low_var}")
    train = train.drop(columns=low_var)
    test = test.drop(columns=low_var, errors="ignore")
    feature_cols = [c for c in feature_cols if c not in low_var]

# --- 8. Escalar las variables numéricas ---
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train[feature_cols])
test_scaled = scaler.transform(test[feature_cols])

# --- 9. Reconstruir DataFrames escalados ---
train_ready = pd.DataFrame(train_scaled, columns=feature_cols)
test_ready = pd.DataFrame(test_scaled, columns=feature_cols)

train_ready["label"] = train["label"].values  # agregar etiqueta al train
train_ready["customer_id"] = train["customer_id"].values
test_ready["customer_id"] = test["customer_id"].values

# --- 10. Guardar archivos listos ---
train_ready.to_csv("customer_purchase_train_ready.csv", index=False)
test_ready.to_csv("customer_purchase_test_ready.csv", index=False)

print("\n✅ Archivos generados correctamente:")
print(" - customer_purchase_train_ready.csv")
print(" - customer_purchase_test_ready.csv")
print(f"Variables finales: {len(feature_cols)}")
print(f"Train listo: {train_ready.shape}, Test listo: {test_ready.shape}")
