import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- Cargar datasets ---
train = pd.read_csv("customer_purchase_train_modify.csv")
test = pd.read_csv("customer_purchase_test_modify.csv")

# ==========================================================
# 1️⃣ CREAR LABEL SI NO EXISTE
# ==========================================================
if "label" not in train.columns:
    train["label"] = (train["purchases_in_category"] > 0).astype(int)

# ==========================================================
# 2️⃣ PREPARAR DATOS
# ==========================================================
X = train.drop(columns=["customer_id", "label", "item_category"], errors="ignore")
y = train["label"]
X_test = test.drop(columns=["customer_id", "item_category"], errors="ignore")

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# 3️⃣ ENTRENAR MODELO (Random Forest)
# ==========================================================
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced_subsample"
)
model.fit(X_scaled, y)

# ==========================================================
# 4️⃣ PREDICCIONES SOBRE TEST
# ==========================================================
test["pred"] = model.predict(X_test_scaled)

# ==========================================================
# 5️⃣ AGRUPAR POR CLIENTE → “COMPRAR MÁS DEL 50%”
# ==========================================================
# Calcular el porcentaje de productos que predice como 1
agg = (
    test.groupby("customer_id")["pred"]
    .agg(["mean", "count"])  # mean = proporción de 1s
    .reset_index()
)

# Si la proporción > 0.5 → 1, si no → 0
agg["will_buy"] = (agg["mean"] > 0.25).astype(int)

# ==========================================================
# 6️⃣ GUARDAR RESULTADO FINAL
# ==========================================================
agg[["customer_id", "will_buy"]].to_csv("predicciones_final.csv", index=False)

print("✅ Archivo 'predicciones_final.csv' generado correctamente.")
print(agg.head())
