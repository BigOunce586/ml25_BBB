import pandas as pd
import numpy as np

# --- 1. Cargar dataset original ---
train_path = "customer_purchase_train_modify.csv"
df = pd.read_csv(train_path)

# --- 2. Crear etiqueta ---
df["label"] = df["purchases_in_category"].apply(lambda x: 1 if x > 0 else 0)

print(f"\n📦 Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# --- 3. Calcular correlaciones ---
corr = df.corr(numeric_only=True)["label"].sort_values(ascending=False)
print("\n📊 Correlación con la etiqueta (label):")
print(corr.head(20))

# --- 4. Detectar columnas sospechosas ---
umbral_fuga = 0.85
fugas = corr[abs(corr) > umbral_fuga]
fugas = fugas.drop("label", errors="ignore")

if not fugas.empty:
    print("\n🚨 Posibles fugas de información detectadas (|corr| > 0.85):")
    for col, val in fugas.items():
        print(f" - {col}: correlación {val:.4f}")
else:
    print("\n✅ No se detectaron fugas directas (todas las correlaciones < 0.85).")

# --- 5. Detectar columnas con baja variabilidad ---
low_var = [col for col in df.columns if df[col].nunique() <= 1]
if low_var:
    print("\n⚠️ Columnas con un solo valor (sin variabilidad):")
    for col in low_var:
        print(f" - {col}")

# --- 6. Crear lista final de columnas a eliminar ---
cols_to_drop = [
    "purchases_in_category",
    "total_spent_in_category",
    "avg_spent_in_category"
]

cols_to_drop += list(fugas.index) + low_var
cols_to_drop = sorted(list(set(cols_to_drop)))  # eliminar duplicados

print("\n🧹 Columnas que se eliminarán del entrenamiento:")
print(cols_to_drop if cols_to_drop else "✅ Ninguna (dataset ya limpio)")

# --- 7. Generar dataset limpio ---
clean_df = df.drop(columns=cols_to_drop, errors="ignore")

# --- 8. Guardar versión limpia ---
output_path = "customer_purchase_train_clean.csv"
clean_df.to_csv(output_path, index=False)

print(f"\n✅ Archivo '{output_path}' generado correctamente.")
print(f"Columnas finales: {clean_df.shape[1]}")
print(f"Primeras columnas: {list(clean_df.columns[:10])}")
