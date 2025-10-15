import pandas as pd

# --- Cargar el dataset ---
train = pd.read_csv("customer_purchases_train.csv")

# --- Limpiar ID si viene con 'CUST_' ---
train['customer_id'] = train['customer_id'].astype(str).str.replace('CUST_', '').astype(int)

# --- Obtener IDs únicos ---
ids_unicos = sorted(train['customer_id'].unique())

# --- Guardar en CSV ---
pd.DataFrame({'customer_id': ids_unicos}).to_csv("train_ids_unicos.csv", index=False)

print(f"✅ Archivo 'train_ids_unicos.csv' generado correctamente.")
print(f"Total de IDs únicos: {len(ids_unicos)}")
