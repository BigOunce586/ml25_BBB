import pandas as pd
import numpy as np
import re

# === CARGAR DATA ORIGINAL ===
ruta_origen = r"C:\Users\paola\Downloads\ml25_BBB\src\ml25\datasets\customer_purchases\customer_purchases_train_modify.csv"
df = pd.read_csv(ruta_origen)

# === TRANSFORMACIONES DE COLOR ===
df["color"] = df["color"].replace({
    "imgb.jpg": 1,
    "imgbl.jpg": 0.714,
    "imgg.jpg": 0.429,
    "imgp.jpg": -0.429,
    "imgw.jpg": -0.714,
    "imgo.jpg": 0.143,
    "imgy.jpg": -1,
    "imgr.jpg": -0.1
})

# === GÉNERO ===
df["customer_gender"] = df["customer_gender"].replace({"female": 1, "male": 2}).fillna(3)

# === MAPEO DE CATEGORÍAS ===
mapa = {
    "jacket": 1.0,
    "blouse": 0.6,
    "jeans": 0.2,
    "skirt": -0.2,
    "shoes": -0.6,
    "dress": -1.0
}
df["item_category"] = df["item_category"].map(mapa)

# === EXTRAER ADJETIVOS DEL TÍTULO ===
adjectives = [
    "premium", "casual", "modern", "exclusive",
    "classic", "stylish", "elegant", "lightweight", "durable"
]
pattern = re.compile(r'\b(' + '|'.join(map(re.escape, adjectives)) + r')\b', flags=re.IGNORECASE)

def keep_adjectives(title):
    if pd.isna(title):
        return ""
    matches = [m.group(0) for m in pattern.finditer(str(title))]
    seen, result = set(), []
    for m in matches:
        key = m.lower()
        if key not in seen:
            seen.add(key)
            result.append(m)
    return " ".join(result)

df["item_title"] = df["item_title"].apply(keep_adjectives)

# === LABEL ORIGINAL ===
if "label" not in df.columns:
    df["label"] = 1
else:
    df["label"] = 1

# === GENERAR DATASET BALANCEADO ===
todas_categorias = df["item_category"].dropna().unique().tolist()
perfil = df.groupby("customer_id")["item_category"].apply(list).reset_index(name="compradas")

no_compras = []
for _, row in perfil.iterrows():
    compradas = set(row["compradas"])
    no_compradas = [c for c in todas_categorias if c not in compradas]
    cliente = row["customer_id"]
    cliente_info = df[df["customer_id"] == cliente].iloc[0]

    if len(no_compradas) == 0:
        no_compradas = np.random.choice(todas_categorias, size=2, replace=False)

    n_pos = len(compradas)
    n_neg = min(len(no_compradas), n_pos)
    seleccion = np.random.choice(no_compradas, size=n_neg, replace=False)

    for cat in seleccion:
        fila = cliente_info.copy()
        fila["item_category"] = cat
        fila["label"] = 0
        fila["item_id"] = f"FAKE_{np.random.randint(1000,9999)}"
        fila["item_title"] = "Fake item"
        fila["purchase_timestamp"] = ""
        fila["item_avg_rating"] = df["item_avg_rating"].mean()
        fila["item_num_ratings"] = df["item_num_ratings"].mean()
        fila["customer_item_views"] = np.random.randint(0, 3)
        no_compras.append(fila)

# === CONCATENAR COMPRAS Y NO COMPRAS ===
df_no = pd.DataFrame(no_compras)
df_balanced = pd.concat([df, df_no], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# === GUARDAR NUEVO DATASET ===
ruta_nuevo = r"C:\Users\paola\Downloads\ml25_BBB\src\ml25\datasets\customer_purchases\customer_purchases_train_balanced.csv"
df_balanced.to_csv(ruta_nuevo, index=False)

print("✅ Nuevo dataset balanceado creado correctamente.")
print(df_balanced["label"].value_counts())
print(df_balanced.sample(5)[["customer_id", "item_category", "label"]])
