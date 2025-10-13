import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- Cargar datasets originales ---
train = pd.read_csv("customer_purchases_train.csv")
test = pd.read_csv("customer_purchases_test.csv")

# ==========================================================
# 1️⃣ LIMPIEZA Y FEATURE ENGINEERING
# ==========================================================
def clean_dataset(df):
    df = df.copy()

    # Tipos correctos
    df["customer_id"] = df["customer_id"].astype(str)
    df["item_category"] = df["item_category"].astype(str)

    # Fechas
    df["customer_date_of_birth"] = pd.to_datetime(df["customer_date_of_birth"], errors="coerce")
    df["customer_signup_date"] = pd.to_datetime(df["customer_signup_date"], errors="coerce")

    # Edad y tenure
    df["edad"] = (pd.Timestamp.now() - df["customer_date_of_birth"]).dt.days // 365
    df["tenure_months"] = (pd.Timestamp.now() - df["customer_signup_date"]).dt.days // 30

    # Género (-1 si falta)
    df["customer_gender"] = df["customer_gender"].fillna("-1").astype(str).str.lower()
    genero_map = {"male": 1, "female": 0, "-1": -1}
    df["genero"] = df["customer_gender"].map(genero_map).fillna(-1).astype(int)

    # Color desde el nombre del archivo
    df["color"] = df["item_img_filename"].str.extract(r'([a-zA-Z]+)\.')[0].str.lower()

    # Adjetivos
    adjetivos = ["premium", "casual", "modern", "stylish", "exclusive",
                 "elegant", "classic", "lightweight", "durable"]
    for adj in adjetivos:
        df[f"adj_{adj}"] = df["item_title"].str.lower().str.contains(adj, na=False).astype(int)

    return df


# ==========================================================
# 2️⃣ AGREGAR FEATURES POR CLIENTE Y CATEGORÍA
# ==========================================================
def build_features(df, le_cat=None):
    agg = df.groupby(["customer_id", "item_category"]).agg(
        purchases_in_category=("item_id", "count"),
        total_spent_in_category=("item_price", "sum"),
        avg_spent_in_category=("item_price", "mean"),
        avg_rating=("item_avg_rating", "mean"),
        edad=("edad", "first"),
        genero=("genero", "first"),
        tenure_months=("tenure_months", "first")
    ).reset_index()

    # Contar colores
    colors = ["blue", "black", "green", "pink", "white", "orange", "yellow", "red"]
    for c in colors:
        color_count = df[df["color"] == c].groupby("customer_id")["color"].count()
        agg[f"color_count_{c}"] = agg["customer_id"].map(color_count).fillna(0)

    # Contar adjetivos
    adjetivos = ["premium", "casual", "modern", "stylish", "exclusive",
                 "elegant", "classic", "lightweight", "durable"]
    for adj in adjetivos:
        adj_count = df[df[f"adj_{adj}"] == 1].groupby("customer_id")[f"adj_{adj}"].count()
        agg[f"adj_count_{adj}"] = agg["customer_id"].map(adj_count).fillna(0)

    # Total de compras por cliente
    total_purchases = df.groupby("customer_id")["item_id"].count()
    agg["total_purchases"] = agg["customer_id"].map(total_purchases).fillna(0)

    # --- Codificar categoría a número ---
    if le_cat is None:
        le_cat = LabelEncoder()
        agg["item_category_encoded"] = le_cat.fit_transform(agg["item_category"])
    else:
        agg["item_category_encoded"] = le_cat.transform(agg["item_category"])

    return agg, le_cat


# ==========================================================
# 3️⃣ CONSTRUIR DATASETS MODIFICADOS
# ==========================================================
train = clean_dataset(train)
test = clean_dataset(test)

train_mod, le_cat = build_features(train)
test_mod, _ = build_features(test, le_cat)

# ==========================================================
# 4️⃣ AGREGAR CLIENTES/CATEGORÍAS SIN COMPRAS (LABEL 0)
# ==========================================================
# Todos los IDs y categorías posibles
all_ids = train["customer_id"].unique()
all_cats = train_mod["item_category"].unique()

# Crear combinaciones completas de cliente + categoría
all_pairs = pd.MultiIndex.from_product([all_ids, all_cats], names=["customer_id", "item_category"]).to_frame(index=False)

# Combinar con las existentes
train_full = all_pairs.merge(train_mod, on=["customer_id", "item_category"], how="left")

# Rellenar compras vacías con 0
fill_cols = ["purchases_in_category", "total_spent_in_category", "avg_spent_in_category", "avg_rating"]
for c in fill_cols:
    train_full[c] = train_full[c].fillna(0)

# Rellenar datos de cliente con promedios
for c in ["edad", "genero", "tenure_months"]:
    train_full[c] = train_full[c].fillna(train_mod[c].mean())

# Rellenar colores y adjetivos
for c in [col for col in train_mod.columns if "color_count" in col or "adj_count" in col or c == "total_purchases"]:
    train_full[c] = train_full[c].fillna(0)

# Etiqueta: 1 si compró (purchases_in_category > 0), 0 si no
train_full["label"] = (train_full["purchases_in_category"] > 0).astype(int)

# ==========================================================
# 5️⃣ GUARDAR RESULTADOS
# ==========================================================
train_full.to_csv("customer_purchase_train_modify2.csv", index=False)
test_mod.to_csv("customer_purchase_test_modify2.csv", index=False)

print("✅ Archivos creados correctamente:")
print("- customer_purchase_train_modify.csv (balanceado con 0 y 1)")
print("- customer_purchase_test_modify.csv (categoría numérica)")
