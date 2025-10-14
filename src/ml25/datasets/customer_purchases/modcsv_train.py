import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- Cargar CSV de train ---
ruta_csv = r"C:\Users\paola\Downloads\ml25_BBB\src\ml25\datasets\customer_purchases\customer_purchases_train.csv"
df = pd.read_csv(ruta_csv)

# --- Limpiar customer_id ---
df['customer_id'] = df['customer_id'].str.replace('CUST_', '').astype(int)

# --- Convertir fechas ---
df['customer_date_of_birth'] = pd.to_datetime(df['customer_date_of_birth'], errors='coerce')
df['customer_signup_date'] = pd.to_datetime(df['customer_signup_date'], errors='coerce')
reference_date = pd.to_datetime('2025-09-21')

# --- Edad y tenure ---
df['edad'] = ((reference_date - df['customer_date_of_birth']).dt.days / 365).fillna(0).astype(int)
df['tenure_months'] = ((reference_date - df['customer_signup_date']).dt.days / 30).fillna(0).astype(int)

# --- Gender codificado automáticamente ---
le_gender = LabelEncoder()
df['customer_gender'] = df['customer_gender'].fillna('unknown')
df['customer_gender'] = le_gender.fit_transform(df['customer_gender'].astype(str))

# --- Colores codificados automáticamente ---
# Lista de colores fijos (8)
color_prefixes = ['imgb', 'imgbl', 'imgg', 'imgp', 'imgw', 'imgo', 'imgy', 'imgr']
color_names = ['blue', 'black', 'green', 'pink', 'white', 'orange', 'yellow', 'red']

le_color = LabelEncoder()
df['color'] = df['item_img_filename'].str[:4]
df['color'] = le_color.fit_transform(df['color'].astype(str))

# --- Extraer adjetivos del título ---
adjetivos = ["premium", "casual", "modern", "stylish", "exclusive",
             "elegant", "classic", "lightweight", "durable"]

def extract_adjectives(title):
    title = str(title).lower()
    found = [adj for adj in adjetivos if adj in title]
    return found if found else []

df['title_adjectives'] = df['item_title'].apply(extract_adjectives)

# --- Categorías fijas y codificación ---
categories_list = ["jacket", "blouse", "jeans", "skirt", "shoes", "dress"]
le_category = LabelEncoder()
le_category.fit(categories_list)

# --- Generar filas por cliente × categoría ---
rows = []
for cid in df['customer_id'].unique():
    df_cust = df[df['customer_id'] == cid]
    for cat_name in categories_list:
        cat_num = le_category.transform([cat_name])[0]
        df_cat = df_cust[df_cust['item_category'] == cat_name]
        
        temp = {'customer_id': cid, 'item_category': cat_num}
        
        # Compras en esta categoría
        temp['purchases_in_category'] = len(df_cat)
        temp['total_spent_in_category'] = df_cat['item_price'].sum() if not df_cat.empty else 0
        temp['avg_spent_in_category'] = df_cat['item_price'].mean() if not df_cat.empty else 0
        temp['label'] = 1 if not df_cat.empty else 0
        
        # Cantidad por color (fijas 8 columnas)
        for i, color in enumerate(color_names):
            df_color = df_cat[df_cat['color'] == i]
            temp[f'color_count_{color}'] = len(df_color)
        
        # Cantidad por adjetivo (todas)
        for adj in adjetivos:
            df_adj = df_cat[df_cat['title_adjectives'].apply(lambda x: adj in x)]
            temp[f'adj_count_{adj}'] = len(df_adj)
        
        # Stats globales del cliente
        temp['edad'] = df_cust['edad'].iloc[0]
        temp['genero'] = df_cust['customer_gender'].iloc[0]
        temp['tenure_months'] = df_cust['tenure_months'].iloc[0]
        temp['total_purchases'] = len(df_cust)
        
        rows.append(temp)

# --- Crear DataFrame final ---
model_df = pd.DataFrame(rows)

# --- Guardar CSV final con nombres de colores legibles ---
model_df.to_csv("customer_purchase_train_modify.csv", index=False)
model_df.head()
