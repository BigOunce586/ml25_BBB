import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- Cargar CSV de test ---
ruta_csv_test = r"C:\Users\paola\Downloads\ml25_BBB\src\ml25\datasets\customer_purchases\customer_purchases_test.csv"
df_test = pd.read_csv(ruta_csv_test)

# --- Limpiar customer_id ---
df_test['customer_id'] = df_test['customer_id'].str.replace('CUST_', '').astype(int)

# --- Convertir fechas ---
df_test['customer_date_of_birth'] = pd.to_datetime(df_test['customer_date_of_birth'], errors='coerce')
df_test['customer_signup_date'] = pd.to_datetime(df_test['customer_signup_date'], errors='coerce')
reference_date = pd.to_datetime('2025-09-21')

# --- Edad y tenure ---
df_test['edad'] = ((reference_date - df_test['customer_date_of_birth']).dt.days / 365).fillna(0).astype(int)
df_test['tenure_months'] = ((reference_date - df_test['customer_signup_date']).dt.days / 30).fillna(0).astype(int)

# --- Gender codificado automáticamente ---
le_gender = LabelEncoder()
df_test['customer_gender'] = df_test['customer_gender'].fillna('unknown')
df_test['customer_gender'] = le_gender.fit_transform(df_test['customer_gender'].astype(str))

# --- Colores codificados automáticamente ---
color_prefixes = ['imgb', 'imgbl', 'imgg', 'imgp', 'imgw', 'imgo', 'imgy', 'imgr']
color_names = ['blue', 'black', 'green', 'pink', 'white', 'orange', 'yellow', 'red']
le_color = LabelEncoder()
df_test['color'] = df_test['item_img_filename'].str[:4]
df_test['color'] = le_color.fit_transform(df_test['color'].astype(str))

# --- Extraer adjetivos del título ---
adjetivos = ["premium", "casual", "modern", "stylish", "exclusive",
             "elegant", "classic", "lightweight", "durable"]

def extract_adjectives(title):
    title = str(title).lower()
    found = [adj for adj in adjetivos if adj in title]
    return found if found else []

df_test['title_adjectives'] = df_test['item_title'].apply(extract_adjectives)

# --- Categorías fijas y codificación ---
categories_list = ["jacket", "blouse", "jeans", "skirt", "shoes", "dress"]
le_category = LabelEncoder()
le_category.fit(categories_list)

# --- Crear filas por cliente × categoría ---
rows_test = []
for cid in df_test['customer_id'].unique():
    df_cust = df_test[df_test['customer_id'] == cid]
    for cat_name in categories_list:
        cat_num = le_category.transform([cat_name])[0]
        df_cat = df_cust[df_cust['item_category'] == cat_name]
        
        temp = {'customer_id': cid, 'item_category': cat_num}
        
        # Compras en esta categoría
        temp['purchases_in_category'] = len(df_cat)
        temp['total_spent_in_category'] = df_cat['item_price'].sum() if not df_cat.empty else 0
        temp['avg_spent_in_category'] = df_cat['item_price'].mean() if not df_cat.empty else 0
        
        # Cantidad por color (fijas 8 columnas)
        for i, color in enumerate(color_names):
            df_color = df_cat[df_cat['color'] == i]
            temp[f'color_count_{color}'] = len(df_color)
        
        # Cantidad por adjetivo
        for adj in adjetivos:
            df_adj = df_cat[df_cat['title_adjectives'].apply(lambda x: adj in x)]
            temp[f'adj_count_{adj}'] = len(df_adj)
        
        # Stats globales del cliente
        temp['edad'] = df_cust['edad'].iloc[0]
        temp['genero'] = df_cust['customer_gender'].iloc[0]
        temp['tenure_months'] = df_cust['tenure_months'].iloc[0]
        temp['total_purchases'] = len(df_cust)
        
        rows_test.append(temp)

# --- Crear DataFrame final de test ---
model_test_df = pd.DataFrame(rows_test)

# --- Guardar CSV final con nombres de colores legibles ---
model_test_df.to_csv("customer_profiles_test_cliente_categoria_final_named.csv", index=False)
model_test_df.head()
