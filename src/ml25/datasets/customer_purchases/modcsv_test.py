import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- 1. Cargar test original ---
ruta_csv_test = r"C:\Users\paola\Downloads\ml25_BBB\src\ml25\datasets\customer_purchases\customer_purchases_test.csv"
df_test = pd.read_csv(ruta_csv_test)

# --- 2. Limpiar y convertir customer_id ---
df_test['customer_id'] = df_test['customer_id'].str.replace('CUST_', '').astype(int)

# --- 3. Fechas y cálculo de edad / tenure ---
df_test['customer_date_of_birth'] = pd.to_datetime(df_test['customer_date_of_birth'], errors='coerce')
df_test['customer_signup_date'] = pd.to_datetime(df_test['customer_signup_date'], errors='coerce')
reference_date = pd.to_datetime('2025-09-21')

df_test['edad'] = ((reference_date - df_test['customer_date_of_birth']).dt.days / 365).fillna(0).astype(int)
df_test['tenure_months'] = ((reference_date - df_test['customer_signup_date']).dt.days / 30).fillna(0).astype(int)

# --- 4. Codificar género ---
le_gender = LabelEncoder()
df_test['customer_gender'] = df_test['customer_gender'].fillna('unknown')
df_test['customer_gender'] = le_gender.fit_transform(df_test['customer_gender'].astype(str))

# --- 5. Codificar color ---
color_prefixes = ['imgb', 'imgbl', 'imgg', 'imgp', 'imgw', 'imgo', 'imgy', 'imgr']
color_names = ['blue', 'black', 'green', 'pink', 'white', 'orange', 'yellow', 'red']
df_test['color'] = df_test['item_img_filename'].str[:4]
le_color = LabelEncoder()
df_test['color'] = le_color.fit_transform(df_test['color'].astype(str))

# --- 6. Adjetivos del título ---
adjetivos = ["premium", "casual", "modern", "stylish", "exclusive",
             "elegant", "classic", "lightweight", "durable"]

def extract_adjectives(title):
    title = str(title).lower()
    found = [adj for adj in adjetivos if adj in title]
    return found if found else []

df_test['title_adjectives'] = df_test['item_title'].apply(extract_adjectives)

# --- 7. Codificar categoría ---
categories_list = ["jacket", "blouse", "jeans", "skirt", "shoes", "dress"]
le_category = LabelEncoder()
le_category.fit(categories_list)
df_test['item_category_encoded'] = le_category.transform(df_test['item_category'])

# --- 8. Crear columnas agregadas igual que en train ---
rows_test = []

for _, row in df_test.iterrows():
    temp = {
        'customer_id': row['customer_id'],
        'item_category': row['item_category_encoded'],
        'purchases_in_category': 0,  # en test no sabemos si compró
        'total_spent_in_category': row['item_price'],
        'edad': row['edad'],
        'genero': row['customer_gender'],
        'tenure_months': row['tenure_months'],
        'total_purchases': 1  # cada fila es un producto
    }

    # --- 🔹 Generar columnas de color (color_count_...) igual que en train ---
    for color in color_names:
        if row['item_img_filename'].startswith('img' + color[0]):  # Ej: imgb = blue
            temp[f'color_count_{color}'] = 1
        else:
            temp[f'color_count_{color}'] = 0

    # --- 🔹 Adjetivos igual que en train ---
    for adj in adjetivos:
        temp[f'adj_count_{adj}'] = 1 if adj in row['title_adjectives'] else 0

    rows_test.append(temp)

# --- 9. Crear DataFrame final ---
model_test_df = pd.DataFrame(rows_test)

# --- 10. Eliminar columnas que no van ---
# En el test no se necesita avg_spent_in_category
if 'avg_spent_in_category' in model_test_df.columns:
    model_test_df = model_test_df.drop(columns=['avg_spent_in_category'])

# --- 11. Guardar CSV final con el formato del train ---
output_path = "customer_purchase_test_modify.csv"
model_test_df.to_csv(output_path, index=False)

print(f"✅ Archivo '{output_path}' generado correctamente.")
print(f"Filas: {model_test_df.shape[0]}, Columnas: {model_test_df.shape[1]}")
model_test_df.head()
