import pandas as pd
from datetime import datetime
import re

ruta_csv = r"C:\Users\paola\Downloads\ml25_BBB\src\ml25\datasets\customer_purchases\customer_purchases_train_modify.csv"

# === Leer CSV original ===
dataFrame = pd.read_csv(ruta_csv)

# === Reemplazar valores en 'color' ===
dataFrame["color"] = dataFrame["color"].replace({
    "imgb.jpg": 1,
    "imgbl.jpg": 0.714,
    "imgg.jpg": 0.429,
    "imgp.jpg": -0.429,
    "imgw.jpg": -0.714,
    "imgo.jpg": 0.143,
    "imgy.jpg": -1,
    "imgr.jpg": -0.1
})

# === Codificar género ===
dataFrame["customer_gender"] = dataFrame["customer_gender"].replace({"female": 1, "male": 2}).fillna(3)

# === Calcular edad (a partir de la fecha de nacimiento) ===
columna_fecha = "edad"
dataFrame[columna_fecha] = pd.to_datetime(dataFrame[columna_fecha], errors="coerce")
hoy = datetime.today()
dataFrame[columna_fecha] = dataFrame[columna_fecha].apply(
    lambda x: hoy.year - x.year - ((hoy.month, hoy.day) < (x.month, x.day)) if pd.notnull(x) else None
)

# === Tiempo desde registro (en meses) ===
hoy2 = pd.Timestamp.now()
dataFrame["customer_signup_date"] = pd.to_datetime(dataFrame["customer_signup_date"], errors="coerce")
dataFrame["customer_signup_date"] = ((hoy2 - dataFrame["customer_signup_date"]).dt.days / 30.44).round(3)

# === Mapear categorías ===
mapa = {
    "jacket": 1.0,
    "blouse": 0.6,
    "jeans": 0.2,
    "skirt": -0.2,
    "shoes": -0.6,
    "dress": -1.0
}
dataFrame["item_category"] = dataFrame["item_category"].map(mapa)

# === Mantener solo adjetivos del título ===
adjectives = [
    "premium", "casual", "modern", "stylish",
    "exclusive", "elegant", "classic",
    "lightweight", "durable"
]
pattern = re.compile(r'\b(' + '|'.join(map(re.escape, adjectives)) + r')\b', flags=re.IGNORECASE)

def keep_adjectives(title):
    if pd.isna(title):
        return ""
    matches = [m.group(0) for m in pattern.finditer(str(title))]
    seen = set()
    result = []
    for m in matches:
        key = m.lower()
        if key not in seen:
            seen.add(key)
            result.append(m)
    return " ".join(result)

dataFrame["item_title"] = dataFrame["item_title"].apply(keep_adjectives)

# === Guardar sobre el mismo archivo ===
dataFrame.to_csv(ruta_csv, index=False)

print("✅ Archivo sobrescrito correctamente:", ruta_csv)
