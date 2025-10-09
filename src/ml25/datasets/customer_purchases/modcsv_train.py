import pandas as pd
from datetime import datetime
import re

ruta_csv = r"C:\Users\paola\Downloads\ml25_BBB\src\ml25\datasets\customer_purchases\customer_purchases_train_modify.csv"

dataFrame = pd.read_csv(ruta_csv)

dataFrame["color"] = dataFrame["color"].replace({
 "imgb.jpg": 1,
 "imgbl.jpg": 2,
 "imgg.jpg": 3,
 "imgp.jpg": 4,
  "imgw.jpg": 5,
  "imgo.jpg": 6,
  "imgy.jpg": 7,
 "imgr.jpg": 8
})

dataFrame["customer_gender"] = dataFrame["customer_gender"].replace({"female": 1, "male": 2}).fillna(0)

# columna_fecha = "edad"
# dataFrame[columna_fecha] = pd.to_datetime(dataFrame[columna_fecha], errors="coerce")
# hoy = datetime.today()
# dataFrame[columna_fecha] = dataFrame[columna_fecha].apply(
#     lambda x: hoy.year - x.year - ((hoy.month, hoy.day) < (x.month, x.day)) if pd.notnull(x) else None
# )

# hoy2 = pd.Timestamp.now()
# dataFrame["customer_signup_date"] = pd.to_datetime(dataFrame["customer_signup_date"], errors="coerce")
# dataFrame["customer_signup_date"] = ((hoy2 - dataFrame["customer_signup_date"]).dt.days / 30.44).round(3)

mapa = {
    "jacket": 1.0,
    "blouse": 2,
    "jeans": 3,
    "skirt": 4,
    "shoes": 5,
    "dress": 6,
}
dataFrame["item_category"] = dataFrame["item_category"].map(mapa)

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

dataFrame.to_csv(ruta_csv, index=False)
