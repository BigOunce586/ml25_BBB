import pandas as pd
from datetime import datetime
import re

ruta_csv = r"C:\Users\paola\Downloads\ml25_BBB\src\ml25\datasets\customer_purchases\customer_purchases_test_modify.csv"

dataFrame = pd.read_csv(ruta_csv)

dataFrame["color"] = dataFrame["color"].replace("imgb.jpg", 1)
dataFrame["color"] = dataFrame["color"].replace("imgbl.jpg", 0.714)
dataFrame["color"] = dataFrame["color"].replace("imgg.jpg", 0.429)
dataFrame["color"] = dataFrame["color"].replace("imgp.jpg", -0.429)
dataFrame["color"] = dataFrame["color"].replace("imgw.jpg", -0.714)
dataFrame["color"] = dataFrame["color"].replace("imgo.jpg", 0.143)
dataFrame["color"] = dataFrame["color"].replace("imgy.jpg",-1)
dataFrame["color"] = dataFrame["color"].replace("imgr.jpg",-0.1)

dataFrame["customer_gender"] = dataFrame["customer_gender"].replace("female", 1)
dataFrame["customer_gender"] = dataFrame["customer_gender"].replace("male",2)
dataFrame["customer_gender"] = dataFrame["customer_gender"].fillna(3)

#columna_fecha = "edad" 
#dataFrame[columna_fecha] = pd.to_datetime(dataFrame[columna_fecha], errors="coerce")
#hoy = datetime.today()
#dataFrame[columna_fecha] = dataFrame[columna_fecha].apply(
#    lambda x: hoy.year - x.year - ((hoy.month, hoy.day) < (x.month, x.day)) if pd.notnull(x) else None
#)

# meses de sign up hoy2 = pd.Timestamp.now()
#dataFrame["customer_signup_date"] = pd.to_datetime(dataFrame["customer_signup_date"], errors="coerce")
#dataFrame["customer_signup_date"] = ((hoy2 - dataFrame["customer_signup_date"]).dt.days / 30.44).round(3) # 30.44 es el promedio de días por mes

mapa = {
    "jacket": 1.0,
    "blouse": 0.6,
    "jeans": 0.2,
    "skirt": -0.2,
    "shoes": -0.6,
    "dress": -1.0
}

dataFrame["item_category"] = dataFrame["item_category"].map(mapa)
dataFrame.to_csv("customer_purchases_train.csv", index=False)

adjectives = [
    "premium",
    "casual",
    "modern",
    "stylish",
    "exclusive",
    "elegant",
    "classic",
    "lightweight",
    "durable"
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
dataFrame = pd.read_csv(ruta_csv)
dataFrame["item_title"] = dataFrame["item_title"].apply(keep_adjectives)


dataFrame.to_csv(ruta_csv, index=False)
