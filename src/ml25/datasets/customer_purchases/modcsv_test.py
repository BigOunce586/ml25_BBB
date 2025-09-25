import pandas as pd
from datetime import datetime
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

columna_fecha = "customer_date_of_birth" 
dataFrame[columna_fecha] = pd.to_datetime(dataFrame[columna_fecha], errors="coerce")
hoy = datetime.today()
dataFrame[columna_fecha] = dataFrame[columna_fecha].apply(
    lambda x: hoy.year - x.year - ((hoy.month, hoy.day) < (x.month, x.day)) if pd.notnull(x) else None
)
dataFrame.to_csv(ruta_csv, index=False)
