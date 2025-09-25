import pandas as pd

ruta_csv = r"C:\Users\paola\Downloads\ml25_BBB\src\ml25\datasets\customer_purchases\customer_purchases_train_modify.csv"

dataFrame = pd.read_csv(ruta_csv)

# Modificar la columna 'color'
dataFrame["color"] = dataFrame["color"].replace("imgp.jpg", -0.429)

# Guardar en el mismo archivo
dataFrame.to_csv(ruta_csv, index=False)
