import pandas as pd

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


dataFrame.to_csv(ruta_csv, index=False)
