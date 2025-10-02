import pandas as pd
import os
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE / "../../datasets/customer_purchases/"


def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    df = pd.read_csv(fullfilename)
    return df


if __name__ == "__main__":
    
    #train_df = read_csv("customer_purchases_train")
    #print(train_df.info())
    #test_df = read_csv("customer_purchases_test")
    #print(test_df.columns)

    #print("----- Información del set de entrenamiento -----")
    #print(train_df.info())
    #print("\n----- Columnas del set de prueba -----")
    #print(test_df.columns)
    
    # 1 Cargar datos
    train_df = read_csv("customer_purchases_train")

    # 2 Convertir fechas a datetime
    train_df['customer_date_of_birth'] = pd.to_datetime(train_df['customer_date_of_birth'], errors='coerce')
    train_df['customer_signup_date'] = pd.to_datetime(train_df['customer_signup_date'], errors='coerce')
    train_df['purchase_timestamp'] = pd.to_datetime(train_df['purchase_timestamp'], errors='coerce')
    
    # 3 Crear columnas derivadas numéricas
    
    # Edad del cliente
    today = pd.to_datetime('today')
    train_df['customer_age'] = (today - train_df['customer_date_of_birth']).dt.days // 365
    
    # Días desde que el cliente se registró hasta la compra
    train_df['days_since_signup'] = (train_df['purchase_timestamp'] - train_df['customer_signup_date']).dt.days
    
    # Mes y día de la semana de la compra
    train_df['purchase_month'] = train_df['purchase_timestamp'].dt.month
    train_df['purchase_weekday'] = train_df['purchase_timestamp'].dt.weekday  # Lunes=0, Domingo=6
    
    # Ver las nuevas columnas y los primeros datos
    print("----- Primeras filas con columnas derivadas -----")
    print(train_df[['customer_age','days_since_signup','purchase_month','purchase_weekday']].head())


    # 4 Visualizaciones

    # Suponiendo que ya cargaste train_df y creaste columnas derivadas

    # Columnas a graficar
    numeric_cols = ['customer_age','days_since_signup','purchase_month','purchase_weekday']

    # 1 Histogramas para ver distribuciones

    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(train_df[col].dropna(), kde=True)  # dropna() ignora valores vacíos
        plt.title(f'Distribución de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.show()

    # 2 Boxplots para detectar valores atípicos
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=train_df[col].dropna())
        plt.title(f'Boxplot de {col}')
        plt.show()

