# Proyecto 1 - Identificando patrones de compra   
**Curso:** Aprendizaje de Máquina  
**Equipo:** ml25_BBB  
**Integrantes:**  
- Kevin Ledezma – Análisis de datos  
- Ana Paola Moguel – Modelado y modificación del CSV  

---

## Descripción
El proyecto busca predecir si un cliente existente comprará un nuevo producto de una tienda de ropa, usando datos históricos de compras. Se trabajó con un conjunto de datos multimodal que incluía información de clientes y productos.

---

## Modelos y metodología
Se probaron dos modelos:
- **Random Forest**  
- **XGBoost**

Ambos fueron entrenados después de limpiar y balancear los datos.  
El modelo **Random Forest fue el que obtuvo el mejor desempeño**.  

---

## Resultados
- **Accuracy (Random Forest):** 0.72  
- **Puntaje Kaggle:** 0.80  
- Se utilizó la **matriz de confusión** para analizar el rendimiento del modelo.

---

## Archivos principales
- `analisis_exploratorio.py` – análisis inicial y limpieza  
- `random_forest.py` – entrenamiento y validación  
- `predictions_random_forest.csv` – exportación de predicciones  
- `random_forest_model.pkl` – modelo 
- `README.md` – descripción del proyecto  
- `Presentacion de proyecto 1.pdf` – presentación del proyecto  

---

## Conclusión
El proyecto permitió entender mejor cómo preparar y balancear datos reales antes del modelado.  
El **Random Forest** mostró buena capacidad para generalizar y fue la mejor opción del equipo.  
Agregar los ejemplos negativos al dataset fue clave para mejorar la precisión del modelo.
