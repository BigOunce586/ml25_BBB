def obtener_primer_elemento(lista):
    if lista:
        return lista[0]  # Siempre 1 operación
    return None

import time
lista = [12,123,1234,12345]

inicio = time.time()
obtener_primer_elemento(lista)
print(f"Tiempo de ejecucion = {time.time()-inicio}")
