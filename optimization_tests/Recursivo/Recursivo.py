import time
import random
import uuid
import os
import sys
from collections import deque
from config import CAPACIDAD_MAXIMA, MAX_ITEMS, NUM_SAMPLES, FORCE_THRESHOLD

# Aumentamos el límite de recursión para evitar RecursionError en problemas grandes
sys.setrecursionlimit(10000)

#################################
# 1. Función recursiva tipo Knapsack con poda
#################################
def recursive_knapsack(items, capacity, index=0):
    """
    Resuelve el problema de la mochila 0/1 de forma recursiva con poda.
    
    Cada ítem es una tupla (id, valor, count) y se usa 'valor' para la suma.
    
    Retorna:
      - (suma, subset) donde 'suma' es la suma total de valores y
        'subset' es la lista de ítems seleccionados.
    
    Poda: si en alguna rama se alcanza exactamente la capacidad, se devuelve esa solución.
    """
    if capacity == 0 or index == len(items):
        return 0, []
    
    # Opción 1: No incluir el ítem actual
    value_without, subset_without = recursive_knapsack(items, capacity, index + 1)
    
    # Opción 2: Incluir el ítem actual (si cabe)
    current_item = items[index]
    ident, valor, count = current_item
    if valor <= capacity:
        value_with, subset_with = recursive_knapsack(items, capacity - valor, index + 1)
        value_with += valor
        subset_with = [current_item] + subset_with
        # Poda: si se alcanza exactamente la capacidad, devolvemos inmediatamente esta solución.
        if value_with == capacity:
            return value_with, subset_with
    else:
        value_with, subset_with = 0, []
    
    if value_with > value_without:
        return value_with, subset_with
    else:
        return value_without, subset_without

#################################
# 2. Función para generar elementos de la cola
#################################
def generar_cola(num_elementos):
    """
    Genera una cola (deque) de 'num_elementos' elementos.
    
    Cada elemento es una tupla (id, valor, count) donde:
      - id: un identificador único (generado con uuid4)
      - valor: entero aleatorio entre 1 y 12 (qBits)
      - count: contador de prioridad (inicia en 0)
    """
    cola = deque()
    for _ in range(num_elementos):
        identificador = str(uuid.uuid4())
        valor = random.randint(1, 12)
        count = 0
        cola.append((identificador, valor, count))
    return cola

#################################
# 3. Función de optimización recursiva de la cola
#################################
def optimizar_espacio_recursivo(queue, capacidad, forced_threshold=FORCE_THRESHOLD):
    """
    Optimiza la selección de elementos de la cola de forma recursiva.
    
    Procedimiento:
      1. Se fuerza la selección de los elementos cuyo count >= forced_threshold y que quepan en la capacidad.
      2. Sobre los elementos restantes se aplica el algoritmo recursivo (recursive_knapsack) para 
         hallar la mejor combinación sin exceder la capacidad restante.
    
    Retorna:
      - selected: lista de elementos seleccionados (tuplas (id, valor, count))
      - total_valor: suma total de los valores seleccionados.
      - nueva_cola: cola actualizada (sin los elementos seleccionados).
    """
    forced_selected = []
    capacidad_restante = capacidad
    remaining = []
    
    for item in queue:
        ident, valor, count = item
        if count >= forced_threshold and valor <= capacidad_restante:
            forced_selected.append(item)
            capacidad_restante -= valor
        else:
            remaining.append(item)
    
    rec_value, rec_subset = recursive_knapsack(remaining, capacidad_restante)
    
    selected = forced_selected + rec_subset
    total_valor = sum(item[1] for item in selected)
    
    # Se elimina de la cola original (comparando por id) los elementos que se han seleccionado.
    selected_ids = {item[0] for item in selected}
    nueva_cola = [item for item in queue if item[0] not in selected_ids]
    
    return selected, total_valor, nueva_cola

#################################
# 4. Función interactiva para procesar la cola recursivamente
#################################
def procesar_cola_recursivo(cola, capacidad, forced_threshold=FORCE_THRESHOLD):
    """
    Procesa la cola de forma interactiva usando el enfoque recursivo.
    
    Flujo:
      - Se pregunta al usuario cuántas veces procesar la cola.
         * Si se ingresa 0: se procesa iteraciones hasta vaciar la cola. En este modo, en **cada iteración**
           se añaden automáticamente 3 nuevos elementos a la cola.
         * Si se ingresa un número mayor que 0: se procesan ese número de iteraciones.
      - En cada iteración:
          * Se incrementa en 1 el contador (count) de cada elemento.
          * Se muestra el estado actual de la cola.
          * Se procesa la cola llamando a optimizar_espacio_recursivo.
      - Al finalizar las iteraciones, se pregunta si se desean añadir nuevos elementos manualmente.
      - En modo 0 (vaciado automático), se inyectan 3 nuevos elementos en cada iteración.
      - Además, si al final quedan elementos, la función se llama recursivamente para continuar el procesamiento.
    """
    iter_global = 0
    num_iteraciones = int(input("¿Cuántas veces deseas procesar la cola? (0 para vaciarla automáticamente): "))
    
    if num_iteraciones == 0:
        # Modo vaciado automático
        while cola:
            # Incrementar count en cada elemento
            cola = deque([(ident, valor, count + 1) for (ident, valor, count) in cola])
            iter_global += 1
            print(f"\nIteración {iter_global} - Estado de la cola:")
            for item in cola:
                ident, valor, count = item
                print(f"  Elemento {ident}: Valor = {valor}, Iteraciones = {count}")
            # Cada iteración, se inyectan 3 nuevos elementos automáticamente
            print("\n--- Se añaden 3 nuevos elementos a la cola automáticamente ---")
            nuevos = generar_cola(3)
            for elem in nuevos:
                cola.append(elem)
            selected, total_valor, nueva_cola = optimizar_espacio_recursivo(list(cola), capacidad, forced_threshold)
            print(f"\nProcesando cola en iteración {iter_global}:")
            print(f"  Cola actual (IDs): {[item[0] for item in nueva_cola]}")
            print(f"  Elementos seleccionados (IDs): {[item[0] for item in selected]}")
            print(f"  Valor total obtenido: {total_valor}")
            cola = deque(nueva_cola)
        print("La cola se ha vaciado.")
    else:
        for _ in range(num_iteraciones):
            cola = deque([(ident, valor, count + 1) for (ident, valor, count) in cola])
            iter_global += 1
            print(f"\nIteración {iter_global} - Estado de la cola:")
            for item in cola:
                ident, valor, count = item
                print(f"  Elemento {ident}: Valor = {valor}, Iteraciones = {count}")
            # Si se desea inyectar nuevos elementos en modo iterativo, se puede habilitar esta sección:
            # if iter_global % 3 == 0:
            #     print("\n--- Se añaden 3 nuevos elementos a la cola automáticamente ---")
            #     nuevos = generar_cola(3)
            #     for elem in nuevos:
            #         cola.append(elem)
            selected, total_valor, nueva_cola = optimizar_espacio_recursivo(list(cola), capacidad, forced_threshold)
            print(f"\nProcesando cola en iteración {iter_global}:")
            print(f"  Cola actual (IDs): {[item[0] for item in nueva_cola]}")
            print(f"  Elementos seleccionados (IDs): {[item[0] for item in selected]}")
            print(f"  Valor total obtenido: {total_valor}")
            cola = deque(nueva_cola)
        if cola:
            añadir = input("\n¿Deseas añadir nuevos elementos a la cola? (s/n): ").lower()
            if añadir == 's':
                nuevos = input("Introduce los pares id:valor separados por espacios: ").split()
                for par in nuevos:
                    try:
                        ident, valor_str = par.split(':')
                        valor = int(valor_str)
                        cola.append((ident, valor, 0))
                    except Exception as e:
                        print(f"Error al procesar '{par}': {e}")
            # Continuar procesando recursivamente si la cola aún contiene elementos
            procesar_cola_recursivo(cola, capacidad, forced_threshold)

#################################
# Programa principal
#################################
if __name__ == "__main__":
    # Se genera una cola inicial con 20 elementos

    inicio = time.time()  # Inicio del conteo de tiempo

    # Se genera una cola inicial con 20 elementos
    cola_inicial = generar_cola(20)
    print("\nProcesando cola de forma recursiva (sin redes neuronales)...")
    
    procesar_cola_recursivo(cola_inicial, CAPACIDAD_MAXIMA, forced_threshold=FORCE_THRESHOLD)

    # Fin del conteo de tiempo
    fin = time.time()
    print(f"\n⏱️ Tiempo total de ejecución: {fin - inicio:.2f} segundos.")
