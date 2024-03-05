import os

# Obtener el diccionario de variables de entorno
variables_entorno = os.environ

# Iterar sobre el diccionario e imprimir cada variable de entorno
for variable, valor in variables_entorno.items():
    print(f"{variable}: {valor}")
