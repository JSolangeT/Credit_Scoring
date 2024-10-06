import pandas as pd

import joblib

# Cargar el modelo
model = joblib.load('modelo_credito.pkl')

# 1. Cargar datos de nuevos clientes

nuevos_clientes_df = pd.read_excel('c:/Users/Joselyn/Trabajo_Final/nuevos_clientes_prueba.xlsx')


# 2. Preprocesamiento
nuevos_clientes_df = pd.get_dummies(nuevos_clientes_df, columns=['genero', 'estado_civil', 'ocupacion'], drop_first=True)

# 3. Predicciones
predicciones = model.predict(nuevos_clientes_df)

# 4. Agregar resultados al DataFrame
nuevos_clientes_df['Predicción'] = predicciones

# 5. Mostrar resultados
print(nuevos_clientes_df[['id_cliente', 'Predicción']])  # Muestra ID y Predicción
