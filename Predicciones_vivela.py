import pandas as pd

import joblib

# Cargar el modelo
model = joblib.load('modelo_credito.pkl')

# 1. Cargar datos de nuevos clientes
Cartera_actual_vivela = pd.read_excel('cc:/Users/Joselyn/Trabajo_Final/cartera_actual_Vivela.xlsx')

# 2. Preprocesamiento
Cartera_actual_vivela = pd.get_dummies(Cartera_actual_vivela, columns=['genero', 'estado_civil', 'ocupacion'], drop_first=True)

# 3. Predicciones
predicciones = model.predict(Cartera_actual_vivela)

# 4. Agregar resultados al DataFrame
Cartera_actual_vivela['Predicción'] = predicciones

# 5. Guardar resultados en un archivo Excel
output_path = 'c:/Users/Joselyn/Trabajo_Final/Credit_Scoring_Vivela.xlsx'
Cartera_actual_vivela.to_excel(output_path, index=False)

# 6. Mostrar confirmación
print(f"Predicciones guardadas exitosamente en {output_path}")

