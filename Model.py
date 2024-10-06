import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

import joblib

# Cargar los datos
ruta1 = 'c:/Users/Joselyn/Desktop/PERSONALES/CURSO CERTUS/DataOps/Trabajo_Final/Riesgo_de_credito.xlsx'

df = pd.read_excel(ruta1)

# 1. Cargar datos de nuevos clientes
Cartera_actual_vivela = pd.read_excel('c:/Users/Joselyn/Desktop/PERSONALES/CURSO CERTUS/DataOps/Trabajo_Final/cartera_actual_Vivela.xlsx')

print(df.columns)

# Preprocesamiento: Codificar variables categóricas
df = pd.get_dummies(df, columns=['genero', 'estado_civil', 'ocupacion'], drop_first=True)

# Dividir los datos en características (X) y variable objetivo (y)
X = df.drop(columns=['id_cliente', 'incumplimiento_pago'])
y = df['incumplimiento_pago']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar un modelo de regresión logística
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predicciones
y_pred = model.predict(X_test_scaled)

# Evaluar el modelo
print("Precisión:", accuracy_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Guardar el modelo entrenado
joblib.dump(model, 'modelo_credito.pkl')

print(len(df.columns))
print(df.columns)

print(Cartera_actual_vivela.columns)
print(len(Cartera_actual_vivela.columns))
