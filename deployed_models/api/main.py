from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Inicializar la app
app = FastAPI()

# 2. Cargar el modelo (se carga una sola vez al arrancar)
with open("modelo.pkl", "rb") as f:
    modelo = joblib.load(f)

# 3. Definir la estructura de los datos de entrada
class InputData(BaseModel):
    PUlocationID: float
    DOlocationID: float
    trip_distance: float

# 4. Crear el endpoint de predicción
@app.post("/predict")
def predict(data: InputData):
    # 1. Convertimos el objeto de Pydantic a un diccionario
    features = data.dict() 
    
    # 2. El pipeline con DictVectorizer exige recibir una lista de diccionarios
    score = modelo.predict([features])[0]
    
    return {"score": float(score)}