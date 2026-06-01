from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import io

app = FastAPI()

# Cargar el modelo una sola vez en memoria
modelo = joblib.load("modelo.pkl")

# Estructura para la entrada manual (Individual)
class InputData(BaseModel):
    PULocationID: int
    DOLocationID: int
    trip_distance: float

# Opción 1: Predicción Individual
@app.post("/predict")
def predict(data: InputData):
    features = data.dict()
    # El DictVectorizer del curso exige una lista de diccionarios
    score = modelo.predict([features])[0]
    return {"score": float(score)}

# Opción 2: Predicción por Lotes (Excel)
@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))
    
    # 1. Normalizar el nombre de la columna si viene mal desde el Excel
    if "Trip Distance" in df.columns:
        df.rename(columns={"Trip Distance": "trip_distance"}, inplace=True)
    
    # 2. Forzar el casteo a string para que DictVectorizer haga el One-Hot Encoding
    df['PULocationID'] = df['PULocationID'].astype(str)
    df['DOLocationID'] = df['DOLocationID'].astype(str)
    
    # El resto del código se mantiene idéntico
    records = df.to_dict(orient="records")
    predicciones = modelo.predict(records)
    
    df["score"] = predicciones
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    
    headers = {'Content-Disposition': f'attachment; filename="scored_{file.filename}"'}
    return StreamingResponse(
        output, 
        headers=headers, 
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )