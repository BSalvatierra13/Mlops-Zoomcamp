import streamlit as st
import requests

st.title("Scoring Model - UI")

# Tomar inputs del usuario
f1 = st.number_input("PUlocationID", value=0.0)
f2 = st.number_input("DOlocationID", value=0.0)
f3 = st.number_input("trip_distance", value=0.0)

# Botón para predecir
if st.button("Calcular Score"):
    # Armar el paquete de datos
    payload = {
        "PUlocationID": f1,
        "DOlocationID": f2,
        "trip_distance": f3
    }
    
    # Llamar a la API (FastAPI)
    # Nota: 'api' es el nombre del contenedor en Docker
    response = requests.post("http://api:8000/predict", json=payload)
    
    if response.status_code == 200:
        resultado = response.json()
        st.success(f"El score calculado es: {resultado['score']}")
    else:
        st.error("Error al conectarse con el modelo.")