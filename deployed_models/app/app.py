import streamlit as st
import requests

st.title("Plataforma de Scoring - MLOps")

# Crear las dos opciones de interfaz
tab1, tab2 = st.tabs(["📝 Entrada Manual", "📊 Carga por Lotes (Excel)"])

# --- PESTAÑA 1: ENTRADA MANUAL ---
with tab1:
    st.header("Calcular registro único")
    
    pu_id = st.number_input("PULocationID", value=151, step=1)
    do_id = st.number_input("DOLocationID", value=151, step=1)
    distance = st.number_input("Trip Distance", value=3.1, step=0.1)

    if st.button("Calcular Score Individual"):
        payload = {
            "PULocationID": int(pu_id),
            "DOLocationID": int(do_id),
            "trip_distance": float(distance)
        }
        
        response = requests.post("http://api:8000/predict", json=payload)
        
        if response.status_code == 200:
            st.success(f"Score calculado: {response.json()['score']:.2f}")
        else:
            st.error(f"Error en el backend. Código: {response.status_code}")

# --- PESTAÑA 2: CARGA POR LOTES ---
with tab2:
    st.header("Procesar archivo de clientes")
    st.info("Asegurate de que el archivo tenga las columnas: PULocationID, DOLocationID, trip_distance")
    
    archivo_subido = st.file_uploader("Subí tu archivo .xlsx", type=["xlsx"])

    if archivo_subido is not None:
        if st.button("Procesar Excel"):
            with st.spinner("Calculando scores para todo el archivo..."):
                files = {
                    "file": (archivo_subido.name, archivo_subido.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                }
                
                response = requests.post("http://api:8000/predict_batch", files=files)
                
                if response.status_code == 200:
                    st.success("¡Archivo procesado con éxito!")
                    st.download_button(
                        label="📥 Descargar Excel con Resultados",
                        data=response.content,
                        file_name=f"resultados_{archivo_subido.name}",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.error(f"Error al procesar el archivo. Código: {response.status_code}")