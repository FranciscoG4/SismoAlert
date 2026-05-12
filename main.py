import joblib
import pandas as pd
import hashlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simulamos la carga del modelo para la presentación
modelo = None
try:
    modelo = joblib.load('modelo_sismos.pkl')
except:
    print("Aviso: Modo simulación activo (modelo.pkl no encontrado)")

class SismoSignal(BaseModel):
    id: str
    lat: float
    lng: float
    mag: float
    place: str

@app.post("/filter_signals")
async def filter_signals(signals: List[SismoSignal]):
    global modelo
    resultados = []
    
    for s in signals:
        # 1. PERSISTENCIA TOTAL: Generamos un valor fijo basado en el ID único del sismo
        # Esto asegura que la decisión de la IA sea permanente para este evento.
        id_hash = int(hashlib.md5(s.id.encode()).hexdigest(), 16)
        
        # 2. LÓGICA DE CLASIFICACIÓN (DETERMINISTA)
        # Aquí simulamos que la IA detecta anomalías en el 20% de los casos.
        # Una vez que se asigna 'True' o 'False' a un ID, nunca cambiará.
        es_anomalia = (id_hash % 100) < 20 

        # 3. PREDICCIÓN DE MAGNITUD
        if modelo:
            input_df = pd.DataFrame([[s.lat, s.lng]], columns=['latitud', 'longitud'])
            mag_ia = modelo.predict(input_df)[0]
        else:
            # Si no hay modelo, la Mag IA es la real (para no inventar datos)
            mag_ia = s.mag

        resultados.append({
            "id": s.id,
            "lat": s.lat,
            "lng": s.lng,
            "mag_original": s.mag,
            "mag_ia": round(float(mag_ia), 2),
            "place": s.place,
            "es_anomalia": es_anomalia
        })
    
    return resultados

if __name__ == "__main__":
    import uvicorn
    # Ejecución en puerto 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)