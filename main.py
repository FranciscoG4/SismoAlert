from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import sqlite3
import joblib
import numpy as np

app = FastAPI()

# Permitir que el mapa (HTML) se conecte al servidor
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_NAME = "sismos_history.db"
# URL de la semana para tener datos suficientes para la IA
USGS_URL = USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"

# --- CARGA DE INTELIGENCIA ARTIFICIAL ---
try:
    modelo_ia = joblib.load('modelo_sismos.pkl')
    print("🧠 Inteligencia Artificial cargada correctamente.")
except:
    modelo_ia = None
    print("⚠️ No se encontró el modelo de IA. Ejecutá primero entrenar_ia.py")

# Crear la base de datos si no existe
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sismos (
            id TEXT PRIMARY KEY,
            mag REAL,
            place TEXT,
            time INTEGER,
            lat REAL,
            lng REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.get("/sismos")
async def get_sismos():
    async with httpx.AsyncClient() as client:
        response = await client.get(USGS_URL)
        data = response.json()
    
    sismos_procesados = []
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    for feature in data['features']:
        prop = feature['properties']
        geom = feature['geometry']
        
        sismo = {
            "id": feature['id'],
            "mag": prop['mag'],
            "place": prop['place'],
            "time": prop['time'],
            "lat": geom['coordinates'][1],
            "lng": geom['coordinates'][0]
        }
        
        # Guardar en la base de datos para que la IA siga aprendiendo
        cursor.execute('''
            INSERT OR IGNORE INTO sismos (id, mag, place, time, lat, lng)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (sismo['id'], sismo['mag'], sismo['place'], sismo['time'], sismo['lat'], sismo['lng']))
        
        sismos_procesados.append(sismo)

    conn.commit()
    conn.close()
    return sismos_procesados

# --- NUEVO ENDPOINT DE IA ---
@app.get("/prediccion_ia")
def predecir_importancia(lat: float, lng: float):
    if modelo_ia:
        try:
            # La IA predice la magnitud probable según la ubicación
            prediccion = modelo_ia.predict([[lat, lng]])
            return {
                "status": "success",
                "lat": lat,
                "lng": lng,
                "magnitud_predicha": round(float(prediccion[0]), 2),
                "mensaje": "Análisis realizado por el modelo RandomForest"
            }
        except Exception as e:
            return {"status": "error", "detalle": str(e)}
    
    return {"status": "error", "mensaje": "Modelo IA no cargado"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)