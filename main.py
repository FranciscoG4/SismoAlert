from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse # <--- NUEVO
import requests
import sqlite3
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
DB_NAME = "sismos_history.db"

# --- RUTA PARA MOSTRAR TU MAPA ---
@app.get("/")
async def read_index():
    # Esto hace que cuando alguien entre al link, vea tu index.html
    return FileResponse('index.html')

# --- INICIALIZACIÓN DE BASE DE DATOS ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sismos (
            id TEXT PRIMARY KEY,
            lat REAL,
            lng REAL,
            mag REAL,
            place TEXT,
            time INTEGER,
            riesgo TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def calcular_riesgo(mag):
    if mag >= 5: return "alto"
    elif mag >= 3: return "medio"
    else: return "bajo"

@app.get("/sismos")
def obtener_sismos():
    try:
        response = requests.get(USGS_URL, timeout=10)
        data = response.json()
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        for evento in data["features"]:
            id_sismo = evento["id"]
            prop = evento["properties"]
            geom = evento["geometry"]["coordinates"]
            
            mag = prop.get("mag")
            if mag is None: continue

            cursor.execute('''
                INSERT OR IGNORE INTO sismos (id, lat, lng, mag, place, time, riesgo)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (id_sismo, geom[1], geom[0], mag, prop.get("place"), prop.get("time"), calcular_riesgo(mag)))

        conn.commit()
        
        cursor.execute("SELECT lat, lng, mag, place, riesgo FROM sismos ORDER BY time DESC LIMIT 200")
        rows = cursor.fetchall()
        conn.close()

        resultado = []
        for r in rows:
            resultado.append({
                "lat": r[0], "lng": r[1], "magnitud_promedio": r[2],
                "ubicacion": r[3], "riesgo": r[4]
            })
        return resultado

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))