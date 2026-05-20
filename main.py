import joblib
import pandas as pd
import hashlib
import httpx
import asyncio
import sqlite3
from math import radians, cos, sin, asin, sqrt
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_NAME = "sismos_history"

# ==========================================
# CONFIGURACIÓN DE BASE DE DATOS (SQLite)
# ==========================================
def inicializar_base_datos():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sismos (
            id TEXT PRIMARY KEY,
            fuentes TEXT,
            lat REAL,
            lng REAL,
            mag_original REAL,
            mag_ia REAL,
            place TEXT,
            es_anomalia INTEGER,
            timestamp REAL
        )
    """)
    conn.commit()
    conn.close()

inicializar_base_datos()

# Carga del modelo de IA
modelo = None
try:
    modelo = joblib.load('modelo_sismos.pkl')
except:
    print("Aviso: Modo simulación activo (modelo.pkl no encontrado)")

def calcular_distancia(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r

async def obtener_sismos_usgs(client: httpx.AsyncClient):
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson"
    try:
        response = await client.get(url, timeout=5.0)
        data = response.json()
        sismos = []
        for feature in data.get("features", []):
            props = feature["properties"]
            geom = feature["geometry"]["coordinates"]
            sismos.append({
                "fuente": "USGS",
                "id_origen": feature["id"],
                "lat": geom[1],
                "lng": geom[0],
                "mag": props["mag"] if props["mag"] is not None else 0.0,
                "place": props["place"],
                "time": props["time"] / 1000.0
            })
        return sismos
    except Exception as e:
        print(f"Error consultando USGS: {e}")
        return []

async def obtener_sismos_emsc(client: httpx.AsyncClient):
    try:
        response = await client.get("https://www.emsc-csem.org/api/v1/earthquakes/?limit=50", timeout=5.0)
        data = response.json()
        sismos = []
        for item in data.get("data", []):
            dt = datetime.fromisoformat(item["time"].replace("Z", "+00:00"))
            sismos.append({
                "fuente": "EMSC",
                "id_origen": str(item["id"]),
                "lat": float(item["latitude"]),
                "lng": float(item["longitude"]),
                "mag": float(item["magnitude"]),
                "place": item["region_name"],
                "time": dt.timestamp()
            })
        return sismos
    except Exception as e:
        print(f"Error consultando EMSC: {e}")
        return []

# ==========================================
# ENDPOINT PRINCIPAL: MONITOREO Y PERSISTENCIA
# ==========================================
@app.get("/sismos_unificados")
async def sismos_unificados():
    async with httpx.AsyncClient() as client:
        lista_usgs, lista_emsc = await asyncio.gather(
            obtener_sismos_usgs(client),
            obtener_sismos_emsc(client)
        )
    
    todos_los_sismos = lista_usgs + lista_emsc
    unificados_memoria = []

    # Algoritmo de Deduplicación
    for sismo in todos_los_sismos:
        duplicado = False
        for unificado in unificados_memoria:
            distancia = calcular_distancia(sismo["lat"], sismo["lng"], unificado["lat"], unificado["lng"])
            diferencia_tiempo = abs(sismo["time"] - unificado["time"])
            
            if distancia < 50.0 and diferencia_tiempo < 120:
                duplicado = True
                if sismo["fuente"] not in unificado["fuentes_confirmadas"]:
                    unificado["fuentes_confirmadas"].append(sismo["fuente"])
                break
        
        if not duplicado:
            sismo["fuentes_confirmadas"] = [sismo["fuente"]]
            unificados_memoria.append(sismo)

    resultados_finales = []
    
    # Conexión a la base de datos para guardar nuevos eventos
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    for s in unificados_memoria:
        id_interno_cadena = f"{s['lat']}-{s['lng']}-{int(s['time'])}"
        id_propio = "SA-" + hashlib.md5(id_interno_cadena.encode()).hexdigest()[:8].upper()
        
        id_hash = int(hashlib.md5(id_propio.encode()).hexdigest(), 16)
        es_anomalia = (id_hash % 100) < 20 

        if modelo:
            input_df = pd.DataFrame([[s["lat"], s["lng"]]], columns=['latitud', 'longitud'])
            mag_ia = modelo.predict(input_df)[0]
        else:
            mag_ia = s["mag"]

        mag_ia_redondeada = round(float(mag_ia), 2)
        fuentes_str = ",".join(s["fuentes_confirmadas"])
        anomalia_int = 1 if es_anomalia else 0

        # GUARDADO INTELIGENTE: Si el sismo ya existe, no hace nada (evita errores de clave duplicada)
        cursor.execute("""
            INSERT OR IGNORE INTO sismos (id, fuentes, lat, lng, mag_original, mag_ia, place, es_anomalia, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (id_propio, fuentes_str, s["lat"], s["lng"], s["mag"], mag_ia_redondeada, s["place"], anomalia_int, s["time"]))

        resultados_finales.append({
            "id": id_propio,
            "fuentes": s["fuentes_confirmadas"],
            "lat": s["lat"],
            "lng": s["lng"],
            "mag_original": s["mag"],
            "mag_ia": mag_ia_redondeada,
            "place": s["place"],
            "es_anomalia": es_anomalia,
            "timestamp": s["time"]
        })
    
    conn.commit()
    conn.close()

    return resultados_finales

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)