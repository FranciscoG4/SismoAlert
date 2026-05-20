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
from contextlib import asynccontextmanager

DB_NAME = "sismos_history.db"

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
    return c * 6371

# ==========================================
# RECOLECCIÓN ASINCRÓNICA DE FUENTES
# ==========================================
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
                "lat": geom[1],
                "lng": geom[0],
                "mag": props["mag"] if props["mag"] is not None else 0.0,
                "place": props["place"],
                "time": props["time"] / 1000.0
            })
        return sismos
    except Exception as e:
        print(f"Error en worker (USGS): {e}")
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
                "lat": float(item["latitude"]),
                "lng": float(item["longitude"]),
                "mag": float(item["magnitude"]),
                "place": item["region_name"],
                "time": dt.timestamp()
            })
        return sismos
    except Exception as e:
        print(f"Error en worker (EMSC): {e}")
        return []

# ==========================================
# WORKER EN SEGUNDO PLANO (BACKGROUND TASK)
# ==========================================
async def sismos_background_worker():
    print("Worker de Sismos iniciado.")
    async with httpx.AsyncClient() as client:
        while True:
            try:
                lista_usgs, lista_emsc = await asyncio.gather(
                    obtener_sismos_usgs(client),
                    obtener_sismos_emsc(client)
                )
                
                todos = lista_usgs + lista_emsc
                unificados = []

                # Algoritmo de Deduplicación veloz en memoria (CORREGIDO)
                for sismo in todos:
                    duplicado = False
                    for u in unificados:
                        distancia = calcular_distancia(sismo["lat"], sismo["lng"], u["lat"], u["lng"])
                        diff_tiempo = abs(sismo["time"] - u["time"])
                        
                        # CORRECCIÓN AQUÍ: Usamos 'distancia' en lugar de 'distance'
                        if distancia < 50.0 and diff_tiempo < 120:
                            duplicado = True
                            if sismo["fuente"] not in u["fuentes_confirmadas"]:
                                u["fuentes_confirmadas"].append(sismo["fuente"])
                            break
                    if not duplicado:
                        sismo["fuentes_confirmadas"] = [sismo["fuente"]]
                        unificados.append(sismo)

                # Persistencia en base de datos
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()

                for s in unificados:
                    id_cadena = f"{s['lat']}-{s['lng']}-{int(s['time'])}"
                    id_propio = "SA-" + hashlib.md5(id_cadena.encode()).hexdigest()[:8].upper()
                    
                    id_hash = int(hashlib.md5(id_propio.encode()).hexdigest(), 16)
                    es_anomalia = (id_hash % 100) < 20

                    if modelo:
                        input_df = pd.DataFrame([[s["lat"], s["lng"]]], columns=['latitud', 'longitud'])
                        mag_ia = modelo.predict(input_df)[0]
                    else:
                        mag_ia = s["mag"]

                    fuentes_str = ",".join(s["fuentes_confirmadas"])
                    cursor.execute("""
                        INSERT OR IGNORE INTO sismos (id, fuentes, lat, lng, mag_original, mag_ia, place, es_anomalia, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (id_propio, fuentes_str, s["lat"], s["lng"], s["mag"], round(float(mag_ia), 2), s["place"], 1 if es_anomalia else 0, s["time"]))
                
                conn.commit()
                conn.close()

            except Exception as e:
                print(f"Error en ciclo del Worker: {e}")
            
            await asyncio.sleep(10)

# Manejo del ciclo de vida de FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    inicializar_base_datos()
    worker_task = asyncio.create_task(sismos_background_worker())
    yield
    worker_task.cancel()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# ENDPOINT ULTRA VELOZ: RESPUESTA INMEDIATA
# ==========================================
@app.get("/sismos_unificados")
async def sismos_unificados():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM sismos ORDER BY timestamp DESC LIMIT 100")
    rows = cursor.fetchall()
    conn.close()

    resultados = []
    for row in rows:
        resultados.append({
            "id": row["id"],
            "fuentes": row["fuentes"].split(","),
            "lat": row["lat"],
            "lng": row["lng"],
            "mag_original": row["mag_original"],
            "mag_ia": row["mag_ia"],
            "place": row["place"],
            "es_anomalia": bool(row["es_anomalia"]),
            "timestamp": row["timestamp"]
        })

    return resultados

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)