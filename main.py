import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="SismoAlert API")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable global para el modelo
modelo = None

class SismoData(BaseModel):
    latitud: float
    longitud: float

@app.get("/")
def home():
    return {"mensaje": "Servidor de SismoAlert funcionando"}

@app.post("/predict")
async def predict_magnitude(data: SismoData):
    global modelo
    
    # Solo cargamos el modelo si no está en memoria
    if modelo is None:
        try:
            modelo = joblib.load('modelo_sismos.pkl')
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error al cargar el cerebro de la IA")

    try:
        input_df = pd.DataFrame([[data.latitud, data.longitud]], columns=['latitud', 'longitud'])
        prediccion = modelo.predict(input_df)[0]
        return {
            "latitud": data.latitud,
            "longitud": data.longitud,
            "magnitud_estimada": round(float(prediccion), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error en la predicción")