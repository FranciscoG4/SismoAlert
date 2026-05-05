import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. CARGA DEL MODELO (Fuera de las funciones para mayor eficiencia)
try:
    modelo = joblib.load('modelo_sismos.pkl')
    print("Modelo IA cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    modelo = None

app = FastAPI(title="SismoAlert API")

# 2. CONFIGURACIÓN DE CORS (Vital para que el index.html no sea bloqueado)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite peticiones desde cualquier origen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definimos el esquema de los datos que recibimos del mapa
class SismoData(BaseModel):
    latitud: float
    longitud: float

@app.get("/")
def home():
    return {"mensaje": "Servidor de SismoAlert funcionando correctamente"}

# 3. RUTA DE PREDICCIÓN CON MANEJO DE ERRORES
@app.post("/predict")
async def predict_magnitude(data: SismoData):
    if modelo is None:
        raise HTTPException(status_code=500, detail="El modelo de IA no está disponible.")
    
    try:
        # Preparamos los datos en el formato que espera el Random Forest
        input_df = pd.DataFrame([[data.latitud, data.longitud]], columns=['latitud', 'longitud'])
        
        # Realizamos la predicción
        prediccion = modelo.predict(input_df)[0]
        
        return {
            "latitud": data.latitud,
            "longitud": data.longitud,
            "magnitud_estimada": round(float(prediccion), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la predicción: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)