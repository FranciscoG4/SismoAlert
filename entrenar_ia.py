import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

print("--- INICIANDO PROCESO DE IA ---")

try:
    # 1. Conexión
    conn = sqlite3.connect('sismos_history.db')
    print("Conectado a la base de datos...")
    
    # 2. Leer datos
    df = pd.read_sql_query("SELECT lat, lng, mag FROM sismos", conn)
    conn.close()
    print(f"Sismos encontrados: {len(df)}")

    if len(df) < 2:
        print("ERROR: Muy pocos datos. Necesitás al menos 2 sismos para que el código no falle.")
    else:
        # 3. Entrenar
        X = df[['lat', 'lng']]
        y = df['mag']
        
        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo.fit(X.values, y.values)
        
        # 4. Guardar
        joblib.dump(modelo, 'modelo_sismos.pkl')
        print("✅ ¡ÉXITO! Se generó 'modelo_sismos.pkl'")

except Exception as e:
    print(f"❌ OCURRIÓ UN ERROR: {e}")

print("--- FIN DEL SCRIPT ---")