"""
Backend de la app de análisis financiero.
Este archivo levanta un servidor web (con FastAPI) que expone "endpoints"
(direcciones URL) que nuestro frontend va a poder consultar para pedir
precios históricos y datos fundamentales de una acción.
"""

import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# La clave de API se lee de una variable de entorno, NUNCA hardcodeada acá.
# Cuando desplieguemos en Render, vamos a configurar esta variable en su panel.
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
FMP_BASE_URL = "https://financialmodelingprep.com/stable"

app = FastAPI(title="API de Análisis Financiero Personal")

# CORS: le permite a nuestro frontend (que va a vivir en otro dominio, Vercel)
# hacerle pedidos a este backend sin que el navegador los bloquee.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción conviene restringir esto a tu dominio de Vercel
    allow_methods=["GET"],
    allow_headers=["*"],
)


def fmp_get(endpoint: str, params: dict) -> dict:
    """Función auxiliar: le hace un pedido a la API de FMP y devuelve el JSON."""
    if not FMP_API_KEY:
        raise HTTPException(status_code=500, detail="Falta configurar FMP_API_KEY en el servidor")

    params["apikey"] = FMP_API_KEY
    url = f"{FMP_BASE_URL}/{endpoint}"
    response = requests.get(url, params=params, timeout=15)

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Error al consultar FMP: {response.text}",
        )
    return response.json()


@app.get("/")
def home():
    """Endpoint de prueba: si esto responde, el backend está vivo."""
    return {"status": "ok", "mensaje": "Backend de análisis financiero funcionando"}


@app.get("/precios/{ticker}")
def obtener_precios(ticker: str, dias: int = 180):
    """
    Devuelve precios históricos (para el gráfico de velas) de un ticker.
    Ejemplo de uso: /precios/AAPL?dias=90
    """
    data = fmp_get(f"historical-price-eod/full", {"symbol": ticker.upper()})

    # FMP devuelve una lista de precios día por día; nos quedamos con los últimos N días
    if isinstance(data, list):
        data = data[:dias]

    return {"ticker": ticker.upper(), "precios": data}


@app.get("/fundamental/{ticker}")
def obtener_fundamental(ticker: str):
    """
    Devuelve un resumen de datos fundamentales: ratios clave y perfil de la empresa.
    Ejemplo de uso: /fundamental/AAPL
    """
    perfil = fmp_get("profile", {"symbol": ticker.upper()})
    ratios = fmp_get("ratios", {"symbol": ticker.upper(), "limit": 1})

    return {
        "ticker": ticker.upper(),
        "perfil": perfil[0] if isinstance(perfil, list) and perfil else {},
        "ratios": ratios[0] if isinstance(ratios, list) and ratios else {},
    }
