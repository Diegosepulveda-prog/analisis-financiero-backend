"""
Backend de la app de análisis financiero.
Este archivo levanta un servidor web (con FastAPI) que expone "endpoints"
(direcciones URL) que nuestro frontend va a poder consultar para pedir
precios históricos y datos fundamentales de una acción.
"""

import os
import json
import requests
import pandas as pd
import psycopg2
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# La clave de API se lee de una variable de entorno, NUNCA hardcodeada acá.
# Cuando desplieguemos en Render, vamos a configurar esta variable en su panel.
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
FMP_BASE_URL = "https://financialmodelingprep.com/stable"

# Cadena de conexión a la base de datos Postgres (Supabase). Se configura
# como variable de entorno en Render, nunca escrita acá en el código.
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Cuánto tiempo consideramos "fresco" un dato guardado antes de volver
# a pedirlo a FMP. 24 horas alcanza de sobra para uso personal.
HORAS_DE_CACHE = 24


def get_conexion():
    """Abre una conexión nueva a la base de datos."""
    if not DATABASE_URL:
        return None
    return psycopg2.connect(DATABASE_URL)


def inicializar_base_datos():
    """
    Crea la tabla de caché si todavía no existe. Es una tabla genérica:
    cada fila guarda "para este ticker, este tipo de dato (precios,
    fundamental, etc.), esto es lo que devolvió la API, actualizado en
    tal momento". Guardamos el dato como JSON, así no hace falta una
    tabla distinta por cada tipo de información.
    """
    conexion = get_conexion()
    if conexion is None:
        return
    with conexion:
        with conexion.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_datos (
                    ticker TEXT NOT NULL,
                    tipo TEXT NOT NULL,
                    datos JSONB NOT NULL,
                    actualizado TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (ticker, tipo)
                )
                """
            )
    conexion.close()


def leer_cache(ticker: str, tipo: str):
    """
    Busca en la base de datos un dato guardado para este ticker/tipo.
    Devuelve None si no existe o si ya pasó el tiempo de vida (HORAS_DE_CACHE).
    """
    conexion = get_conexion()
    if conexion is None:
        return None
    try:
        with conexion.cursor() as cur:
            cur.execute(
                "SELECT datos, actualizado FROM cache_datos WHERE ticker = %s AND tipo = %s",
                (ticker, tipo),
            )
            fila = cur.fetchone()
            if not fila:
                return None
            datos, actualizado = fila
            limite = datetime.now(timezone.utc) - timedelta(hours=HORAS_DE_CACHE)
            if actualizado < limite:
                return None  # el dato existe pero ya está viejo
            return datos
    finally:
        conexion.close()


def guardar_cache(ticker: str, tipo: str, datos) -> None:
    """Guarda (o actualiza) el dato de un ticker/tipo en la base de datos."""
    conexion = get_conexion()
    if conexion is None:
        return
    try:
        with conexion:
            with conexion.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO cache_datos (ticker, tipo, datos, actualizado)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (ticker, tipo)
                    DO UPDATE SET datos = EXCLUDED.datos, actualizado = EXCLUDED.actualizado
                    """,
                    (ticker, tipo, json.dumps(datos), datetime.now(timezone.utc)),
                )
    finally:
        conexion.close()


app = FastAPI(title="API de Análisis Financiero Personal")

# Al arrancar el servidor, nos aseguramos de que la tabla exista.
@app.on_event("startup")
def on_startup():
    inicializar_base_datos()

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


def calcular_indicadores(precios: list) -> list:
    """
    Recibe la lista de precios históricos (más reciente primero, como los
    devuelve FMP) y le agrega, a cada día, la media móvil de 20 y 50 días
    y el RSI de 14 días. Estos son los indicadores técnicos más usados
    para leer el "momentum" de una acción.
    """
    if not precios:
        return precios

    # Pasamos a pandas y damos vuelta el orden (más antiguo primero),
    # porque los indicadores se calculan mirando hacia atrás en el tiempo.
    df = pd.DataFrame(precios)
    df = df.iloc[::-1].reset_index(drop=True)

    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()

    # RSI (Relative Strength Index): mide si una acción está "sobrecomprada"
    # (arriba de 70) o "sobrevendida" (debajo de 30) en los últimos 14 días.
    delta = df["close"].diff()
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)
    media_ganancia = ganancia.rolling(window=14).mean()
    media_perdida = perdida.rolling(window=14).mean()
    rs = media_ganancia / media_perdida
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Redondeamos y volvemos a dejar los datos en orden más reciente primero
    df = df.round({"sma_20": 2, "sma_50": 2, "rsi_14": 2})
    df = df.iloc[::-1].reset_index(drop=True)

    # NaN (los primeros días, que no tienen suficiente historia para calcular)
    # los convertimos a None para que sean JSON válido. Hace falta pasar a
    # tipo "object" primero, porque si no pandas vuelve a convertir None en NaN.
    df = df.astype(object).where(pd.notnull(df), None)
    return df.to_dict(orient="records")


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


@app.get("/indicadores/{ticker}")
def obtener_indicadores(ticker: str, dias: int = 180):
    """
    Devuelve los precios históricos ya con los indicadores técnicos
    (medias móviles de 20 y 50 días, RSI de 14 días) calculados.
    Primero busca en la base de datos; si no hay un dato fresco (menos
    de 24hs), le pregunta a FMP y guarda el resultado para la próxima vez.
    Ejemplo de uso: /indicadores/AAPL?dias=90
    """
    ticker = ticker.upper()
    con_indicadores = leer_cache(ticker, "indicadores")

    if con_indicadores is None:
        data = fmp_get("historical-price-eod/full", {"symbol": ticker})
        if not isinstance(data, list):
            raise HTTPException(status_code=502, detail="Respuesta inesperada de FMP")
        con_indicadores = calcular_indicadores(data)
        guardar_cache(ticker, "indicadores", con_indicadores)

    return {"ticker": ticker, "precios": con_indicadores[:dias]}


@app.get("/fundamental/{ticker}")
def obtener_fundamental(ticker: str):
    """
    Devuelve un resumen de datos fundamentales: ratios clave y perfil de
    la empresa. Usa la misma lógica de caché de 24hs que /indicadores.
    Ejemplo de uso: /fundamental/AAPL
    """
    ticker = ticker.upper()
    resultado = leer_cache(ticker, "fundamental")

    if resultado is None:
        perfil = fmp_get("profile", {"symbol": ticker})
        ratios = fmp_get("ratios", {"symbol": ticker, "limit": 1})
        resultado = {
            "ticker": ticker,
            "perfil": perfil[0] if isinstance(perfil, list) and perfil else {},
            "ratios": ratios[0] if isinstance(ratios, list) and ratios else {},
        }
        guardar_cache(ticker, "fundamental", resultado)

    return resultado
