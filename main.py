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


def calcular_valor_intrinseco(ticker: str) -> dict:
    """
    Calcula el valor intrínseco de una empresa usando un modelo de flujo
    de caja descontado (DCF - Discounted Cash Flow). Es el método clásico
    de análisis fundamental: proyecta el efectivo que va a generar la
    empresa en el futuro y lo "trae a valor de hoy".

    Simplificaciones que hacemos para que el modelo sea manejable:
    - Tasa libre de riesgo fija en 4.5% (aproximación al bono del Tesoro de EE.UU.)
    - Prima de riesgo de mercado fija en 5% (promedio histórico usual)
    - Proyectamos 5 años, con la tasa de crecimiento histórica de la empresa
    - Tasa de crecimiento a perpetuidad (terminal) fija en 2.5% (similar a
      la inflación de largo plazo esperada)
    """
    TASA_LIBRE_DE_RIESGO = 0.045
    PRIMA_DE_RIESGO_MERCADO = 0.05
    CRECIMIENTO_TERMINAL = 0.025
    ANIOS_DE_PROYECCION = 5

    perfil_resp = fmp_get("profile", {"symbol": ticker})
    flujo_caja_resp = fmp_get("cash-flow-statement", {"symbol": ticker, "limit": 5})
    balance_resp = fmp_get("balance-sheet-statement", {"symbol": ticker, "limit": 1})
    resultados_resp = fmp_get("income-statement", {"symbol": ticker, "limit": 1})

    if not perfil_resp or not flujo_caja_resp or not balance_resp:
        raise HTTPException(status_code=502, detail="No hay suficientes datos financieros para calcular el DCF")

    perfil = perfil_resp[0]
    balance = balance_resp[0]
    resultados = resultados_resp[0] if resultados_resp else {}

    # Flujo de caja libre (FCF) = efectivo generado por el negocio menos
    # lo que reinvierte en bienes de capital (maquinaria, equipos, etc.)
    flujos_libres = [
        (fila.get("operatingCashFlow") or fila.get("freeCashFlow") or 0)
        - abs(fila.get("capitalExpenditure") or 0)
        for fila in reversed(flujo_caja_resp)  # del más antiguo al más reciente
    ]
    if not flujos_libres or all(f == 0 for f in flujos_libres):
        raise HTTPException(status_code=502, detail="FMP no devolvió datos de flujo de caja utilizables para este ticker")

    # Tasa de crecimiento promedio histórica entre años consecutivos
    tasas_crecimiento = [
        (flujos_libres[i] - flujos_libres[i - 1]) / abs(flujos_libres[i - 1])
        for i in range(1, len(flujos_libres))
        if flujos_libres[i - 1] != 0
    ]
    crecimiento_estimado = sum(tasas_crecimiento) / len(tasas_crecimiento) if tasas_crecimiento else 0.05
    # Topamos el crecimiento a un rango razonable (-10% a 20%) para evitar
    # que un salto puntual en los datos históricos distorsione todo el modelo
    crecimiento_estimado = max(-0.10, min(0.20, crecimiento_estimado))

    # WACC (Weighted Average Cost of Capital): mezcla el costo de financiarse
    # con capital propio (acciones) y con deuda, ponderado por cuánto pesa
    # cada uno en la estructura de la empresa.
    beta = perfil.get("beta") or 1.0
    costo_capital_propio = TASA_LIBRE_DE_RIESGO + beta * PRIMA_DE_RIESGO_MERCADO

    deuda_total = balance.get("totalDebt", 0) or 0
    efectivo = balance.get("cashAndCashEquivalents", 0) or 0
    market_cap = perfil.get("marketCap", 0) or 0

    gasto_intereses = abs(resultados.get("interestExpense", 0) or 0)
    tasa_impositiva = resultados.get("incomeTaxExpense", 0) / resultados["incomeBeforeTax"] \
        if resultados.get("incomeBeforeTax") else 0.21  # 21% como default razonable
    tasa_impositiva = max(0, min(0.40, tasa_impositiva))

    costo_deuda = (gasto_intereses / deuda_total) if deuda_total else 0.05
    costo_deuda_despues_impuestos = costo_deuda * (1 - tasa_impositiva)

    valor_total = market_cap + deuda_total
    peso_capital = (market_cap / valor_total) if valor_total else 1
    peso_deuda = (deuda_total / valor_total) if valor_total else 0

    wacc = (peso_capital * costo_capital_propio) + (peso_deuda * costo_deuda_despues_impuestos)
    wacc = max(0.04, min(0.20, wacc))  # límites de sanidad

    # Proyectamos los flujos de los próximos 5 años y los descontamos a valor presente
    ultimo_flujo = flujos_libres[-1]
    flujos_proyectados = []
    valor_presente_flujos = 0
    for anio in range(1, ANIOS_DE_PROYECCION + 1):
        flujo_futuro = ultimo_flujo * ((1 + crecimiento_estimado) ** anio)
        valor_presente = flujo_futuro / ((1 + wacc) ** anio)
        flujos_proyectados.append({"anio": anio, "flujo_proyectado": round(flujo_futuro, 0), "valor_presente": round(valor_presente, 0)})
        valor_presente_flujos += valor_presente

    # Valor terminal: todo lo que la empresa genera después del año 5,
    # asumiendo que a partir de ahí crece a un ritmo estable para siempre
    flujo_terminal = flujos_proyectados[-1]["flujo_proyectado"] * (1 + CRECIMIENTO_TERMINAL)
    valor_terminal = flujo_terminal / (wacc - CRECIMIENTO_TERMINAL)
    valor_presente_terminal = valor_terminal / ((1 + wacc) ** ANIOS_DE_PROYECCION)

    valor_empresa = valor_presente_flujos + valor_presente_terminal  # Enterprise Value
    valor_patrimonio = valor_empresa - deuda_total + efectivo  # Equity Value

    precio_actual = perfil.get("price")
    acciones_en_circulacion = None
    if market_cap and precio_actual:
        acciones_en_circulacion = market_cap / precio_actual
    elif perfil.get("sharesOutstanding"):
        acciones_en_circulacion = perfil["sharesOutstanding"]

    valor_intrinseco_por_accion = (
        valor_patrimonio / acciones_en_circulacion if acciones_en_circulacion else None
    )

    diferencia_pct = (
        ((valor_intrinseco_por_accion - precio_actual) / precio_actual) * 100
        if valor_intrinseco_por_accion and precio_actual else None
    )

    return {
        "ticker": ticker,
        "precio_actual": precio_actual,
        "valor_intrinseco_por_accion": round(valor_intrinseco_por_accion, 2) if valor_intrinseco_por_accion else None,
        "diferencia_porcentual": round(diferencia_pct, 2) if diferencia_pct is not None else None,
        "veredicto": (
            "subvaluada" if diferencia_pct and diferencia_pct > 10 else
            "sobrevaluada" if diferencia_pct and diferencia_pct < -10 else
            "valuada razonablemente" if diferencia_pct is not None else None
        ),
        "supuestos": {
            "wacc": round(wacc * 100, 2),
            "crecimiento_estimado_fcf": round(crecimiento_estimado * 100, 2),
            "crecimiento_terminal": round(CRECIMIENTO_TERMINAL * 100, 2),
            "beta": beta,
        },
        "flujos_proyectados": flujos_proyectados,
        "valor_terminal_presente": round(valor_presente_terminal, 0),
        "valor_empresa": round(valor_empresa, 0),
    }


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


@app.get("/valor-intrinseco/{ticker}")
def obtener_valor_intrinseco(ticker: str, forzar: bool = False):
    """
    Devuelve el valor intrínseco de una empresa calculado con un modelo
    de flujo de caja descontado (DCF), comparado contra el precio actual
    de mercado. Usa la misma caché de 24hs que los demás endpoints.
    Agregá ?forzar=true a la URL para ignorar la caché y recalcular.
    Ejemplo de uso: /valor-intrinseco/AAPL?forzar=true
    """
    ticker = ticker.upper()
    resultado = None if forzar else leer_cache(ticker, "valor_intrinseco")

    if resultado is None:
        resultado = calcular_valor_intrinseco(ticker)
        guardar_cache(ticker, "valor_intrinseco", resultado)

    return resultado
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
