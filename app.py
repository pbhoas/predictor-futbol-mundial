
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import date
from collections import defaultdict, deque
from scipy.stats import poisson
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from xgboost import XGBRegressor

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

st.set_page_config(
    page_title="Predictor de Fútbol",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Predictor de fútbol internacional")
st.caption("Modelo experimental basado en resultados históricos. No usar como recomendación financiera ni de apuestas.")

# ----------------------------
# Datos
# ----------------------------

@st.cache_data(show_spinner="Cargando resultados históricos...")
def cargar_partidos():
    partidos = pd.read_csv(URL)
    partidos["date"] = pd.to_datetime(partidos["date"])

    partidos_jugados = partidos.dropna(
        subset=["home_score", "away_score"]
    ).copy()

    partidos_jugados["home_score"] = partidos_jugados["home_score"].astype(int)
    partidos_jugados["away_score"] = partidos_jugados["away_score"].astype(int)

    partidos_jugados = partidos_jugados.sort_values("date").reset_index(drop=True)

    return partidos, partidos_jugados


partidos, partidos_jugados = cargar_partidos()


def tipo_de_partido(torneo):
    if torneo == "Friendly":
        return "Amistoso"
    elif torneo == "FIFA World Cup":
        return "Mundial"
    else:
        return "Oficial"


def preparar_datos(fecha_prediccion, anos_historial=10):
    fecha_prediccion = pd.to_datetime(fecha_prediccion)
    fecha_inicio = fecha_prediccion - pd.DateOffset(years=anos_historial)

    datos = partidos_jugados[
        (partidos_jugados["date"] < fecha_prediccion) &
        (partidos_jugados["date"] >= fecha_inicio)
    ].copy()

    dias_antiguedad = (fecha_prediccion - datos["date"]).dt.days
    datos["peso_recencia"] = np.exp(-dias_antiguedad / 730)

    datos["peso_torneo"] = np.where(
        datos["tournament"] == "Friendly",
        0.50,
        1.00
    )

    datos.loc[
        datos["tournament"] == "FIFA World Cup",
        "peso_torneo"
    ] = 1.30

    datos["peso_final"] = datos["peso_recencia"] * datos["peso_torneo"]

    return datos


@st.cache_resource(show_spinner=False)
def entrenar_modelo_poisson(fecha_prediccion_str):
    fecha_prediccion = pd.to_datetime(fecha_prediccion_str)
    datos = preparar_datos(fecha_prediccion)

    filas_local = pd.DataFrame({
        "equipo": datos["home_team"],
        "rival": datos["away_team"],
        "sede": np.where(datos["neutral"], "Neutral", "Local"),
        "tipo_partido": datos["tournament"].apply(tipo_de_partido),
        "goles": datos["home_score"],
        "peso": datos["peso_final"]
    })

    filas_visitante = pd.DataFrame({
        "equipo": datos["away_team"],
        "rival": datos["home_team"],
        "sede": np.where(datos["neutral"], "Neutral", "Visitante"),
        "tipo_partido": datos["tournament"].apply(tipo_de_partido),
        "goles": datos["away_score"],
        "peso": datos["peso_final"]
    })

    entrenamiento = pd.concat([filas_local, filas_visitante], ignore_index=True)

    columnas = ["equipo", "rival", "sede", "tipo_partido"]

    X = entrenamiento[columnas]
    y = entrenamiento["goles"]
    pesos = entrenamiento["peso"]

    preprocesador = ColumnTransformer([
        ("categorias", OneHotEncoder(handle_unknown="ignore"), columnas)
    ])

    X_codificado = preprocesador.fit_transform(X)

    modelo = PoissonRegressor(
        alpha=0.01,
        max_iter=2000
    )

    modelo.fit(X_codificado, y, sample_weight=pesos)

    return modelo, preprocesador, entrenamiento


@st.cache_data(show_spinner=False)
def calcular_elo_cached(fecha_limite_str):
    fecha_limite = pd.to_datetime(fecha_limite_str)

    datos = partidos_jugados[
        partidos_jugados["date"] < fecha_limite
    ].copy()

    datos = datos.sort_values("date")

    ratings = {}
    elo_inicial = 1500
    k = 30

    for _, partido in datos.iterrows():
        local = partido["home_team"]
        visitante = partido["away_team"]

        elo_local = ratings.get(local, elo_inicial)
        elo_visitante = ratings.get(visitante, elo_inicial)

        ventaja_local = 0 if partido["neutral"] else 80

        esperado_local = 1 / (
            1 + 10 ** ((elo_visitante - elo_local - ventaja_local) / 400)
        )

        if partido["home_score"] > partido["away_score"]:
            resultado_local = 1
        elif partido["home_score"] == partido["away_score"]:
            resultado_local = 0.5
        else:
            resultado_local = 0

        diferencia = resultado_local - esperado_local

        ratings[local] = elo_local + k * diferencia
        ratings[visitante] = elo_visitante - k * diferencia

    return ratings


def predecir_goles_poisson_elo(equipo_a, equipo_b, fecha, neutral=True, tipo_partido="Mundial"):
    modelo, preprocesador, entrenamiento = entrenar_modelo_poisson(str(fecha))

    sede_a = "Neutral" if neutral else "Local"
    sede_b = "Neutral" if neutral else "Visitante"

    partido_a = pd.DataFrame([{
        "equipo": equipo_a,
        "rival": equipo_b,
        "sede": sede_a,
        "tipo_partido": tipo_partido
    }])

    partido_b = pd.DataFrame([{
        "equipo": equipo_b,
        "rival": equipo_a,
        "sede": sede_b,
        "tipo_partido": tipo_partido
    }])

    goles_a = float(modelo.predict(preprocesador.transform(partido_a))[0])
    goles_b = float(modelo.predict(preprocesador.transform(partido_b))[0])

    goles_a = float(np.clip(goles_a, 0.05, 5.00))
    goles_b = float(np.clip(goles_b, 0.05, 5.00))

    ratings = calcular_elo_cached(str(fecha))
    elo_a = ratings.get(equipo_a, 1500)
    elo_b = ratings.get(equipo_b, 1500)

    diferencia_elo = elo_a - elo_b

    intensidad = 0.35
    factor = np.exp(intensidad * np.log(10) * diferencia_elo / 400)

    goles_a_ajustados = goles_a * factor
    goles_b_ajustados = goles_b / factor

    total_original = goles_a + goles_b
    total_ajustado = goles_a_ajustados + goles_b_ajustados

    goles_a_ajustados *= total_original / total_ajustado
    goles_b_ajustados *= total_original / total_ajustado

    return goles_a_ajustados, goles_b_ajustados, elo_a, elo_b


# ----------------------------
# XGBoost
# ----------------------------

COLUMNAS_XGB = [
    "elo_local",
    "elo_visitante",
    "diferencia_elo",
    "gf_local_5",
    "gc_local_5",
    "gf_visitante_5",
    "gc_visitante_5",
    "gf_local_10",
    "gc_local_10",
    "gf_visitante_10",
    "gc_visitante_10",
    "puntos_local_5",
    "puntos_visitante_5",
    "dias_descanso_local",
    "dias_descanso_visitante",
    "cancha_neutral",
    "tipo_torneo"
]


@st.cache_data(show_spinner="Preparando variables de XGBoost...")
def crear_datos_xgboost():
    elo = defaultdict(lambda: 1500.0)
    goles_favor = defaultdict(lambda: deque(maxlen=10))
    goles_contra = defaultdict(lambda: deque(maxlen=10))
    resultados = defaultdict(lambda: deque(maxlen=10))
    ultima_fecha = {}

    filas = []

    datos = partidos_jugados.sort_values("date").copy()

    for _, partido in datos.iterrows():
        local = partido["home_team"]
        visitante = partido["away_team"]
        fecha = partido["date"]

        goles_local = int(partido["home_score"])
        goles_visitante = int(partido["away_score"])

        elo_local = elo[local]
        elo_visitante = elo[visitante]

        def promedio(lista, valor_defecto):
            return float(np.mean(lista)) if len(lista) > 0 else valor_defecto

        def promedio_ultimos(lista, cantidad, defecto):
            valores = list(lista)[-cantidad:]
            return float(np.mean(valores)) if len(valores) > 0 else defecto

        dias_local = (fecha - ultima_fecha[local]).days if local in ultima_fecha else 90
        dias_visitante = (fecha - ultima_fecha[visitante]).days if visitante in ultima_fecha else 90

        if partido["tournament"] == "Friendly":
            tipo_torneo = 0
        elif partido["tournament"] == "FIFA World Cup":
            tipo_torneo = 2
        else:
            tipo_torneo = 1

        filas.append({
            "date": fecha,
            "home_team": local,
            "away_team": visitante,
            "elo_local": elo_local,
            "elo_visitante": elo_visitante,
            "diferencia_elo": elo_local - elo_visitante,
            "gf_local_5": promedio_ultimos(goles_favor[local], 5, 1.35),
            "gc_local_5": promedio_ultimos(goles_contra[local], 5, 1.35),
            "gf_visitante_5": promedio_ultimos(goles_favor[visitante], 5, 1.35),
            "gc_visitante_5": promedio_ultimos(goles_contra[visitante], 5, 1.35),
            "gf_local_10": promedio(goles_favor[local], 1.35),
            "gc_local_10": promedio(goles_contra[local], 1.35),
            "gf_visitante_10": promedio(goles_favor[visitante], 1.35),
            "gc_visitante_10": promedio(goles_contra[visitante], 1.35),
            "puntos_local_5": promedio_ultimos(resultados[local], 5, 1.0),
            "puntos_visitante_5": promedio_ultimos(resultados[visitante], 5, 1.0),
            "dias_descanso_local": min(dias_local, 365),
            "dias_descanso_visitante": min(dias_visitante, 365),
            "cancha_neutral": int(partido["neutral"]),
            "tipo_torneo": tipo_torneo,
            "goles_local": goles_local,
            "goles_visitante": goles_visitante
        })

        ventaja_local = 0 if partido["neutral"] else 80

        esperado_local = 1 / (
            1 + 10 ** ((elo_visitante - elo_local - ventaja_local) / 400)
        )

        if goles_local > goles_visitante:
            resultado_local = 1.0
            puntos_local, puntos_visitante = 3, 0
        elif goles_local == goles_visitante:
            resultado_local = 0.5
            puntos_local, puntos_visitante = 1, 1
        else:
            resultado_local = 0.0
            puntos_local, puntos_visitante = 0, 3

        k = 40 if tipo_torneo == 2 else 30
        cambio = k * (resultado_local - esperado_local)

        elo[local] += cambio
        elo[visitante] -= cambio

        goles_favor[local].append(goles_local)
        goles_contra[local].append(goles_visitante)

        goles_favor[visitante].append(goles_visitante)
        goles_contra[visitante].append(goles_local)

        resultados[local].append(puntos_local)
        resultados[visitante].append(puntos_visitante)

        ultima_fecha[local] = fecha
        ultima_fecha[visitante] = fecha

    return pd.DataFrame(filas)


@st.cache_resource(show_spinner="Entrenando XGBoost...")
def entrenar_xgboost(fecha_str):
    fecha = pd.to_datetime(fecha_str)
    datos_xgb = crear_datos_xgboost()

    datos_entrenamiento = datos_xgb[
        (datos_xgb["date"] >= "2016-01-01") &
        (datos_xgb["date"] < fecha)
    ].copy()

    X_train = datos_entrenamiento[COLUMNAS_XGB]

    modelo_local = XGBRegressor(
        objective="count:poisson",
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.10,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    modelo_visitante = XGBRegressor(
        objective="count:poisson",
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.10,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    modelo_local.fit(X_train, datos_entrenamiento["goles_local"])
    modelo_visitante.fit(X_train, datos_entrenamiento["goles_visitante"])

    return modelo_local, modelo_visitante, len(datos_entrenamiento)


def crear_partido_futuro_xgb(equipo_a, equipo_b, fecha, neutral=True, tipo_partido="Mundial"):
    fecha = pd.to_datetime(fecha)

    historial = partidos_jugados[
        partidos_jugados["date"] < fecha
    ].sort_values("date")

    elo = defaultdict(lambda: 1500.0)
    goles_favor = defaultdict(lambda: deque(maxlen=10))
    goles_contra = defaultdict(lambda: deque(maxlen=10))
    resultados = defaultdict(lambda: deque(maxlen=10))
    ultima_fecha = {}

    for _, partido in historial.iterrows():
        local = partido["home_team"]
        visitante = partido["away_team"]

        gl = int(partido["home_score"])
        gv = int(partido["away_score"])

        elo_local = elo[local]
        elo_visitante = elo[visitante]

        ventaja = 0 if partido["neutral"] else 80

        esperado = 1 / (
            1 + 10 ** ((elo_visitante - elo_local - ventaja) / 400)
        )

        if gl > gv:
            resultado_local = 1.0
            puntos_local, puntos_visitante = 3, 0
        elif gl == gv:
            resultado_local = 0.5
            puntos_local, puntos_visitante = 1, 1
        else:
            resultado_local = 0.0
            puntos_local, puntos_visitante = 0, 3

        k = 40 if partido["tournament"] == "FIFA World Cup" else 30
        cambio = k * (resultado_local - esperado)

        elo[local] += cambio
        elo[visitante] -= cambio

        goles_favor[local].append(gl)
        goles_contra[local].append(gv)
        goles_favor[visitante].append(gv)
        goles_contra[visitante].append(gl)

        resultados[local].append(puntos_local)
        resultados[visitante].append(puntos_visitante)

        ultima_fecha[local] = partido["date"]
        ultima_fecha[visitante] = partido["date"]

    def promedio(valores, cantidad, defecto):
        lista = list(valores)[-cantidad:]
        return float(np.mean(lista)) if lista else defecto

    def descanso(equipo):
        if equipo not in ultima_fecha:
            return 90
        return min((fecha - ultima_fecha[equipo]).days, 365)

    tipos = {
        "Amistoso": 0,
        "Oficial": 1,
        "Mundial": 2
    }

    fila = pd.DataFrame([{
        "elo_local": elo[equipo_a],
        "elo_visitante": elo[equipo_b],
        "diferencia_elo": elo[equipo_a] - elo[equipo_b],
        "gf_local_5": promedio(goles_favor[equipo_a], 5, 1.35),
        "gc_local_5": promedio(goles_contra[equipo_a], 5, 1.35),
        "gf_visitante_5": promedio(goles_favor[equipo_b], 5, 1.35),
        "gc_visitante_5": promedio(goles_contra[equipo_b], 5, 1.35),
        "gf_local_10": promedio(goles_favor[equipo_a], 10, 1.35),
        "gc_local_10": promedio(goles_contra[equipo_a], 10, 1.35),
        "gf_visitante_10": promedio(goles_favor[equipo_b], 10, 1.35),
        "gc_visitante_10": promedio(goles_contra[equipo_b], 10, 1.35),
        "puntos_local_5": promedio(resultados[equipo_a], 5, 1.0),
        "puntos_visitante_5": promedio(resultados[equipo_b], 5, 1.0),
        "dias_descanso_local": descanso(equipo_a),
        "dias_descanso_visitante": descanso(equipo_b),
        "cancha_neutral": int(neutral),
        "tipo_torneo": tipos[tipo_partido]
    }])

    return fila[COLUMNAS_XGB]


def predecir_goles_xgb(equipo_a, equipo_b, fecha, neutral=True, tipo_partido="Mundial"):
    modelo_local, modelo_visitante, n_entrenamiento = entrenar_xgboost(str(fecha))

    partido_futuro = crear_partido_futuro_xgb(
        equipo_a,
        equipo_b,
        fecha,
        neutral,
        tipo_partido
    )

    goles_a = float(modelo_local.predict(partido_futuro)[0])
    goles_b = float(modelo_visitante.predict(partido_futuro)[0])

    goles_a = float(np.clip(goles_a, 0.05, 5))
    goles_b = float(np.clip(goles_b, 0.05, 5))

    return goles_a, goles_b, n_entrenamiento


# ----------------------------
# Probabilidades y gráficos
# ----------------------------

def calcular_probabilidades(goles_a, goles_b, max_goles=8):
    prob_a = poisson.pmf(range(max_goles + 1), goles_a)
    prob_b = poisson.pmf(range(max_goles + 1), goles_b)

    matriz = np.outer(prob_a, prob_b)
    matriz = matriz / matriz.sum()

    gana_a = np.tril(matriz, -1).sum()
    empate = np.trace(matriz)
    gana_b = np.triu(matriz, 1).sum()

    ambos_marcan = matriz[1:, 1:].sum()

    mas_15 = sum(matriz[i, j] for i in range(max_goles + 1) for j in range(max_goles + 1) if i + j >= 2)
    mas_25 = sum(matriz[i, j] for i in range(max_goles + 1) for j in range(max_goles + 1) if i + j >= 3)
    mas_35 = sum(matriz[i, j] for i in range(max_goles + 1) for j in range(max_goles + 1) if i + j >= 4)

    marcadores = []
    for goles_equipo_a in range(max_goles + 1):
        for goles_equipo_b in range(max_goles + 1):
            marcadores.append({
                "Marcador": f"{goles_equipo_a}-{goles_equipo_b}",
                "Probabilidad": matriz[goles_equipo_a, goles_equipo_b] * 100
            })

    marcadores = pd.DataFrame(marcadores)
    marcadores = marcadores.sort_values("Probabilidad", ascending=False).reset_index(drop=True)

    resumen = {
        "Gana equipo A": gana_a * 100,
        "Empate": empate * 100,
        "Gana equipo B": gana_b * 100,
        "Más de 1.5 goles": mas_15 * 100,
        "Menos de 1.5 goles": (1 - mas_15) * 100,
        "Más de 2.5 goles": mas_25 * 100,
        "Menos de 2.5 goles": (1 - mas_25) * 100,
        "Más de 3.5 goles": mas_35 * 100,
        "Menos de 3.5 goles": (1 - mas_35) * 100,
        "Ambos marcan": ambos_marcan * 100,
        "No marcan ambos": (1 - ambos_marcan) * 100
    }

    return matriz, resumen, marcadores


def graficar_marcadores(matriz, equipo_a, equipo_b, max_goles=5, vmin=None, vmax=None):
    datos = matriz[:max_goles + 1, :max_goles + 1] * 100

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    im = ax.imshow(datos, vmin=vmin, vmax=vmax)

    ax.set_title(f"Probabilidad por marcador\n{equipo_a} vs {equipo_b}", fontsize=10)
    ax.set_xlabel(f"Goles de {equipo_b}", fontsize=9)
    ax.set_ylabel(f"Goles de {equipo_a}", fontsize=9)

    ax.set_xticks(range(max_goles + 1))
    ax.set_yticks(range(max_goles + 1))
    ax.tick_params(labelsize=8)

    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            ax.text(j, i, f"{datos[i, j]:.1f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Probabilidad (%)", fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


EVENTOS_COMPARACION = [
    ("Gana Equipo A", "Gana equipo A"),
    ("Empate", "Empate"),
    ("Gana Equipo B", "Gana equipo B"),
    ("Más de 1.5 goles", "Más de 1.5 goles"),
    ("Más de 2.5 goles", "Más de 2.5 goles"),
    ("Más de 3.5 goles", "Más de 3.5 goles"),
    ("Ambos marcan", "Ambos marcan"),
]

ESTADOS_COMPETITIVOS = [
    "Normal",
    "Necesita ganar",
    "Le sirve empatar",
    "Necesita ganar por diferencia de goles",
    "Ya clasificado",
    "Ya eliminado",
]

RIESGOS_ROTACION = ["Bajo", "Medio", "Alto"]


def formato_porcentaje(valor):
    return f"{valor:.1f}%"


def formato_diferencia(valor):
    return f"{valor:+.1f} pp"


def calcular_modelo(modelo, equipo_a, equipo_b, fecha, neutral, tipo_partido):
    if modelo == "Poisson + Elo":
        goles_a, goles_b, elo_a, elo_b = predecir_goles_poisson_elo(
            equipo_a,
            equipo_b,
            fecha,
            neutral,
            tipo_partido
        )
        extra = {
            "Elo Equipo A": f"{elo_a:.0f}",
            "Elo Equipo B": f"{elo_b:.0f}",
            "Detalle": f"Elo: {equipo_a} {elo_a:.0f} | {equipo_b} {elo_b:.0f}"
        }
    else:
        goles_a, goles_b, n_entrenamiento = predecir_goles_xgb(
            equipo_a,
            equipo_b,
            fecha,
            neutral,
            tipo_partido
        )
        extra = {
            "Partidos de entrenamiento": f"{n_entrenamiento:,}",
            "Detalle": f"Partidos usados para entrenar: {n_entrenamiento:,}"
        }

    matriz, resumen, marcadores = calcular_probabilidades(goles_a, goles_b)

    return {
        "modelo": modelo,
        "goles_a": goles_a,
        "goles_b": goles_b,
        "matriz": matriz,
        "resumen": resumen,
        "marcadores": marcadores,
        "extra": extra,
    }


def crear_tabla_eventos(resultados, comparar, modelo_unico=None):
    filas = []

    for evento, clave in EVENTOS_COMPARACION:
        if comparar:
            poisson_elo = resultados["Poisson + Elo"]["resumen"][clave]
            xgboost = resultados["XGBoost"]["resumen"][clave]
            filas.append({
                "Evento": evento,
                "Poisson + Elo": formato_porcentaje(poisson_elo),
                "XGBoost": formato_porcentaje(xgboost),
                "Diferencia": formato_diferencia(xgboost - poisson_elo),
            })
        else:
            valor = resultados[modelo_unico]["resumen"][clave]
            filas.append({
                "Evento": evento,
                modelo_unico: formato_porcentaje(valor),
            })

    return pd.DataFrame(filas)


def tabla_marcadores(marcadores):
    tabla = marcadores.head(10).copy()
    tabla["Probabilidad"] = tabla["Probabilidad"].map(formato_porcentaje)
    return tabla


def crear_tabla_marcadores_comparativa(resultados, limite=12):
    poisson = resultados["Poisson + Elo"]["marcadores"][["Marcador", "Probabilidad"]].rename(
        columns={"Probabilidad": "Poisson + Elo"}
    )
    xgboost = resultados["XGBoost"]["marcadores"][["Marcador", "Probabilidad"]].rename(
        columns={"Probabilidad": "XGBoost"}
    )

    tabla = poisson.merge(xgboost, on="Marcador", how="outer").fillna(0)
    tabla["Relevancia"] = tabla["Poisson + Elo"] + tabla["XGBoost"]
    tabla["Diferencia"] = tabla["XGBoost"] - tabla["Poisson + Elo"]
    tabla = tabla.sort_values("Relevancia", ascending=False).head(limite)

    tabla = tabla[["Marcador", "Poisson + Elo", "XGBoost", "Diferencia"]].copy()
    tabla["Poisson + Elo"] = tabla["Poisson + Elo"].map(formato_porcentaje)
    tabla["XGBoost"] = tabla["XGBoost"].map(formato_porcentaje)
    tabla["Diferencia"] = tabla["Diferencia"].map(formato_diferencia)

    return tabla


def obtener_favorito(resumen, equipo_a, equipo_b):
    opciones = {
        equipo_a: resumen["Gana equipo A"],
        "Empate": resumen["Empate"],
        equipo_b: resumen["Gana equipo B"],
    }
    return max(opciones, key=opciones.get)


def probabilidad_favorito(resumen, favorito, equipo_a, equipo_b):
    if favorito == equipo_a:
        return resumen["Gana equipo A"]
    if favorito == equipo_b:
        return resumen["Gana equipo B"]
    return resumen["Empate"]


def renderizar_cards_principales(resultados, comparar, equipo_a, equipo_b, modelo_unico=None):
    eventos = [
        (f"Gana {equipo_a}", "Gana equipo A"),
        ("Empate", "Empate"),
        (f"Gana {equipo_b}", "Gana equipo B"),
    ]

    columnas = st.columns(3, gap="small")

    for columna, (titulo, clave) in zip(columnas, eventos):
        with columna:
            with st.container(border=True):
                st.markdown(f"**{titulo}**")
                if comparar:
                    poisson_elo = resultados["Poisson + Elo"]["resumen"][clave]
                    xgboost = resultados["XGBoost"]["resumen"][clave]
                    diferencia = xgboost - poisson_elo

                    c_poisson, c_xgb = st.columns(2, gap="small")
                    c_poisson.metric("Poisson + Elo", formato_porcentaje(poisson_elo))
                    c_xgb.metric("XGBoost", formato_porcentaje(xgboost))
                    st.metric(
                        "Diferencia XGBoost - Poisson + Elo",
                        formato_diferencia(diferencia),
                        delta=formato_diferencia(diferencia),
                        delta_color="normal"
                    )
                else:
                    valor = resultados[modelo_unico]["resumen"][clave]
                    st.metric(modelo_unico, formato_porcentaje(valor))


def renderizar_lectura_rapida(resultados, comparar, equipo_a, equipo_b, modelo_unico=None):
    st.markdown("### Lectura rápida")

    if comparar:
        resumen_poisson = resultados["Poisson + Elo"]["resumen"]
        resumen_xgb = resultados["XGBoost"]["resumen"]
        favorito_poisson = obtener_favorito(resumen_poisson, equipo_a, equipo_b)
        favorito_xgb = obtener_favorito(resumen_xgb, equipo_a, equipo_b)
        coinciden_favorito = favorito_poisson == favorito_xgb

        prob_fav_poisson = probabilidad_favorito(resumen_poisson, favorito_poisson, equipo_a, equipo_b)
        prob_fav_xgb = probabilidad_favorito(resumen_xgb, favorito_xgb, equipo_a, equipo_b)

        if np.isclose(prob_fav_poisson, prob_fav_xgb, atol=0.05):
            conservador = "Ambos modelos tienen una confianza muy similar en su favorito."
        elif prob_fav_poisson < prob_fav_xgb:
            conservador = "Poisson + Elo es más conservador con el favorito."
        else:
            conservador = "XGBoost es más conservador con el favorito."

        top_poisson = resultados["Poisson + Elo"]["marcadores"].head(10)["Marcador"].tolist()
        top_xgb = resultados["XGBoost"]["marcadores"].head(10)["Marcador"].tolist()
        coincidencias = [marcador for marcador in top_poisson if marcador in set(top_xgb)]

        st.markdown(
            "\n".join([
                f"- Favorito según Poisson + Elo: **{favorito_poisson}** ({prob_fav_poisson:.1f}%).",
                f"- Favorito según XGBoost: **{favorito_xgb}** ({prob_fav_xgb:.1f}%).",
                f"- Coincidencia de favorito: **{'sí' if coinciden_favorito else 'no'}**.",
                f"- Modelo más conservador con el favorito: **{conservador}**",
                "- Marcadores que aparecen en el top de ambos modelos: "
                f"**{', '.join(coincidencias) if coincidencias else 'sin coincidencias entre los top 10'}**."
            ])
        )
    else:
        resultado = resultados[modelo_unico]
        favorito = obtener_favorito(resultado["resumen"], equipo_a, equipo_b)
        marcador_principal = resultado["marcadores"].iloc[0]

        st.markdown(
            "\n".join([
                f"- Favorito del modelo: **{favorito}**.",
                f"- Marcador más probable: **{marcador_principal['Marcador']}** ({marcador_principal['Probabilidad']:.1f}%).",
                "- La coincidencia entre modelos se muestra al elegir **Comparar ambos**."
            ])
        )


def estado_obliga_a_atacar(estado):
    return estado in ["Necesita ganar", "Necesita ganar por diferencia de goles"]


def renderizar_lectura_estrategica(contexto, equipo_a, equipo_b):
    st.markdown("### Lectura estratégica")

    estado_a = contexto["estado_a"]
    estado_b = contexto["estado_b"]
    rotacion_a = contexto["rotacion_a"]
    rotacion_b = contexto["rotacion_b"]
    diferencia_a = contexto["diferencia_a"]
    diferencia_b = contexto["diferencia_b"]

    contexto_normal = estado_a == "Normal" and estado_b == "Normal"
    rotacion_baja = rotacion_a == "Bajo" and rotacion_b == "Bajo"

    st.caption(
        "Esta lectura estratégica no recalcula el modelo ni altera las probabilidades numéricas; "
        "solo agrega contexto para interpretar el resultado."
    )

    if contexto_normal and rotacion_baja:
        st.info("Sin señales especiales de contexto competitivo. Interpretar principalmente el modelo estadístico base.")
        return

    condicionados = []
    for equipo, estado in [(equipo_a, estado_a), (equipo_b, estado_b)]:
        if estado != "Normal":
            condicionados.append(f"{equipo}: {estado.lower()}")

    if condicionados:
        lectura_partido = "Partido condicionado por clasificación: " + "; ".join(condicionados) + "."
    else:
        lectura_partido = "El partido parece competitivo en condiciones normales de clasificación."

    equipos_obligados = []
    if estado_obliga_a_atacar(estado_a):
        detalle = f"{equipo_a} está obligado a atacar"
        if estado_a == "Necesita ganar por diferencia de goles" and diferencia_a > 0:
            detalle += f" y necesita una diferencia de {diferencia_a} goles"
        equipos_obligados.append(detalle)
    if estado_obliga_a_atacar(estado_b):
        detalle = f"{equipo_b} está obligado a atacar"
        if estado_b == "Necesita ganar por diferencia de goles" and diferencia_b > 0:
            detalle += f" y necesita una diferencia de {diferencia_b} goles"
        equipos_obligados.append(detalle)

    if equipos_obligados:
        lectura_ataque = "; ".join(equipos_obligados) + "."
    elif estado_a == "Le sirve empatar" or estado_b == "Le sirve empatar":
        lectura_ataque = "Al menos un equipo puede priorizar control y cautela porque le sirve empatar."
    else:
        lectura_ataque = "No hay una obligación clara de atacar más allá del planteamiento normal del partido."

    riesgos = []
    for equipo, estado, rotacion in [(equipo_a, estado_a, rotacion_a), (equipo_b, estado_b, rotacion_b)]:
        if estado in ["Ya clasificado", "Ya eliminado"] or rotacion in ["Medio", "Alto"]:
            riesgos.append(f"{equipo}: rotación {rotacion.lower()} ({estado.lower()})")

    if riesgos:
        lectura_rotacion = "Riesgo de rotación o intensidad irregular: " + "; ".join(riesgos) + "."
    else:
        lectura_rotacion = "No aparecen señales fuertes de rotación por clasificación o eliminación."

    incertidumbre = any([
        estado_a != "Normal",
        estado_b != "Normal",
        rotacion_a in ["Medio", "Alto"],
        rotacion_b in ["Medio", "Alto"],
        diferencia_a > 0,
        diferencia_b > 0,
    ])

    if incertidumbre:
        lectura_incertidumbre = "El contexto aumenta la incertidumbre del marcador porque puede cambiar ritmo, intensidad o gestión de riesgos."
        lectura_cautela = "Conviene interpretar el favorito con cautela: el modelo base no incorpora esta situación competitiva."
    else:
        lectura_incertidumbre = "El contexto no agrega una señal fuerte de incertidumbre adicional sobre el marcador."
        lectura_cautela = "El favorito puede leerse principalmente desde el modelo estadístico base."

    st.markdown(
        "\n".join([
            f"- {lectura_partido}",
            f"- {lectura_ataque}",
            f"- {lectura_rotacion}",
            f"- {lectura_incertidumbre}",
            f"- {lectura_cautela}",
        ])
    )


# ----------------------------
# Interfaz
# ----------------------------

equipos = sorted(pd.concat([
    partidos_jugados["home_team"],
    partidos_jugados["away_team"]
]).unique())

col1, col2, col3 = st.columns(3)

with col1:
    equipo_a = st.selectbox("Equipo A", equipos, index=equipos.index("Portugal") if "Portugal" in equipos else 0)

with col2:
    equipo_b = st.selectbox("Equipo B", equipos, index=equipos.index("Uzbekistan") if "Uzbekistan" in equipos else 1)

with col3:
    fecha = st.date_input("Fecha del partido", value=date(2026, 6, 23))

col4, col5, col6 = st.columns(3)

with col4:
    neutral = st.checkbox("Cancha neutral", value=True)

with col5:
    tipo_partido = st.selectbox("Tipo de partido", ["Mundial", "Oficial", "Amistoso"], index=0)

with col6:
    modelo_elegido = st.selectbox(
        "Modelo",
        ["Comparar ambos", "Poisson + Elo", "XGBoost"],
        index=0
    )

st.markdown("### Contexto competitivo")
contexto_col_a, contexto_col_b = st.columns(2)

with contexto_col_a:
    estado_equipo_a = st.selectbox("Situación Equipo A", ESTADOS_COMPETITIVOS, index=0)
    diferencia_necesaria_a = st.number_input(
        "Diferencia de goles necesaria Equipo A",
        min_value=0,
        max_value=6,
        value=0,
        step=1
    )
    rotacion_equipo_a = st.selectbox("Riesgo de rotación Equipo A", RIESGOS_ROTACION, index=0)

with contexto_col_b:
    estado_equipo_b = st.selectbox("Situación Equipo B", ESTADOS_COMPETITIVOS, index=0)
    diferencia_necesaria_b = st.number_input(
        "Diferencia de goles necesaria Equipo B",
        min_value=0,
        max_value=6,
        value=0,
        step=1
    )
    rotacion_equipo_b = st.selectbox("Riesgo de rotación Equipo B", RIESGOS_ROTACION, index=0)

contexto_competitivo = {
    "estado_a": estado_equipo_a,
    "estado_b": estado_equipo_b,
    "diferencia_a": int(diferencia_necesaria_a),
    "diferencia_b": int(diferencia_necesaria_b),
    "rotacion_a": rotacion_equipo_a,
    "rotacion_b": rotacion_equipo_b,
}

calcular = st.button("Calcular pronóstico", type="primary")

if calcular:
    if equipo_a == equipo_b:
        st.error("Selecciona dos equipos diferentes.")
    else:
        comparar = modelo_elegido == "Comparar ambos"
        modelos = ["Poisson + Elo", "XGBoost"] if comparar else [modelo_elegido]

        with st.spinner("Calculando pronóstico..."):
            resultados = {
                modelo: calcular_modelo(
                    modelo,
                    equipo_a,
                    equipo_b,
                    fecha,
                    neutral,
                    tipo_partido
                )
                for modelo in modelos
            }

        sede = "cancha neutral" if neutral else "localía para Equipo A"
        st.header(f"{equipo_a} vs {equipo_b}")
        st.caption(f"{tipo_partido} | {sede} | {fecha.strftime('%d/%m/%Y')}")

        tab_resumen, tab_marcadores, tab_matriz, tab_detalles = st.tabs([
            "Resumen",
            "Marcadores",
            "Matriz de goles",
            "Detalles técnicos"
        ])

        with tab_resumen:
            renderizar_lectura_rapida(
                resultados,
                comparar,
                equipo_a,
                equipo_b,
                modelo_unico=modelo_elegido if not comparar else None
            )

            renderizar_lectura_estrategica(
                contexto_competitivo,
                equipo_a,
                equipo_b
            )

            renderizar_cards_principales(
                resultados,
                comparar,
                equipo_a,
                equipo_b,
                modelo_unico=modelo_elegido if not comparar else None
            )

            st.markdown("### Tabla comparativa")
            tabla_eventos = crear_tabla_eventos(
                resultados,
                comparar,
                modelo_unico=modelo_elegido if not comparar else None
            )
            st.dataframe(tabla_eventos, use_container_width=True, hide_index=True)

        with tab_marcadores:
            st.markdown("### Marcadores más probables")

            if comparar:
                col_poisson, col_xgb = st.columns(2)

                with col_poisson:
                    st.markdown("#### Top marcadores Poisson + Elo")
                    st.dataframe(
                        tabla_marcadores(resultados["Poisson + Elo"]["marcadores"]),
                        use_container_width=True,
                        hide_index=True
                    )

                with col_xgb:
                    st.markdown("#### Top marcadores XGBoost")
                    st.dataframe(
                        tabla_marcadores(resultados["XGBoost"]["marcadores"]),
                        use_container_width=True,
                        hide_index=True
                    )

                st.markdown("#### Comparativa de marcadores")
                st.dataframe(
                    crear_tabla_marcadores_comparativa(resultados),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.dataframe(
                    tabla_marcadores(resultados[modelo_elegido]["marcadores"]),
                    use_container_width=True,
                    hide_index=True
                )

        with tab_matriz:
            if comparar:
                max_goles_matriz = 5
                vmax_matriz = max(
                    float((resultados["Poisson + Elo"]["matriz"][:max_goles_matriz + 1, :max_goles_matriz + 1] * 100).max()),
                    float((resultados["XGBoost"]["matriz"][:max_goles_matriz + 1, :max_goles_matriz + 1] * 100).max())
                )
                col_poisson, col_xgb = st.columns(2, gap="small")

                with col_poisson:
                    st.markdown("#### Poisson + Elo")
                    fig_poisson = graficar_marcadores(
                        resultados["Poisson + Elo"]["matriz"],
                        equipo_a,
                        equipo_b,
                        max_goles=max_goles_matriz,
                        vmin=0,
                        vmax=vmax_matriz
                    )
                    st.pyplot(fig_poisson, use_container_width=False)

                with col_xgb:
                    st.markdown("#### XGBoost")
                    fig_xgb = graficar_marcadores(
                        resultados["XGBoost"]["matriz"],
                        equipo_a,
                        equipo_b,
                        max_goles=max_goles_matriz,
                        vmin=0,
                        vmax=vmax_matriz
                    )
                    st.pyplot(fig_xgb, use_container_width=False)
            else:
                fig = graficar_marcadores(
                    resultados[modelo_elegido]["matriz"],
                    equipo_a,
                    equipo_b
                )
                st.pyplot(fig, use_container_width=False)

        with tab_detalles:
            filas_detalle = []

            for modelo, resultado in resultados.items():
                fila = {
                    "Modelo": modelo,
                    f"Goles esperados {equipo_a}": f"{resultado['goles_a']:.2f}",
                    f"Goles esperados {equipo_b}": f"{resultado['goles_b']:.2f}",
                    "Detalle": resultado["extra"]["Detalle"],
                }
                filas_detalle.append(fila)

            st.dataframe(
                pd.DataFrame(filas_detalle),
                use_container_width=True,
                hide_index=True
            )
            st.caption("Diferencia = XGBoost menos Poisson + Elo, expresada en puntos porcentuales.")

with st.expander("Notas importantes"):
    st.write(
        """
        Este predictor es experimental. Usa resultados históricos, Elo, Poisson y XGBoost.
        No considera lesiones, alineaciones confirmadas, clima, cuotas de apuestas, tácticas ni noticias de último minuto.
        Las probabilidades son aproximaciones estadísticas, no garantías.
        """
    )
