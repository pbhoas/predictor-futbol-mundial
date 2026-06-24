
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
FORMAS_TORNEO = [
    "Sin ajuste",
    "Mejor de lo esperado",
    "Mucho mejor de lo esperado",
    "Peor de lo esperado",
    "Mucho peor de lo esperado",
]
RESULTADO_KEYS = ["Gana equipo A", "Empate", "Gana equipo B"]


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


def limitar_goles(valor):
    return float(np.clip(valor, 0.15, 5.50))


def factor_forma_reciente(forma, partidos_considerados):
    ajustes_base = {
        "Sin ajuste": 0.00,
        "Mejor de lo esperado": 0.04,
        "Mucho mejor de lo esperado": 0.08,
        "Peor de lo esperado": -0.04,
        "Mucho peor de lo esperado": -0.08,
    }
    escalas = {0: 0.00, 1: 0.50, 2: 0.75, 3: 1.00}
    return ajustes_base[forma] * escalas[int(partidos_considerados)]


def texto_porcentaje_factor(valor):
    return f"{valor * 100:.1f}%"


def aplicar_reglas_equipo_goles(factores, contexto, lado, reglas):
    estado = contexto[f"estado_{lado}"]
    rotacion = contexto[f"rotacion_{lado}"]
    diferencia = contexto[f"diferencia_{lado}"]
    equipo = "Equipo A" if lado == "a" else "Equipo B"
    rival = "b" if lado == "a" else "a"

    if estado == "Necesita ganar":
        factores[lado] += 0.05
        factores[rival] += 0.025
        reglas.append(
            f"{equipo} necesita ganar: se aumentaron sus goles esperados 5.0% y los del rival 2.5% por mayor exposición."
        )

    elif estado == "Le sirve empatar":
        factores[lado] -= 0.045
        reglas.append(
            f"{equipo} le sirve empatar: se redujeron sus goles esperados 4.5% para reflejar menor ritmo/agresividad."
        )

    elif estado == "Necesita ganar por diferencia de goles":
        ajuste_propio = min(0.08 + diferencia * 0.012, 0.15)
        ajuste_rival = min(0.04 + diferencia * 0.007, 0.08)
        factores[lado] += ajuste_propio
        factores[rival] += ajuste_rival
        extra = f" con objetivo de {diferencia} goles" if diferencia > 0 else ""
        reglas.append(
            f"{equipo} necesita ganar por diferencia{extra}: se aumentaron sus goles esperados {texto_porcentaje_factor(ajuste_propio)} y los del rival {texto_porcentaje_factor(ajuste_rival)} por exposición defensiva."
        )

    elif estado == "Ya clasificado":
        if rotacion == "Medio":
            factores[lado] -= 0.04
            factores[rival] += 0.02
            reglas.append(
                f"{equipo} ya clasificado con rotación media: se redujeron sus goles esperados 4.0% y se elevó 2.0% la expectativa del rival."
            )
        elif rotacion == "Alto":
            factores[lado] -= 0.08
            factores[rival] += 0.04
            reglas.append(
                f"{equipo} ya clasificado con rotación alta: se redujeron sus goles esperados 8.0% y se elevó 4.0% la expectativa del rival."
            )
        else:
            reglas.append(
                f"{equipo} ya clasificado con rotación baja: no se cambió su expectativa de goles, solo se marca cautela interpretativa."
            )

    elif estado == "Ya eliminado":
        if rotacion == "Alto":
            factores[lado] -= 0.05
            reglas.append(
                f"{equipo} ya eliminado con rotación alta: se redujeron sus goles esperados 5.0%, sin asumir caída automática por eliminación."
            )
        elif rotacion in ["Bajo", "Medio"]:
            reglas.append(
                f"{equipo} ya eliminado con rotación {rotacion.lower()}: no se penalizó automáticamente su rendimiento."
            )


def aplicar_forma_reciente_goles(factores, contexto, lado, reglas):
    forma = contexto[f"forma_{lado}"]
    partidos = int(contexto["partidos_forma"])
    equipo = "Equipo A" if lado == "a" else "Equipo B"
    ajuste = factor_forma_reciente(forma, partidos)

    if partidos == 0:
        if forma != "Sin ajuste":
            reglas.append(
                f"{equipo} tiene forma reciente marcada como '{forma}', pero con 0 partidos considerados no se aplicó ajuste."
            )
        return

    if np.isclose(ajuste, 0.0):
        return

    factores[lado] += ajuste
    direccion = "aumentó" if ajuste > 0 else "redujo"
    reglas.append(
        f"{equipo} viene rindiendo {forma.lower()} en el torneo: se {direccion} su expectativa ofensiva {abs(ajuste) * 100:.1f}% según {partidos} partido(s) considerados."
    )


def aplicar_ajuste_contexto_goles(goles_a, goles_b, contexto):
    factores = {"a": 1.0, "b": 1.0}
    reglas = []

    aplicar_reglas_equipo_goles(factores, contexto, "a", reglas)
    aplicar_reglas_equipo_goles(factores, contexto, "b", reglas)
    aplicar_forma_reciente_goles(factores, contexto, "a", reglas)
    aplicar_forma_reciente_goles(factores, contexto, "b", reglas)

    if estado_obliga_a_atacar(contexto["estado_a"]) and estado_obliga_a_atacar(contexto["estado_b"]):
        factores["a"] += 0.02
        factores["b"] += 0.02
        reglas.append(
            "Ambos equipos están obligados a atacar: se añadió 2.0% a cada expectativa de gol por perfil de partido abierto."
        )

    goles_a_ajustado = limitar_goles(goles_a * max(0.20, factores["a"]))
    goles_b_ajustado = limitar_goles(goles_b * max(0.20, factores["b"]))

    if not reglas:
        reglas.append("Sin ajuste contextual ni de forma aplicado: se mantienen los goles esperados base.")

    return goles_a_ajustado, goles_b_ajustado, reglas


def aplicar_ajuste_contexto(resultado_modelo, contexto):
    goles_a, goles_b, reglas = aplicar_ajuste_contexto_goles(
        resultado_modelo["goles_a"],
        resultado_modelo["goles_b"],
        contexto
    )
    matriz, resumen, marcadores = calcular_probabilidades(goles_a, goles_b)

    return {
        "goles_a": goles_a,
        "goles_b": goles_b,
        "matriz": matriz,
        "resumen": resumen,
        "marcadores": marcadores,
        "reglas": reglas,
    }


def crear_resultados_ajustados(resultados, contexto):
    return {
        modelo: aplicar_ajuste_contexto(resultado, contexto)
        for modelo, resultado in resultados.items()
    }


def crear_tabla_cambio_contexto(resultados, ajustes_contexto, comparar=False, modelo_unico=None):
    filas = []

    for evento, clave in EVENTOS_COMPARACION:
        if comparar:
            poisson_base = resultados["Poisson + Elo"]["resumen"][clave]
            poisson_ajustado = ajustes_contexto["Poisson + Elo"]["resumen"][clave]
            xgb_base = resultados["XGBoost"]["resumen"][clave]
            xgb_ajustado = ajustes_contexto["XGBoost"]["resumen"][clave]
            filas.append({
                "Evento": evento,
                "Poisson + Elo base": formato_porcentaje(poisson_base),
                "Poisson + Elo ajustado": formato_porcentaje(poisson_ajustado),
                "Cambio Poisson": formato_diferencia(poisson_ajustado - poisson_base),
                "XGBoost base": formato_porcentaje(xgb_base),
                "XGBoost ajustado": formato_porcentaje(xgb_ajustado),
                "Cambio XGBoost": formato_diferencia(xgb_ajustado - xgb_base),
            })
        else:
            base = resultados[modelo_unico]["resumen"][clave]
            ajustado = ajustes_contexto[modelo_unico]["resumen"][clave]
            filas.append({
                "Evento": evento,
                f"{modelo_unico} base": formato_porcentaje(base),
                f"{modelo_unico} ajustado": formato_porcentaje(ajustado),
                "Cambio": formato_diferencia(ajustado - base),
            })

    return pd.DataFrame(filas)


def tabla_marcadores(marcadores):
    tabla = marcadores.head(10).copy()
    tabla["Probabilidad"] = tabla["Probabilidad"].map(formato_porcentaje)
    return tabla


def crear_tabla_marcadores_base_ajustada(resultados, ajustes_contexto, comparar, modelo_unico=None, limite=15):
    if comparar:
        columnas = []
        for modelo in ["Poisson + Elo", "XGBoost"]:
            base = resultados[modelo]["marcadores"][["Marcador", "Probabilidad"]].rename(
                columns={"Probabilidad": f"{modelo} base"}
            )
            ajustada = ajustes_contexto[modelo]["marcadores"][["Marcador", "Probabilidad"]].rename(
                columns={"Probabilidad": f"{modelo} ajustado"}
            )
            columnas.extend([base, ajustada])

        tabla = columnas[0]
        for columna in columnas[1:]:
            tabla = tabla.merge(columna, on="Marcador", how="outer")
        tabla = tabla.fillna(0)
        columnas_probabilidad = [col for col in tabla.columns if col != "Marcador"]
        tabla["Relevancia"] = tabla[columnas_probabilidad].sum(axis=1)
        tabla = tabla.sort_values("Relevancia", ascending=False).head(limite)
        tabla = tabla[[
            "Marcador",
            "Poisson + Elo base",
            "Poisson + Elo ajustado",
            "XGBoost base",
            "XGBoost ajustado",
        ]].copy()
    else:
        base = resultados[modelo_unico]["marcadores"][["Marcador", "Probabilidad"]].rename(
            columns={"Probabilidad": f"{modelo_unico} base"}
        )
        ajustada = ajustes_contexto[modelo_unico]["marcadores"][["Marcador", "Probabilidad"]].rename(
            columns={"Probabilidad": f"{modelo_unico} ajustado"}
        )
        tabla = base.merge(ajustada, on="Marcador", how="outer").fillna(0)
        columnas_probabilidad = [f"{modelo_unico} base", f"{modelo_unico} ajustado"]
        tabla["Relevancia"] = tabla[columnas_probabilidad].sum(axis=1)
        tabla = tabla.sort_values("Relevancia", ascending=False).head(limite)
        tabla = tabla[["Marcador", f"{modelo_unico} base", f"{modelo_unico} ajustado"]].copy()

    for columna in tabla.columns:
        if columna != "Marcador":
            tabla[columna] = tabla[columna].map(formato_porcentaje)

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


def cambio_principal_resultado(resultado_base, resultado_ajustado):
    cambios = {
        clave: resultado_ajustado["resumen"][clave] - resultado_base["resumen"][clave]
        for clave in RESULTADO_KEYS
    }
    clave_principal = max(cambios, key=lambda clave: abs(cambios[clave]))
    return clave_principal, cambios[clave_principal]


def estado_obliga_a_atacar(estado):
    return estado in ["Necesita ganar", "Necesita ganar por diferencia de goles"]


def reglas_unicas(ajustes_contexto):
    reglas = []
    for ajuste in ajustes_contexto.values():
        for regla in ajuste["reglas"]:
            if regla not in reglas:
                reglas.append(regla)
    return reglas


def renderizar_cards_resultado(resultados, ajustes_contexto, comparar, equipo_a, equipo_b, modelo_unico=None):
    st.markdown("### Probabilidades base y ajustadas por contexto")
    st.caption(
        "El ajuste por contexto competitivo y forma reciente es una capa heurística sobre el modelo base; "
        "no reemplaza el modelo estadístico ni está entrenado con datos históricos de contexto."
    )

    eventos = [
        (f"Gana {equipo_a}", "Gana equipo A"),
        ("Empate", "Empate"),
        (f"Gana {equipo_b}", "Gana equipo B"),
    ]
    modelos = ["Poisson + Elo", "XGBoost"] if comparar else [modelo_unico]
    columnas = st.columns(3, gap="small")

    for columna, (titulo, clave) in zip(columnas, eventos):
        with columna:
            with st.container(border=True):
                st.markdown(f"**{titulo}**")
                for modelo in modelos:
                    base = resultados[modelo]["resumen"][clave]
                    ajustado = ajustes_contexto[modelo]["resumen"][clave]
                    cambio = ajustado - base
                    st.markdown(f"**{modelo}**")
                    col_base, col_ajustada, col_cambio = st.columns(3, gap="small")
                    col_base.metric("Base", formato_porcentaje(base))
                    col_ajustada.metric("Ajustado", formato_porcentaje(ajustado))
                    col_cambio.metric(
                        "Cambio",
                        formato_diferencia(cambio),
                        delta=formato_diferencia(cambio),
                        delta_color="normal"
                    )


def renderizar_lectura_rapida(resultados, ajustes_contexto, comparar, equipo_a, equipo_b, modelo_unico=None):
    st.markdown("### Lectura rápida")
    st.caption("Prioriza el escenario ajustado; el escenario base queda como referencia estadística sin contexto.")

    if comparar:
        resumen_poisson = ajustes_contexto["Poisson + Elo"]["resumen"]
        resumen_xgb = ajustes_contexto["XGBoost"]["resumen"]
        favorito_poisson = obtener_favorito(resumen_poisson, equipo_a, equipo_b)
        favorito_xgb = obtener_favorito(resumen_xgb, equipo_a, equipo_b)
        coinciden_favorito = favorito_poisson == favorito_xgb

        prob_fav_poisson = probabilidad_favorito(resumen_poisson, favorito_poisson, equipo_a, equipo_b)
        prob_fav_xgb = probabilidad_favorito(resumen_xgb, favorito_xgb, equipo_a, equipo_b)

        if np.isclose(prob_fav_poisson, prob_fav_xgb, atol=0.05):
            conservador = "Ambos modelos tienen una confianza muy similar en su favorito ajustado."
        elif prob_fav_poisson < prob_fav_xgb:
            conservador = "Poisson + Elo es más conservador con el favorito ajustado."
        else:
            conservador = "XGBoost es más conservador con el favorito ajustado."

        top_poisson = ajustes_contexto["Poisson + Elo"]["marcadores"].head(10)["Marcador"].tolist()
        top_xgb = ajustes_contexto["XGBoost"]["marcadores"].head(10)["Marcador"].tolist()
        coincidencias = [marcador for marcador in top_poisson if marcador in set(top_xgb)]

        cambio_poisson = cambio_principal_resultado(resultados["Poisson + Elo"], ajustes_contexto["Poisson + Elo"])
        cambio_xgb = cambio_principal_resultado(resultados["XGBoost"], ajustes_contexto["XGBoost"])

        st.markdown(
            "\n".join([
                f"- Favorito ajustado según Poisson + Elo: **{favorito_poisson}** ({prob_fav_poisson:.1f}%).",
                f"- Favorito ajustado según XGBoost: **{favorito_xgb}** ({prob_fav_xgb:.1f}%).",
                f"- Coincidencia de favorito ajustado: **{'sí' if coinciden_favorito else 'no'}**.",
                f"- Modelo más conservador con el favorito: **{conservador}**",
                "- Mayor cambio por contexto: "
                f"Poisson + Elo mueve **{cambio_poisson[0]}** ({formato_diferencia(cambio_poisson[1])}); "
                f"XGBoost mueve **{cambio_xgb[0]}** ({formato_diferencia(cambio_xgb[1])}).",
                "- Marcadores ajustados que aparecen en el top de ambos modelos: "
                f"**{', '.join(coincidencias) if coincidencias else 'sin coincidencias entre los top 10'}**."
            ])
        )
    else:
        base = resultados[modelo_unico]
        ajustado = ajustes_contexto[modelo_unico]
        favorito_base = obtener_favorito(base["resumen"], equipo_a, equipo_b)
        favorito_ajustado = obtener_favorito(ajustado["resumen"], equipo_a, equipo_b)
        marcador_principal = ajustado["marcadores"].iloc[0]
        cambio = cambio_principal_resultado(base, ajustado)

        st.markdown(
            "\n".join([
                f"- Favorito base: **{favorito_base}**.",
                f"- Favorito ajustado por contexto y forma: **{favorito_ajustado}**.",
                f"- Mayor cambio por contexto: **{cambio[0]}** ({formato_diferencia(cambio[1])}).",
                f"- Marcador ajustado más probable: **{marcador_principal['Marcador']}** ({marcador_principal['Probabilidad']:.1f}%).",
            ])
        )


def renderizar_lectura_estrategica(contexto, equipo_a, equipo_b, ajustes_contexto=None):
    st.markdown("### Lectura estratégica")

    estado_a = contexto["estado_a"]
    estado_b = contexto["estado_b"]
    rotacion_a = contexto["rotacion_a"]
    rotacion_b = contexto["rotacion_b"]
    diferencia_a = contexto["diferencia_a"]
    diferencia_b = contexto["diferencia_b"]
    forma_a = contexto["forma_a"]
    forma_b = contexto["forma_b"]
    partidos_forma = contexto["partidos_forma"]

    contexto_normal = estado_a == "Normal" and estado_b == "Normal"
    rotacion_baja = rotacion_a == "Bajo" and rotacion_b == "Bajo"
    sin_forma = partidos_forma == 0 or (forma_a == "Sin ajuste" and forma_b == "Sin ajuste")

    st.caption(
        "El modelo base es estadístico. El ajuste por contexto y forma reciente recalcula goles esperados de manera heurística; "
        "no representa certeza, garantía ni un modelo entrenado con datos históricos de contexto."
    )

    if contexto_normal and rotacion_baja and sin_forma:
        st.info("Sin señales especiales de contexto competitivo. Interpretar principalmente el modelo estadístico base.")
    else:
        condicionados = []
        for equipo, estado in [(equipo_a, estado_a), (equipo_b, estado_b)]:
            if estado != "Normal":
                condicionados.append(f"{equipo}: {estado.lower()}")

        if condicionados:
            lectura_partido = "Partido condicionado por clasificación: " + "; ".join(condicionados) + "."
        else:
            lectura_partido = "El partido parece normal en clasificación, pero puede matizarse por forma reciente o rotación."

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

        formas = []
        if partidos_forma > 0:
            for equipo, forma in [(equipo_a, forma_a), (equipo_b, forma_b)]:
                if forma != "Sin ajuste":
                    formas.append(f"{equipo}: {forma.lower()}")

        if formas:
            lectura_forma = (
                f"Forma reciente considerada sobre {partidos_forma} partido(s): " + "; ".join(formas) + "."
            )
        elif forma_a != "Sin ajuste" or forma_b != "Sin ajuste":
            lectura_forma = "Hay forma reciente seleccionada, pero con 0 partidos considerados no se aplica ajuste."
        else:
            lectura_forma = "No se agregó ajuste por forma reciente en el torneo."

        incertidumbre = any([
            estado_a != "Normal",
            estado_b != "Normal",
            rotacion_a in ["Medio", "Alto"],
            rotacion_b in ["Medio", "Alto"],
            diferencia_a > 0,
            diferencia_b > 0,
            bool(formas),
        ])

        if incertidumbre:
            lectura_incertidumbre = "El contexto aumenta la incertidumbre del marcador porque puede cambiar ritmo, intensidad, exposición o gestión de riesgos."
            lectura_cautela = "Conviene interpretar el favorito con cautela y comparar siempre base contra ajustado."
        else:
            lectura_incertidumbre = "El contexto no agrega una señal fuerte de incertidumbre adicional sobre el marcador."
            lectura_cautela = "El favorito puede leerse principalmente desde el modelo estadístico base."

        st.markdown(
            "\n".join([
                f"- {lectura_partido}",
                f"- {lectura_ataque}",
                f"- {lectura_rotacion}",
                f"- {lectura_forma}",
                f"- {lectura_incertidumbre}",
                f"- {lectura_cautela}",
            ])
        )

    if ajustes_contexto:
        reglas = reglas_unicas(ajustes_contexto)
        st.markdown("**Ajustes aplicados:**")
        st.markdown("\n".join([f"- {regla}" for regla in reglas]))


def renderizar_marcadores(resultados, ajustes_contexto, comparar, modelo_unico=None):
    st.markdown("### Marcadores más probables")

    modelos = ["Poisson + Elo", "XGBoost"] if comparar else [modelo_unico]
    for modelo in modelos:
        st.markdown(f"#### {modelo}: base vs ajustado")
        col_base, col_ajustado = st.columns(2, gap="small")
        with col_base:
            st.markdown("**Top marcadores base**")
            st.dataframe(
                tabla_marcadores(resultados[modelo]["marcadores"]),
                use_container_width=True,
                hide_index=True
            )
        with col_ajustado:
            st.markdown("**Top marcadores ajustados**")
            st.dataframe(
                tabla_marcadores(ajustes_contexto[modelo]["marcadores"]),
                use_container_width=True,
                hide_index=True
            )

    st.markdown("#### Comparativa de marcadores base y ajustados")
    st.dataframe(
        crear_tabla_marcadores_base_ajustada(
            resultados,
            ajustes_contexto,
            comparar=comparar,
            modelo_unico=modelo_unico
        ),
        use_container_width=True,
        hide_index=True
    )


def renderizar_matriz(resultados, ajustes_contexto, comparar, equipo_a, equipo_b, modelo_unico=None):
    vista = st.radio(
        "Vista de matriz",
        ["Base", "Ajustada por contexto"],
        horizontal=True
    )
    fuente = resultados if vista == "Base" else ajustes_contexto
    modelos = ["Poisson + Elo", "XGBoost"] if comparar else [modelo_unico]
    max_goles_matriz = 5
    vmax_matriz = max(
        float((fuente[modelo]["matriz"][:max_goles_matriz + 1, :max_goles_matriz + 1] * 100).max())
        for modelo in modelos
    )

    columnas = st.columns(len(modelos), gap="small")
    for columna, modelo in zip(columnas, modelos):
        with columna:
            st.markdown(f"#### {modelo}")
            fig = graficar_marcadores(
                fuente[modelo]["matriz"],
                equipo_a,
                equipo_b,
                max_goles=max_goles_matriz,
                vmin=0,
                vmax=vmax_matriz
            )
            st.pyplot(fig, use_container_width=False)


def renderizar_detalles_ajuste_contexto(contexto, resultados, ajustes_contexto, equipo_a, equipo_b):
    st.markdown("### Goles esperados base vs ajustados")
    filas_detalle = []

    for modelo, resultado in resultados.items():
        ajuste = ajustes_contexto[modelo]
        filas_detalle.append({
            "Modelo": modelo,
            f"Base {equipo_a}": f"{resultado['goles_a']:.2f}",
            f"Ajustado {equipo_a}": f"{ajuste['goles_a']:.2f}",
            f"Cambio {equipo_a}": f"{ajuste['goles_a'] - resultado['goles_a']:+.2f}",
            f"Base {equipo_b}": f"{resultado['goles_b']:.2f}",
            f"Ajustado {equipo_b}": f"{ajuste['goles_b']:.2f}",
            f"Cambio {equipo_b}": f"{ajuste['goles_b'] - resultado['goles_b']:+.2f}",
            "Detalle modelo base": resultado["extra"]["Detalle"],
        })

    st.dataframe(pd.DataFrame(filas_detalle), use_container_width=True, hide_index=True)
    st.caption("Diferencia en tablas de probabilidad = valor ajustado menos valor base, expresada en puntos porcentuales.")

    st.markdown("### Contexto competitivo y forma reciente")
    tabla_contexto = pd.DataFrame([
        {
            "Equipo": equipo_a,
            "Situación": contexto["estado_a"],
            "Diferencia necesaria": contexto["diferencia_a"],
            "Riesgo de rotación": contexto["rotacion_a"],
            "Forma reciente": contexto["forma_a"],
            "Partidos considerados": contexto["partidos_forma"],
        },
        {
            "Equipo": equipo_b,
            "Situación": contexto["estado_b"],
            "Diferencia necesaria": contexto["diferencia_b"],
            "Riesgo de rotación": contexto["rotacion_b"],
            "Forma reciente": contexto["forma_b"],
            "Partidos considerados": contexto["partidos_forma"],
        },
    ])
    st.dataframe(tabla_contexto, use_container_width=True, hide_index=True)

    st.markdown("### Reglas de ajuste aplicadas")
    st.caption("Ajuste heurístico, no entrenado. No reemplaza el modelo estadístico base ni garantiza el resultado.")
    filas_reglas = []
    for modelo, ajuste in ajustes_contexto.items():
        for regla in ajuste["reglas"]:
            filas_reglas.append({"Modelo": modelo, "Regla de ajuste aplicada": regla})

    st.dataframe(pd.DataFrame(filas_reglas), use_container_width=True, hide_index=True)


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

st.markdown("### Forma reciente en el torneo")
forma_col_a, forma_col_b, forma_col_partidos = st.columns(3)

with forma_col_a:
    forma_equipo_a = st.selectbox("Forma reciente Equipo A", FORMAS_TORNEO, index=0)

with forma_col_b:
    forma_equipo_b = st.selectbox("Forma reciente Equipo B", FORMAS_TORNEO, index=0)

with forma_col_partidos:
    partidos_forma = st.selectbox("Partidos recientes del torneo considerados", [0, 1, 2, 3], index=0)

contexto_competitivo = {
    "estado_a": estado_equipo_a,
    "estado_b": estado_equipo_b,
    "diferencia_a": int(diferencia_necesaria_a),
    "diferencia_b": int(diferencia_necesaria_b),
    "rotacion_a": rotacion_equipo_a,
    "rotacion_b": rotacion_equipo_b,
    "forma_a": forma_equipo_a,
    "forma_b": forma_equipo_b,
    "partidos_forma": int(partidos_forma),
}

calcular = st.button("Calcular pron?stico", type="primary")

if calcular:
    if equipo_a == equipo_b:
        st.error("Selecciona dos equipos diferentes.")
        st.session_state.pop("ultimo_pronostico", None)
    else:
        comparar = modelo_elegido == "Comparar ambos"
        modelos = ["Poisson + Elo", "XGBoost"] if comparar else [modelo_elegido]

        with st.spinner("Calculando pron?stico..."):
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
            ajustes_contexto = crear_resultados_ajustados(resultados, contexto_competitivo)

        st.session_state["ultimo_pronostico"] = {
            "resultados": resultados,
            "ajustes_contexto": ajustes_contexto,
            "contexto_competitivo": contexto_competitivo,
            "comparar": comparar,
            "modelo_elegido": modelo_elegido,
            "equipo_a": equipo_a,
            "equipo_b": equipo_b,
            "fecha": fecha,
            "neutral": neutral,
            "tipo_partido": tipo_partido,
        }

if "ultimo_pronostico" in st.session_state:
    pronostico = st.session_state["ultimo_pronostico"]
    resultados = pronostico["resultados"]
    ajustes_contexto = pronostico["ajustes_contexto"]
    contexto_competitivo = pronostico["contexto_competitivo"]
    comparar = pronostico["comparar"]
    modelo_elegido = pronostico["modelo_elegido"]
    equipo_a = pronostico["equipo_a"]
    equipo_b = pronostico["equipo_b"]
    fecha = pronostico["fecha"]
    neutral = pronostico["neutral"]
    tipo_partido = pronostico["tipo_partido"]

    sede = "cancha neutral" if neutral else "local?a para Equipo A"
    st.header(f"{equipo_a} vs {equipo_b}")
    st.caption(f"{tipo_partido} | {sede} | {fecha.strftime('%d/%m/%Y')}")

    tab_resumen, tab_marcadores, tab_matriz, tab_detalles = st.tabs([
        "Resumen",
        "Marcadores",
        "Matriz de goles",
        "Detalles t?cnicos"
    ])

    with tab_resumen:
        renderizar_lectura_rapida(
            resultados,
            ajustes_contexto,
            comparar,
            equipo_a,
            equipo_b,
            modelo_unico=modelo_elegido if not comparar else None
        )

        renderizar_lectura_estrategica(
            contexto_competitivo,
            equipo_a,
            equipo_b,
            ajustes_contexto=ajustes_contexto
        )

        renderizar_cards_resultado(
            resultados,
            ajustes_contexto,
            comparar,
            equipo_a,
            equipo_b,
            modelo_unico=modelo_elegido if not comparar else None
        )

        st.markdown("### Cambio por contexto")
        tabla_eventos = crear_tabla_cambio_contexto(
            resultados,
            ajustes_contexto,
            comparar=comparar,
            modelo_unico=modelo_elegido if not comparar else None
        )
        st.dataframe(tabla_eventos, use_container_width=True, hide_index=True)

    with tab_marcadores:
        renderizar_marcadores(
            resultados,
            ajustes_contexto,
            comparar,
            modelo_unico=modelo_elegido if not comparar else None
        )

    with tab_matriz:
        renderizar_matriz(
            resultados,
            ajustes_contexto,
            comparar,
            equipo_a,
            equipo_b,
            modelo_unico=modelo_elegido if not comparar else None
        )

    with tab_detalles:
        renderizar_detalles_ajuste_contexto(
            contexto_competitivo,
            resultados,
            ajustes_contexto,
            equipo_a,
            equipo_b
        )

with st.expander("Notas importantes"):
    st.write(
        """
        Este predictor es experimental. Usa resultados históricos, Elo, Poisson y XGBoost.
        El modelo base es estadístico; el ajuste por contexto competitivo y forma reciente es heurístico,
        separado y todavía no entrenado con datos históricos de contexto.
        No considera lesiones, alineaciones confirmadas, clima, cuotas de apuestas, tácticas ni noticias de último minuto.
        Las probabilidades son aproximaciones estadísticas, no garantías ni certezas.
        """
    )
