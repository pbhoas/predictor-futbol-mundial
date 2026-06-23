
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


def graficar_marcadores(matriz, equipo_a, equipo_b, max_goles=5):
    datos = matriz[:max_goles + 1, :max_goles + 1] * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(datos)

    ax.set_title(f"Probabilidad por marcador\n{equipo_a} vs {equipo_b}")
    ax.set_xlabel(f"Goles de {equipo_b}")
    ax.set_ylabel(f"Goles de {equipo_a}")

    ax.set_xticks(range(max_goles + 1))
    ax.set_yticks(range(max_goles + 1))

    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            ax.text(j, i, f"{datos[i, j]:.1f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, label="Probabilidad (%)")
    fig.tight_layout()
    return fig


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
    modelo_elegido = st.selectbox("Modelo", ["XGBoost", "Poisson + Elo"], index=0)

calcular = st.button("Calcular pronóstico", type="primary")

if calcular:
    if equipo_a == equipo_b:
        st.error("Selecciona dos equipos diferentes.")
    else:
        with st.spinner("Calculando pronóstico..."):
            if modelo_elegido == "Poisson + Elo":
                goles_a, goles_b, elo_a, elo_b = predecir_goles_poisson_elo(
                    equipo_a,
                    equipo_b,
                    fecha,
                    neutral,
                    tipo_partido
                )
                extra = f"Elo: {equipo_a} {elo_a:.0f} | {equipo_b} {elo_b:.0f}"
            else:
                goles_a, goles_b, n_entrenamiento = predecir_goles_xgb(
                    equipo_a,
                    equipo_b,
                    fecha,
                    neutral,
                    tipo_partido
                )
                extra = f"Partidos usados para entrenar: {n_entrenamiento}"

            matriz, resumen, marcadores = calcular_probabilidades(goles_a, goles_b)

        st.subheader(f"{equipo_a} vs {equipo_b}")
        st.write(f"**Modelo:** {modelo_elegido}")
        st.write(extra)
        st.write(f"**Goles esperados:** {equipo_a} {goles_a:.2f} | {equipo_b} {goles_b:.2f}")

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Gana {equipo_a}", f"{resumen['Gana equipo A']:.1f}%")
        c2.metric("Empate", f"{resumen['Empate']:.1f}%")
        c3.metric(f"Gana {equipo_b}", f"{resumen['Gana equipo B']:.1f}%")

        st.markdown("### Mercados principales")

        tabla_eventos = pd.DataFrame([
            ["Más de 1.5 goles", resumen["Más de 1.5 goles"]],
            ["Menos de 1.5 goles", resumen["Menos de 1.5 goles"]],
            ["Más de 2.5 goles", resumen["Más de 2.5 goles"]],
            ["Menos de 2.5 goles", resumen["Menos de 2.5 goles"]],
            ["Más de 3.5 goles", resumen["Más de 3.5 goles"]],
            ["Menos de 3.5 goles", resumen["Menos de 3.5 goles"]],
            ["Ambos marcan", resumen["Ambos marcan"]],
            ["No marcan ambos", resumen["No marcan ambos"]],
        ], columns=["Evento", "Probabilidad"])

        tabla_eventos["Probabilidad"] = tabla_eventos["Probabilidad"].map(lambda x: f"{x:.1f}%")
        st.dataframe(tabla_eventos, use_container_width=True, hide_index=True)

        st.markdown("### 10 marcadores más probables")
        tabla_marcadores = marcadores.head(10).copy()
        tabla_marcadores["Probabilidad"] = tabla_marcadores["Probabilidad"].round(1).astype(str) + "%"
        st.dataframe(tabla_marcadores, use_container_width=True, hide_index=True)

        st.markdown("### Matriz de marcadores")
        fig = graficar_marcadores(matriz, equipo_a, equipo_b)
        st.pyplot(fig)

with st.expander("Notas importantes"):
    st.write(
        """
        Este predictor es experimental. Usa resultados históricos, Elo, Poisson y XGBoost.
        No considera lesiones, alineaciones confirmadas, clima, cuotas de apuestas, tácticas ni noticias de último minuto.
        Las probabilidades son aproximaciones estadísticas, no garantías.
        """
    )
