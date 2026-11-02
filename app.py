import os
import duckdb
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Análisis CPC de agua", page_icon="💧", layout="wide")

st.markdown("""
<style>
.main {background-color: #f6f9fc;}
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero {padding: 1.4rem 1.7rem; border-radius: 18px; color: white;
       background: linear-gradient(120deg,#073b63,#087eaa);}
.hero h1 {color:white; margin:0;}
.hero p {color:#e4f5ff; margin:.35rem 0 0 0;}
div[data-testid="stMetric"] {background:white; padding:14px; border-radius:12px;
                             border:1px solid #e6edf3;}
</style>
""", unsafe_allow_html=True)

FEATURES = ["temperatura_max", "humedad_max", "personas_familia", "consumo_bidon_pc"]
LABELS = {
    "temperatura_max": "Temperatura máxima (°C)",
    "humedad_max": "Humedad máxima (%)",
    "personas_familia": "Personas por familia",
    "consumo_bidon_pc": "Consumo de bidón (L/hab-día)",
}

@st.cache_data
def load_data(seed: int, n: int):
    # Datos sintéticos reproducibles. Los coeficientes conservan la dirección
    # de las relaciones que se estudian en la práctica.
    rng = np.random.default_rng(seed)
    temp = rng.uniform(18, 32, n)
    humidity = rng.uniform(50, 90, n)
    family = rng.uniform(3, 7, n)
    bidon = rng.uniform(0.2, 2.0, n)
    noise = rng.normal(0, 2.2, n)
    cpc = 94.0 + 2.5*temp - 0.8*humidity - 5.2*family + 15.0*bidon + noise
    df = pd.DataFrame({
        "residencia_id": np.arange(1, n+1),
        "temperatura_max": np.round(temp, 2),
        "humedad_max": np.round(humidity, 2),
        "personas_familia": np.round(family, 2),
        "consumo_bidon_pc": np.round(bidon, 2),
        "cpc_consumo": np.round(cpc, 2),
    })
    con = duckdb.connect(database=":memory:")
    con.register("df_source", df)
    con.execute("CREATE TABLE telemetria_agua AS SELECT * FROM df_source")
    result = con.execute("SELECT * FROM telemetria_agua ORDER BY residencia_id").df()
    con.close()
    return result

@st.cache_data
def analyze(df):
    X = df[FEATURES]
    y = df["cpc_consumo"]
    X_ols = sm.add_constant(X)
    ols = sm.OLS(y, X_ols).fit()
    sklearn_model = LinearRegression().fit(X, y)
    pred = sklearn_model.predict(X)
    residuals = y - ols.predict(X_ols)
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    dw = sm.stats.stattools.durbin_watson(residuals)
    return ols, sklearn_model, pred, residuals, shapiro_stat, shapiro_p, dw

st.markdown("""
<div class="hero">
<h1>💧 Análisis y predicción del consumo de agua</h1>
<p>Práctica 4.2 · Regresión lineal y correlación · Análisis Estadístico Inferencial</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Configuración del análisis")
    seed = st.number_input("Semilla aleatoria", min_value=0, max_value=999999, value=42, step=1)
    n = st.slider("Número de registros", min_value=50, max_value=500, value=100, step=10)
    st.caption("Los datos son sintéticos y se regeneran de forma reproducible con la misma semilla.")
    st.divider()
    st.markdown("**Variables explicativas**")
    st.write("Temperatura · Humedad · Personas por familia · Consumo de bidón")

df = load_data(int(seed), int(n))
ols, model, pred, residuals, shapiro_stat, shapiro_p, dw = analyze(df)
corr = df.drop(columns=["residencia_id"]).corr(method="pearson")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Resumen", "Datos DuckDB", "Correlaciones", "Regresión OLS", "Diagnóstico y predicción"
])

with tab1:
    st.subheader("Resumen ejecutivo")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(df):,}")
    c2.metric("R² del modelo", f"{ols.rsquared:.3f}")
    c3.metric("R² ajustado", f"{ols.rsquared_adj:.3f}")
    c4.metric("RMSE", f"{np.sqrt(mean_squared_error(df.cpc_consumo, pred)):.2f} L/hab-día")
    st.markdown("#### ¿Qué hace este proyecto?")
    st.write("Construye una tabla de telemetría de agua, explora las relaciones lineales entre variables, "
             "estima una regresión lineal múltiple y evalúa el comportamiento de sus residuos.")
    st.info("Interpretación: R² indica la proporción de variabilidad de CPC explicada por las variables "
            "incluidas. No demuestra causalidad.")
    st.markdown("#### Vista previa de los datos")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Tabla de telemetría almacenada en DuckDB")
    st.write("La aplicación crea una base DuckDB en memoria, registra el DataFrame y consulta la tabla SQL.")
    st.code("CREATE TABLE telemetria_agua AS SELECT * FROM df_source;", language="sql")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Descargar datos CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="telemetria_agua.csv", mime="text/csv")

with tab3:
    st.subheader("Matriz de correlación de Pearson")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Correlación entre variables")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown("#### Correlación de cada variable con CPC")
    st.dataframe(corr[["cpc_consumo"]].sort_values("cpc_consumo", ascending=False),
                 use_container_width=True)

with tab4:
    st.subheader("Regresión lineal múltiple — Mínimos Cuadrados Ordinarios")
    st.write("Variable dependiente: **cpc_consumo**. Variables independientes: temperatura, humedad, "
             "personas por familia y consumo de bidón.")
    st.text(ols.summary().as_text())
    coef = pd.DataFrame({
        "Variable": ols.params.index,
        "Coeficiente": ols.params.values,
        "p-valor": ols.pvalues.values,
        "IC 2.5%": ols.conf_int()[0].values,
        "IC 97.5%": ols.conf_int()[1].values,
    })
    st.markdown("#### Coeficientes e intervalos de confianza")
    st.dataframe(coef, use_container_width=True, hide_index=True)
    st.caption("Un coeficiente representa el cambio esperado en CPC por cada unidad adicional "
               "de la variable, manteniendo constantes las demás variables.")

with tab5:
    st.subheader("Diagnóstico de residuos")
    a, b = st.columns(2)
    a.metric("Shapiro-Wilk · p-valor", f"{shapiro_p:.4f}")
    a.write("No se rechaza normalidad al 5%." if shapiro_p > .05 else "Se rechaza normalidad al 5%.")
    b.metric("Durbin-Watson", f"{dw:.3f}")
    b.write("Valor cercano a 2: sin evidencia fuerte de autocorrelación de primer orden.")
    f1, ax1 = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x=pred, y=residuals, ax=ax1)
    ax1.axhline(0, color="red", linestyle="--")
    ax1.set(xlabel="Valores predichos", ylabel="Residuos", title="Residuos vs. valores ajustados")
    st.pyplot(f1, use_container_width=True)
    plt.close(f1)
    f2, ax2 = plt.subplots(figsize=(7, 4))
    sm.qqplot(residuals, line="45", fit=True, ax=ax2)
    ax2.set_title("Gráfico Q-Q normal")
    st.pyplot(f2, use_container_width=True)
    plt.close(f2)
    f3, ax3 = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x=df["cpc_consumo"], y=pred, ax=ax3)
    lo = min(df["cpc_consumo"].min(), pred.min())
    hi = max(df["cpc_consumo"].max(), pred.max())
    ax3.plot([lo, hi], [lo, hi], "r--")
    ax3.set(xlabel="CPC real", ylabel="CPC predicho", title="Valores reales vs. predichos")
    st.pyplot(f3, use_container_width=True)
    plt.close(f3)

    st.divider()
    st.markdown("#### Calculadora de predicción")
    p1, p2 = st.columns(2)
    with p1:
        temp_in = st.number_input("Temperatura máxima (°C)", 18.0, 32.0, 25.0)
        hum_in = st.number_input("Humedad máxima (%)", 50.0, 90.0, 65.0)
    with p2:
        fam_in = st.number_input("Personas por familia", 3.0, 7.0, 4.0)
        bidon_in = st.number_input("Consumo de bidón (L/hab-día)", 0.2, 2.0, 1.2)
    if st.button("Calcular CPC estimado", type="primary"):
        row = pd.DataFrame([[temp_in, hum_in, fam_in, bidon_in]], columns=FEATURES)
        estimate = float(model.predict(row)[0])
        st.success(f"Consumo per cápita estimado: **{estimate:.2f} L/hab-día**")
        st.caption("Estimación del modelo entrenado con los datos sintéticos de esta sesión.")

st.divider()
st.caption("Proyecto académico demostrativo · Los datos son simulados; no representan mediciones reales.")
