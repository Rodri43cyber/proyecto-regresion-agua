# Análisis y predicción del consumo per cápita de agua

Aplicación académica para la Práctica 4.2: Regresión lineal y correlación.

## Funcionalidades
- Generación reproducible de datos sintéticos.
- Tabla SQL en DuckDB y descarga CSV.
- Correlación de Pearson y mapa de calor.
- Regresión múltiple OLS con `statsmodels`.
- Métricas con `scikit-learn`.
- Diagnóstico de residuos: Shapiro-Wilk, Durbin-Watson, Q-Q y residuos vs. ajustados.
- Calculadora de predicción CPC.

## Ejecutar localmente

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Abre la URL local que Streamlit muestre en la terminal.

## Publicar en Render

1. Sube esta carpeta a un repositorio de GitHub.
2. En Render, selecciona **New > Blueprint** y conecta el repositorio para usar `render.yaml`.
   También puedes crear un **Web Service** manualmente.
3. Configura:
   - Runtime: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`
4. Inicia el despliegue y revisa los logs hasta que el servicio esté Live.

## Nota metodológica importante

Los registros que genera esta versión son **sintéticos** y reproducibles con la semilla elegida. La fórmula de generación se ha documentado en `app.py`. No se deben presentar como observaciones reales ni afirmar que reproduce exactamente los valores de la captura de la práctica. Para reproducir los resultados exactos del documento, es necesario validar la fórmula original, el orden de las operaciones aleatorias y la semilla de DuckDB frente al código fuente completo de la actividad.

## Estructura

```text
proyecto_regresion_agua/
├── app.py
├── requirements.txt
├── render.yaml
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```
