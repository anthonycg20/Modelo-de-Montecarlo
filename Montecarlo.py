import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# ---------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Montecarlo Financial Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Dashboard de Simulación de Montecarlo")
st.markdown(
    "Proyección estocástica de activos financieros usando el **Movimiento Browniano Geométrico (GBM)** con datos de mercado en tiempo real."
)

# ---------------------------------------------------------
# PANEL LATERAL: PARÁMETROS DE ENTRADA
# ---------------------------------------------------------
st.sidebar.header("⚙️ Parámetros del Modelo")

ticker = st.sidebar.text_input("Ticker del Activo", value="AAPL").upper()
years_hist = st.sidebar.slider("Años de datos históricos", min_value=1, max_value=5, value=2)
simulations = st.sidebar.number_input("Número de simulaciones", min_value=100, max_value=20000, value=5000, step=500)
days_to_sim = st.sidebar.number_input("Días hábiles a proyectar", min_value=10, max_value=756, value=252, step=10)
confidence_level = st.sidebar.slider("Nivel de Confianza (VaR %)", min_value=80, max_value=99, value=95)

# ---------------------------------------------------------
# DESCARGA Y CÁLCULO DE PARÁMETROS EMPÍRICOS
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def obtener_datos(symbol, period_years):
    data = yf.download(symbol, period=f"{period_years}y", progress=False)
    if data.empty:
        return None
    return data['Close']

with st.spinner(f"Obteniendo precios de mercado para {ticker}..."):
    precios_serie = obtener_datos(ticker, years_hist)

if precios_serie is None or len(precios_serie) < 30:
    st.error(f"No se pudieron descargar datos para '{ticker}'. Verifica el ticker e intentalo de nuevo.")
    st.stop()

# Limpieza de formato en caso de MultiIndex de pandas
if isinstance(precios_serie, pd.DataFrame):
    precios_serie = precios_serie.iloc[:, 0]

# Rendimientos logarítmicos y parámetros anualizados
log_returns = np.log(precios_serie / precios_serie.shift(1)).dropna()
sigma = float(log_returns.std() * np.sqrt(252))
mu = float((log_returns.mean() * 252) + (0.5 * sigma**2))
S0 = float(precios_serie.iloc[-1])

# ---------------------------------------------------------
# EJECUCIÓN DE LA SIMULACIÓN DE MONTECARLO
# ---------------------------------------------------------
T = days_to_sim / 252.0
dt = T / days_to_sim

# Matriz estocástica
Z = np.random.normal(0, 1, (days_to_sim, simulations))
S = np.zeros((days_to_sim + 1, simulations))
S[0] = S0

for t in range(1, days_to_sim + 1):
    S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1])

precios_finales = S[-1]
precio_medio = float(np.mean(precios_finales))
p_lower = float(np.percentile(precios_finales, 100 - confidence_level))
p_median = float(np.median(precios_finales))
p_upper = float(np.percentile(precios_finales, confidence_level))

var_monto = S0 - p_lower
var_pct = (var_monto / S0) * 100

# ---------------------------------------------------------
# MÉTRICAS PRINCIPALES
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precio Actual (S₀)", f"${S0:.2f}")
col2.metric("Precio Promedio Esperado", f"${precio_medio:.2f}", f"{((precio_medio/S0)-1)*100:+.2f}%")
col3.metric(f"Value at Risk (VaR {confidence_level}%)", f"${var_monto:.2f}", f"-{var_pct:.2f}%", delta_color="inverse")
col4.metric("Volatilidad Anualized (σ)", f"{sigma:.2%}")

st.markdown("---")

# ---------------------------------------------------------
# GRÁFICOS INTERACTIVOS (PLOTLY)
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📉 Trayectorias de Simulación", "📊 Distribución de Resultados"])

with tab1:
    fig_paths = go.Figure()
    
    # Muestra de trayectorias corregida
    sample_paths = min(100, simulations)
    for i in range(sample_paths):
        fig_paths.add_trace(go.Scatter(
            y=S[:, i],
            mode='lines',
            opacity=0.3,              # <-- Colocado fuera de line=dict()
            line=dict(width=0.7),     # <-- Solo propiedades de línea válidas
            showlegend=False
        ))
    
    # Líneas de Percentiles
    fig_paths.add_trace(go.Scatter(y=np.median(S, axis=1), mode='lines', line=dict(color='#00E676', width=3), name='Mediana (P50)'))
    fig_paths.add_trace(go.Scatter(y=np.percentile(S, confidence_level, axis=1), mode='lines', line=dict(color='#29B6F6', width=2, dash='dash'), name=f'Percentil {confidence_level}'))
    fig_paths.add_trace(go.Scatter(y=np.percentile(S, 100-confidence_level, axis=1), mode='lines', line=dict(color='#FF5252', width=2, dash='dash'), name=f'Percentil {100-confidence_level} (VaR)'))

    fig_paths.update_layout(
        title=f"Muestra de {sample_paths} Trayectorias de Montecarlo ({days_to_sim} Días)",
        xaxis_title="Días Bursátiles Proyectados",
        yaxis_title="Precio (USD)",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig_paths, use_container_width=True)

with tab2:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=precios_finales,
        nbinsx=60,
        marker_color='#3F51B5',
        opacity=0.8,
        name="Precios Finales"
    ))
    
    fig_hist.add_vline(x=S0, line_dash="dash", line_color="white", annotation_text=f"S₀: ${S0:.2f}")
    fig_hist.add_vline(x=p_lower, line_dash="dash", line_color="#FF5252", annotation_text=f"VaR {confidence_level}%: ${p_lower:.2f}")
    fig_hist.add_vline(x=precio_medio, line_dash="dash", line_color="#00E676", annotation_text=f"Media: ${precio_medio:.2f}")

    fig_hist.update_layout(
        title=f"Distribución Frecuencial de Precios al Final del Periodo ({days_to_sim} Días)",
        xaxis_title="Precio Final (USD)",
        yaxis_title="Frecuencia",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------
# TABLA RESUMEN DE RESULTADOS
# ---------------------------------------------------------
st.subheader("📋 Cuadro Resumen Estadístico")

resumen_df = pd.DataFrame({
    "Métrica Financiera / Estadística": [
        "Precio Inicial (S₀)",
        "Rendimiento Anual Histórico (μ)",
        "Volatilidad Anual Histórica (σ)",
        "Precio Medio Esperado",
        "Mediana de Precios",
        f"Límite Inferior (Percentil {100-confidence_level}%)",
        f"Límite Superior (Percentil {confidence_level}%)",
        "Rendimiento Promedio Esperado (%)"
    ],
    "Valor Estimado": [
        f"${S0:.2f}",
        f"{mu:.2%}",
        f"{sigma:.2%}",
        f"${precio_medio:.2f}",
        f"${p_median:.2f}",
        f"${p_lower:.2f}",
        f"${p_upper:.2f}",
        f"{((precio_medio/S0) - 1):.2%}"
    ]
})

st.table(resumen_df)