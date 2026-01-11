import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px
import numpy as np

# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================
st.set_page_config(
    page_title="Sistema Preditivo IBOVESPA",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# ESTILO (VISUAL PROFISSIONAL)
# =====================================================
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
h1, h2, h3 { color: #0b3c5d; }
.stMetric {
    background-color: #f0f2f6;
    padding: 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# CARREGAMENTO DE ARQUIVOS
# =====================================================
modelo = joblib.load("modelo_ibov.pkl")
dados = pd.read_csv("dados/historico_ibov.csv")
metricas = json.load(open("metricas.json"))

backtest = pd.read_csv(
    "dados/backtest_catboost.csv",
    parse_dates=["Data"]
)

# =====================================================
# TÍTULO
# =====================================================
st.title("📊 Sistema Preditivo de Tendência do IBOVESPA")

st.markdown("""
Este sistema utiliza **Machine Learning (CatBoost)** para prever a  
**tendência futura do IBOVESPA (Alta ou Queda)** com base em dados históricos.
""")

# =====================================================
# ABAS
# =====================================================
aba1, aba2, aba3 = st.tabs([
    "🔮 Previsão",
    "📉 Backtest",
    "ℹ️ Sobre o Modelo"
])

# =====================================================
# ABA 1 — PREVISÃO (PRODUTO REAL)
# =====================================================
with aba1:
    st.subheader("🔮 Previsão de Tendência do IBOVESPA")

    st.markdown("Ajuste os indicadores e gere a previsão do modelo.")

    # 👉 SOMENTE FEATURES NUMÉRICAS
    features = dados.drop(columns=["target"], errors="ignore")
    features = features.select_dtypes(include=[np.number])

    entrada = {}
    cols = st.columns(3)

    for i, col in enumerate(features.columns):
        with cols[i % 3]:
            entrada[col] = st.number_input(
                label=col,
                value=float(features[col].mean()),
                format="%.4f",
                key=f"input_{col}"
            )

    entrada_df = pd.DataFrame([entrada])

    if st.button("📈 Prever Tendência", key="btn_prever"):
        probs = modelo.predict_proba(entrada_df)[0]

        prob_baixa = probs[0]
        prob_alta = probs[1]

        st.markdown("### 📌 Resultado da Previsão")

        c1, c2 = st.columns(2)
        c1.metric("📉 Probabilidade de Queda", f"{prob_baixa*100:.2f}%")
        c2.metric("📈 Probabilidade de Alta", f"{prob_alta*100:.2f}%")

        if prob_alta >= 0.5:
            st.success("📈 **TENDÊNCIA DE ALTA DO IBOVESPA**")
        else:
            st.error("📉 **TENDÊNCIA DE QUEDA DO IBOVESPA**")

# =====================================================
# ABA 2 — BACKTEST
# =====================================================
with aba2:
    st.subheader("📉 Backtest – Valor Real vs Previsão")

    qtd = st.slider(
        "Quantidade de períodos:",
        min_value=10,
        max_value=100,
        value=30,
        key="slider_backtest"
    )

    dados_bt = backtest.tail(qtd)

    fig = px.line(
        dados_bt,
        x="Data",
        y=["Valor Real", "Previsão"],
        markers=True,
        title="Comparação entre Valor Real e Previsão",
        color_discrete_map={
            "Valor Real": "#0b3c5d",
            "Previsão": "#1abc9c"
        }
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(dados_bt, use_container_width=True, hide_index=True)

# =====================================================
# ABA 3 — SOBRE O MODELO
# =====================================================
with aba3:
    st.subheader("ℹ️ Informações do Modelo")

    st.markdown("""
**Modelo:** CatBoostClassifier  
**Tipo:** Classificação Binária (Alta / Queda)  
**Validação:** TimeSeriesSplit  
""")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Acurácia Treino", f"{metricas['acuracia_treino']*100:.2f}%")
    c2.metric("Acurácia Teste", f"{metricas['acuracia_teste']*100:.2f}%")
    c3.metric("F1-score (CV)", f"{metricas['f1_cv_medio']:.3f}")
    c4.metric("Overfitting", f"{metricas['overfitting_percentual']:.2f}%")

    st.markdown("""
### 🎯 Visão de Produto
Este sistema é um **produto preditivo**, permitindo simular cenários,
avaliar probabilidades e apoiar decisões baseadas em dados.
""")
