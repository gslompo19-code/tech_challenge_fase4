import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px

# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================
st.set_page_config(
    page_title="Sistema Preditivo IBOVESPA",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# ESTILO (VISUAL MAIS PROFISSIONAL)
# =====================================================
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        h1, h2, h3 { color: #0b3c5d; }
        .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

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

st.markdown(
    """
    Este produto utiliza **Machine Learning (CatBoost)** para prever a  
    **tendência futura do IBOVESPA (Alta ou Queda)** com base em dados históricos.
    """
)

# =====================================================
# ABAS
# =====================================================
aba1, aba2, aba3 = st.tabs([
    "🔮 Previsão",
    "📉 Backtest",
    "ℹ️ Sobre o Modelo"
])

# =====================================================
# ABA 1 — PREVISÃO (PRODUTO)
# =====================================================
with aba1:
    st.subheader("🔮 Previsão de Tendência do IBOVESPA")

    st.markdown(
        "Ajuste os indicadores abaixo e clique em **Prever** para obter a tendência esperada."
    )

    features = dados.drop(columns=["target"], errors="ignore")

    entrada = {}
    cols = st.columns(3)

    for i, col in enumerate(features.columns):
        with cols[i % 3]:
            entrada[col] = st.number_input(
                label=col,
                value=float(dados[col].mean()),
                format="%.4f",
                key=f"input_{col}"
            )

    entrada_df = pd.DataFrame([entrada])

    if st.button("📈 Prever Tendência", key="btn_prever"):
        probs = modelo.predict_proba(entrada_df)[0]
        prob_baixa = probs[0]
        prob_alta = probs[1]

        st.markdown("### 📌 Resultado da Previsão")

        col1, col2 = st.columns(2)

        col1.metric(
            "📉 Probabilidade de Queda",
            f"{prob_baixa * 100:.2f}%"
        )

        col2.metric(
            "📈 Probabilidade de Alta",
            f"{prob_alta * 100:.2f}%"
        )

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
        "Quantidade de períodos para visualização:",
        min_value=10,
        max_value=100,
        value=30,
        key="slider_backtest"
    )

    dados_bt = backtest.tail(qtd).copy()

    fig = px.line(
        dados_bt,
        x="Data",
        y=["Valor Real", "Previsão"],
        markers=True,
        title="Comparação entre Valor Real e Previsão do Modelo",
        color_discrete_map={
            "Valor Real": "#0b3c5d",
            "Previsão": "#1abc9c"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        dados_bt,
        use_container_width=True,
        hide_index=True
    )

# =====================================================
# ABA 3 — SOBRE O MODELO
# =====================================================
with aba3:
    st.subheader("ℹ️ Informações do Modelo")

    st.markdown(
        """
        **Modelo:** CatBoostClassifier  
        **Problema:** Classificação Binária (Alta / Queda)  
        **Validação:** TimeSeriesSplit  
        **Objetivo:** Antecipar a tendência do IBOVESPA
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Acurácia Treino",
        f"{metricas['acuracia_treino']*100:.2f}%"
    )

    col2.metric(
        "Acurácia Teste",
        f"{metricas['acuracia_teste']*100:.2f}%"
    )

    col3.metric(
        "F1-score (CV)",
        f"{metricas['f1_cv_medio']:.3f}"
    )

    col4.metric(
        "Overfitting",
        f"{metricas['overfitting_percentual']:.2f}%"
    )

    st.markdown(
        """
        ### 🎯 Visão de Produto
        Este sistema foi desenvolvido como **ferramenta de apoio à decisão**,
        permitindo testar cenários e compreender o comportamento esperado
        do índice com base em dados históricos.
        """
    )
