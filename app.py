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
    layout="wide"
)

# =====================================================
# CARREGAMENTO DE ARQUIVOS
# =====================================================
modelo = joblib.load("modelo_ibov.pkl")
dados = pd.read_csv("dados/historico_ibov.csv")
metricas = json.load(open("metricas.json"))

# Backtest salvo no notebook
backtest = pd.read_csv("dados/backtest_catboost.csv", parse_dates=["Data"])

# =====================================================
# TÍTULO
# =====================================================
st.title("📊 Sistema Preditivo de Tendência do IBOVESPA")

st.markdown("""
Este sistema utiliza **Machine Learning (CatBoost)** para prever a **tendência do IBOVESPA**
com base em dados históricos.
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
# ABA 1 — PREVISÃO (PRODUTO)
# =====================================================
with aba1:
    st.subheader("🔮 Previsão de Tendência do IBOVESPA")

    st.markdown("""
    Este módulo permite **simular um cenário de mercado** e obter a previsão
    da **tendência do IBOVESPA** para o próximo período, com base no modelo treinado.
    """)

    # Garantir exatamente as features usadas no treino
    feature_names = modelo.feature_names_

    entrada = {}
    cols = st.columns(3)

    for i, col in enumerate(feature_names):
        with cols[i % 3]:
            if col in dados.columns:
                valor_padrao = float(dados[col].mean())
                valor_min = float(dados[col].quantile(0.05))
                valor_max = float(dados[col].quantile(0.95))
            else:
                valor_padrao = 0.0
                valor_min = -1.0
                valor_max = 1.0

            entrada[col] = st.number_input(
                label=col,
                min_value=valor_min,
                max_value=valor_max,
                value=valor_padrao,
                format="%.4f"
            )

    # DataFrame FINAL — ordem correta
    entrada_df = pd.DataFrame([entrada])[feature_names]

    if st.button("📈 Prever Tendência"):
        try:
            # Probabilidades
            proba = modelo.predict_proba(entrada_df)[0]
            prob_queda = proba[0]
            prob_alta = proba[1]

            st.markdown("### 📊 Resultado da Previsão")

            # Barra visual
            st.progress(int(prob_alta * 100))
            st.caption("Probabilidade estimada de tendência de alta")

            # Decisão com zona neutra
            if prob_alta >= 0.55:
                st.success(
                    f"📈 **TENDÊNCIA DE ALTA DO IBOVESPA**  \n"
                    f"Probabilidade: **{prob_alta*100:.1f}%**"
                )

            elif prob_queda >= 0.55:
                st.error(
                    f"📉 **TENDÊNCIA DE QUEDA DO IBOVESPA**  \n"
                    f"Probabilidade: **{prob_queda*100:.1f}%**"
                )

            else:
                st.warning(
                    "⚠️ **TENDÊNCIA NEUTRA / INDEFINIDA**  \n"
                    "O modelo não identificou uma direção dominante."
                )

        except Exception as e:
            st.error("Erro ao realizar a previsão.")
            st.exception(e)

# =====================================================
# ABA 2 — BACKTEST
# =====================================================
with aba2:
    st.subheader("📉 Backtest – Valor Real vs Previsão")

    qtd = st.slider(
        "Quantidade de períodos para visualização:",
        min_value=10,
        max_value=100,
        value=30
    )

    dados_bt = backtest.tail(qtd)

    fig = px.line(
        dados_bt,
        x="Data",
        y=["Valor Real", "Previsão"],
        markers=True,
        title="Comparação entre Valor Real e Previsão do Modelo"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(dados_bt, use_container_width=True)

# =====================================================
# ABA 3 — SOBRE O MODELO
# =====================================================
with aba3:
    st.subheader("ℹ️ Informações do Modelo")

    st.markdown("""
    **Modelo utilizado:** CatBoostClassifier  
    **Tipo:** Classificação binária (Alta / Queda)  
    **Validação:** Temporal (TimeSeriesSplit)  
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Acurácia Treino", f"{metricas['acuracia_treino']*100:.2f}%")
    col2.metric("Acurácia Teste", f"{metricas['acuracia_teste']*100:.2f}%")
    col3.metric("F1-score (CV)", f"{metricas['f1_cv_medio']:.3f}")
    col4.metric("Overfitting", f"{metricas['overfitting_percentual']:.2f}%")

    st.markdown("""
    ### 🎯 Objetivo do Modelo
    Antecipar a **tendência do IBOVESPA**, auxiliando na análise de mercado e tomada
    de decisão baseada em dados.
    """)


