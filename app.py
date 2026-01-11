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

# Backtest gerado no notebook
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
**tendência de ALTA ou QUEDA do IBOVESPA** com base em dados históricos.
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
    Simule um cenário de mercado preenchendo os valores abaixo  
    e clique em **Prever Tendência**.
    """)

    # Features exatamente como no treino
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

    # DataFrame na ordem correta
    entrada_df = pd.DataFrame([entrada])[feature_names]

    if st.button("📈 Prever Tendência"):
        try:
            # Probabilidades
            proba = modelo.predict_proba(entrada_df)[0]
            prob_queda = proba[0]
            prob_alta = proba[1]

            # Limiares calibrados
            LIMIAR_ALTA = 0.65
            LIMIAR_QUEDA = 0.65

            st.markdown("### 📊 Probabilidades Estimadas")

            colA, colB = st.columns(2)

            colA.metric(
                "📈 Probabilidade de Alta",
                f"{prob_alta*100:.1f}%"
            )

            colB.metric(
                "📉 Probabilidade de Baixa",
                f"{prob_queda*100:.1f}%"
            )

            st.progress(int(prob_alta * 100))
            st.caption("Barra representa a probabilidade de tendência de ALTA")

            st.markdown("### 🧠 Decisão do Modelo")

            if prob_alta >= LIMIAR_ALTA:
                st.success(
                    f"📈 **TENDÊNCIA DE ALTA DO IBOVESPA**  \n"
                    f"Confiança elevada na direção positiva."
                )

            elif prob_queda >= LIMIAR_QUEDA:
                st.error(
                    f"📉 **TENDÊNCIA DE QUEDA DO IBOVESPA**  \n"
                    f"Confiança elevada na direção negativa."
                )

            else:
                st.warning(
                    "⚖️ **TENDÊNCIA NEUTRA / INDEFINIDA**  \n"
                    "O modelo não identificou uma direção dominante com confiança suficiente."
                )

        except Exception as e:
            st.error("Erro ao realizar a previsão.")
            st.exception(e)


# =====================================================
# with aba2:
    st.subheader("📉 Backtest – Observado vs Previsto")

    col1, col2 = st.columns(2)

    data_min = backtest["Data"].min()
    data_max = backtest["Data"].max()

    with col1:
        inicio = st.date_input(
            "Data inicial",
            value=data_min,
            min_value=data_min,
            max_value=data_max,
            key="bt_inicio"
        )

    with col2:
        fim = st.date_input(
            "Data final",
            value=data_max,
            min_value=data_min,
            max_value=data_max,
            key="bt_fim"
        )

    dados_bt = backtest[
        (backtest["Data"] >= pd.to_datetime(inicio)) &
        (backtest["Data"] <= pd.to_datetime(fim))
    ].copy()

    # Converter para texto (melhor legenda)
    dados_bt["Tipo"] = "Observado"
    dados_prev = dados_bt.copy()
    dados_prev["Tipo"] = "Previsto"
    dados_prev["Valor"] = dados_prev["Previsão"]

    dados_obs = dados_bt.copy()
    dados_obs["Valor"] = dados_obs["Valor Real"]

    dados_plot = pd.concat([
        dados_obs[["Data", "Valor", "Tipo"]],
        dados_prev[["Data", "Valor", "Tipo"]]
    ])

    fig = px.scatter(
        dados_plot,
        x="Data",
        y="Valor",
        color="Tipo",
        title="Backtest – Tendência Observada vs Prevista",
        color_discrete_map={
            "Observado": "#1f77b4",  # azul
            "Previsto": "#ff7f0e"    # laranja
        },
        opacity=0.7
    )

    fig.update_yaxes(
        tickvals=[0, 1],
        ticktext=["Queda", "Alta"],
        title="Tendência"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(dados_bt, use_container_width=True)

# =====================================================
# ABA 3 — SOBRE O MODELO
# =====================================================
with aba3:
    st.subheader("ℹ️ Informações do Modelo")

    st.markdown("""
    **Modelo:** CatBoostClassifier  
    **Tipo:** Classificação Binária (Alta / Queda)  
    **Validação:** Temporal (TimeSeriesSplit)
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Acurácia Treino", f"{metricas['acuracia_treino']*100:.2f}%")
    col2.metric("Acurácia Teste", f"{metricas['acuracia_teste']*100:.2f}%")
    col3.metric("F1-score (CV)", f"{metricas['f1_cv_medio']:.3f}")
    col4.metric("Overfitting", f"{metricas['overfitting_percentual']:.2f}%")

    st.markdown("""
    ### 🎯 Objetivo do Sistema
    Apoiar a análise de mercado por meio da **previsão da tendência do IBOVESPA**,
    utilizando aprendizado de máquina aplicado a séries temporais financeiras.
    """)


