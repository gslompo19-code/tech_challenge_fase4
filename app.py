import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os
import plotly.express as px
import numpy as np

# =========================
# Configuração da página
# =========================
st.set_page_config(
    page_title="Tech Challenge Fase 4 - IBOVESPA",
    layout="wide"
)

# =========================
# Carregar modelo e dados
# =========================
modelo = joblib.load("modelo_ibov.pkl")
colunas_modelo = modelo.feature_names_

dados = pd.read_csv("dados/historico_ibov.csv")

with open("metricas.json") as f:
    metricas = json.load(f)

# =========================
# Título
# =========================
st.title("📊 Previsão IBOVESPA – Modelo CatBoost")

# =========================
# Métricas
# =========================
st.subheader("📈 Métricas do Modelo")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Acurácia Treino", f"{metricas['acuracia_treino']*100:.2f}%")
c2.metric("Acurácia Teste", f"{metricas['acuracia_teste']*100:.2f}%")
c3.metric("F1-score (CV)", metricas["f1_cv_medio"])
c4.metric("Overfitting (%)", metricas["overfitting_percentual"])

# =========================
# Matriz de confusão
# =========================
st.subheader("📊 Matriz de Confusão")

fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(metricas["confusao"], cmap="Blues")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Negativo", "Positivo"])
ax.set_yticklabels(["Negativo", "Positivo"])

for i in range(2):
    for j in range(2):
        ax.text(
            j, i,
            metricas["confusao"][i][j],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold"
        )

st.pyplot(fig, use_container_width=False)

# =========================
# Preparação das features
# =========================
features = dados.drop(columns=["target"], errors="ignore")
features_numericas = features.select_dtypes(include=["int64", "float64"])

# =========================
# Previsão manual
# =========================
st.subheader("🔮 Fazer nova previsão")

entrada = {}

for col in features_numericas.columns:
    entrada[col] = st.slider(
        col,
        float(features_numericas[col].min()),
        float(features_numericas[col].max()),
        float(features_numericas[col].mean())
    )

entrada_df = pd.DataFrame([entrada])
entrada_df = entrada_df.reindex(columns=colunas_modelo)

if st.button("Prever"):
    prob = modelo.predict_proba(entrada_df)[0][1]
    classe = int(prob >= 0.5)

    st.success(
        f"📈 Probabilidade de Alta: {prob:.2%} | Classe prevista: {classe}"
    )

    log = entrada_df.copy()
    log["probabilidade_alta"] = prob
    log["classe_prevista"] = classe
    log["data_hora"] = datetime.now()

    os.makedirs("dados", exist_ok=True)
    log.to_csv(
        "dados/log_uso.csv",
        mode="a",
        header=not os.path.exists("dados/log_uso.csv"),
        index=False
    )

# =========================
# Gráfico de sensibilidade
# =========================
st.subheader("📈 Gráfico Interativo – Sensibilidade do Modelo")

variavel = st.selectbox(
    "Escolha uma variável para variar isoladamente:",
    features_numericas.columns
)

valores = np.linspace(
    features_numericas[variavel].min(),
    features_numericas[variavel].max(),
    20
)

probas = []

for v in valores:
    sim = entrada_df.copy()
    sim[variavel] = v
    sim = sim.reindex(columns=colunas_modelo)
    p = modelo.predict_proba(sim)[0][1]
    probas.append(p)

df_sim = pd.DataFrame({
    variavel: valores,
    "Probabilidade de Alta": probas
})

fig_sim = px.line(
    df_sim,
    x=variavel,
    y="Probabilidade de Alta",
    markers=True,
    title="Resposta do Modelo à Variação de uma Feature"
)

st.plotly_chart(fig_sim, use_container_width=True)

st.info(
    "ℹ️ A linha pode aparecer constante porque modelos baseados em árvores "
    "tomam decisões por regras. Variar uma única feature pode não alterar "
    "a decisão do modelo."
)

# =========================
# Backtest – Valor Real vs Previsão (CSV do notebook)
# =========================
st.subheader("📉 Backtest – Valor Real vs Previsão do Modelo")

# Carregar backtest salvo no notebook
backtest = pd.read_csv("dados/backtest_catboost.csv")

# Converter data
backtest["Data"] = pd.to_datetime(backtest["Data"])

# Slider
n_dias = st.slider(
    "Quantidade de períodos para visualização:",
    min_value=10,
    max_value=len(backtest),
    value=min(30, len(backtest))
)

backtest_plot = backtest.tail(n_dias)

# Criar gráfico manualmente (mais robusto)
fig_bt = px.line(
    backtest_plot,
    x="Data",
    y=["Valor Real", "Previsão"],
    markers=True,
    title="Comparação entre Valor Real e Previsão do Modelo"
)

st.plotly_chart(fig_bt, use_container_width=True)

# Mostrar tabela
st.subheader("📋 Resultados do Backtest")
st.dataframe(backtest_plot)
