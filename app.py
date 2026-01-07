import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os
import plotly.express as px

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
ax.set_xticklabels(["Neg", "Pos"])
ax.set_yticklabels(["Neg", "Pos"])

for i in range(2):
    for j in range(2):
        ax.text(j, i, metricas["confusao"][i][j],
                ha="center", va="center", fontsize=11, fontweight="bold")

st.pyplot(fig, use_container_width=False)

# =========================
# Preparação das features
# =========================
features = dados.drop(columns=["target"], errors="ignore")
features_numericas = features.select_dtypes(include=["int64", "float64"])

# =========================
# Nova previsão
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
    resultado = modelo.predict(entrada_df)[0]
    st.success(f"Resultado da previsão: {resultado}")

    log = entrada_df.copy()
    log["resultado"] = resultado
    log["data_hora"] = datetime.now()

    os.makedirs("dados", exist_ok=True)
    log.to_csv(
        "dados/log_uso.csv",
        mode="a",
        header=not os.path.exists("dados/log_uso.csv"),
        index=False
    )

# =========================
# Gráfico interativo (SEMPRE APARECE)
# =========================
st.subheader("📈 Gráfico Interativo – Simulação de Tendência")

variavel = st.selectbox(
    "Escolha uma variável para analisar a tendência:",
    features_numericas.columns
)

# Valores simulados
valores = features_numericas[variavel].quantile(
    [0.05, 0.25, 0.5, 0.75, 0.95]
).values

resultados = []

for v in valores:
    sim = entrada_df.copy()
    sim[variavel] = v
    sim = sim.reindex(columns=colunas_modelo)
    pred = modelo.predict(sim)[0]
    resultados.append(pred)

df_sim = pd.DataFrame({
    variavel: valores,
    "Previsão do Modelo": resultados
})

fig_sim = px.line(
    df_sim,
    x=variavel,
    y="Previsão do Modelo",
    markers=True,
    title="Resposta do Modelo à Variação da Variável"
)

st.plotly_chart(fig_sim, use_container_width=True)
