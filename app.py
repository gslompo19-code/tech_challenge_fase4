import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os

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
dados = pd.read_csv("dados/historico_ibov.csv")

with open("metricas.json") as f:
    metricas = json.load(f)

# =========================
# Título
# =========================
st.title("📊 Previsão IBOVESPA – Modelo CatBoost")

# =========================
# Métricas do modelo
# =========================
st.subheader("📈 Métricas do Modelo")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Acurácia Treino", f"{metricas['acuracia_treino']*100:.2f}%")
col2.metric("Acurácia Teste", f"{metricas['acuracia_teste']*100:.2f}%")
col3.metric("F1-score (CV)", metricas["f1_cv_medio"])
col4.metric("Overfitting (%)", metricas["overfitting_percentual"])

# =========================
# Matriz de confusão
# =========================
st.subheader("📊 Matriz de Confusão")

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(metricas["confusao"], cmap="Blues")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Negativo", "Positivo"])
ax.set_yticklabels(["Negativo", "Positivo"])

ax.set_xlabel("Predito")
ax.set_ylabel("Real")

for i in range(2):
    for j in range(2):
        ax.text(
            j, i,
            metricas["confusao"][i][j],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="black"
        )

st.pyplot(fig)

# =========================
# Nova previsão (CORRIGIDO)
# =========================
st.subheader("🔮 Fazer nova previsão")

# Remove target se existir
features = dados.drop(columns=["target"], errors="ignore")

# Seleciona apenas colunas numéricas
features_numericas = features.select_dtypes(include=["int64", "float64"])

entrada = {}

for col in features_numericas.columns:
    entrada[col] = st.number_input(
        label=col,
        min_value=float(features_numericas[col].min()),
        max_value=float(features_numericas[col].max()),
        value=float(features_numericas[col].mean())
    )

entrada_df = pd.DataFrame([entrada])

if st.button("Prever"):
    resultado = modelo.predict(entrada_df)[0]
    st.success(f"Resultado da previsão: {resultado}")

    # Log de uso
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
