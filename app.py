import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os

st.set_page_config(page_title="Tech Challenge Fase 4 - IBOVESPA", layout="wide")

# Carregar modelo e dados
modelo = joblib.load("modelo_ibov.pkl")
dados = pd.read_csv("dados/historico_ibov.csv")

with open("metricas.json") as f:
    metricas = json.load(f)

st.title("📊 Previsão IBOVESPA – Modelo CatBoost")

# Métricas
st.subheader("📈 Métricas do Modelo")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Acurácia Treino", f"{metricas['acuracia_treino']*100:.2f}%")
col2.metric("Acurácia Teste", f"{metricas['acuracia_teste']*100:.2f}%")
col3.metric("F1-score (CV)", metricas["f1_cv_medio"])
col4.metric("Overfitting (%)", metricas["overfitting_percentual"])

# Matriz de confusão (CORRIGIDA)
st.subheader("📊 Matriz de Confusão")
fig, ax = plt.subplots()
ax.imshow(metricas["confusao"])
ax.set_xlabel("Predito")
ax.set_ylabel("Real")

for i in range(2):
    for j in range(2):
        ax.text(j, i, metricas["confusao"][i][j],
                ha="center", va="center", color="black")

st.pyplot(fig)

# Previsão
st.subheader("🔮 Fazer nova previsão")

features = dados.drop(columns=["target"], errors="ignore")

entrada = {}
for col in features.columns:
    entrada[col] = st.number_input(
        col,
        float(dados[col].min()),
        float(dados[col].max())
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
