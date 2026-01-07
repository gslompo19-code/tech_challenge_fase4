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
st.subheader("📈 Métricas do Modelo
