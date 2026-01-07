# =========================
# Gráfico de Backtest
# =========================
st.subheader("📉 Backtest – Valor Real vs Previsão do Modelo")

# Número de dias para exibir
n_dias = st.slider(
    "Quantidade de períodos para visualização:",
    min_value=10,
    max_value=100,
    value=30
)

# Separar dados
dados_bt = dados.copy()

# Garantir alinhamento de features
X_bt = dados_bt.drop(columns=["target"], errors="ignore")
X_bt = X_bt.reindex(columns=colunas_modelo)

# Previsões
dados_bt["Previsao_Modelo"] = modelo.predict(X_bt)

# Selecionar últimos N dias
dados_bt = dados_bt.tail(n_dias)

# Criar gráfico interativo
fig_bt = px.line(
    dados_bt,
    x=dados_bt.index,
    y=["target", "Previsao_Modelo"],
    labels={
        "value": "Classe",
        "index": "Tempo"
    },
    title="Comparação entre Valor Real e Previsão do Modelo"
)

fig_bt.update_traces(mode="lines+markers")

st.plotly_chart(fig_bt, use_container_width=True)

st.caption(
    "✔️ Este gráfico apresenta um backtest do modelo, comparando as "
    "previsões com os valores reais ao longo do tempo."
)
