📊 Sistema Preditivo de Tendência do IBOVESPA

Produto Analítico para Apoio à Tomada de Decisão

1. Resumo Executivo

Este projeto entrega um sistema preditivo interativo que utiliza Machine Learning para estimar a tendência futura do IBOVESPA (Alta ou Queda) com base em dados históricos.
O objetivo do produto é apoiar análises estratégicas e decisões de mercado, fornecendo probabilidades, simulações de cenários e validação histórica, indo além da simples apresentação de métricas técnicas.
A solução foi desenvolvida no contexto do Tech Challenge – Fase 4 (Pós-Tech FIAP), com foco em transformação de modelos analíticos em produtos utilizáveis.

2. Objetivo do Produto

2.1 Antecipar a direção provável do IBOVESPA
2.2 Permitir simulação de cenários a partir de variáveis de mercado
2.3 Reduzir subjetividade na análise, apoiando decisões com dados
2.4 Oferecer transparência, por meio de backtests e métricas

Este sistema não substitui análise humana, mas funciona como uma camada quantitativa de apoio à decisão.

3.Estratégia Analítica

Modelo Utilizado:

Algoritmo: CatBoostClassifier
Tipo: Classificação Binária (Alta / Queda)
Validação: Temporal (TimeSeriesSplit)
Métrica-chave: F1-score

Justificativa Técnica
O CatBoost foi selecionado por sua:

-Robustez em dados financeiros
-Capacidade de lidar com relações não lineares
-Menor sensibilidade a overfitting
-Boa performance com features correlacionadas

4. Indicadores de Performance
   
Indicador	Resultado
Acurácia (Treino)	82,03%
Acurácia (Teste)	80,00%
F1-score Médio (CV)	0.531
Overfitting	2,03%

Os resultados indicam boa capacidade de generalização, com diferença controlada entre treino e teste.

5. Backtest e Validação

O modelo foi avaliado por meio de backtesting, comparando:
>Tendência real observada
>Tendência prevista pelo modelo
>Consistência ao longo do tempo

O backtest está integrado à aplicação, permitindo análise visual e transparente do desempenho histórico.

6. Plataforma Analítica (Streamlit)

A aplicação foi estruturada como um produto de dados, organizado em três frentes:

6.1 Previsão (Core do Produto)
6.2 Simulação de cenários ajustando variáveis de mercado
6.3 Exibição das probabilidades:

📈 Probabilidade de Alta

📉 Probabilidade de Queda

Decisão baseada em limiares de confiança

Comunicação clara do nível de certeza do modelo

7. Backtest

- Visualização interativa da performance histórica
- Comparação direta entre valores reais e previstos
- Controle do horizonte temporal analisado

8. Governança do Modelo

Métricas principais
Estratégia de validação
Contexto de uso e limitações

🧱 Arquitetura do Projeto
tech_challenge_fase4/
│
├── app.py                      # Aplicação Streamlit (Produto)
├── modelo_ibov.pkl              # Modelo de Machine Learning
├── metricas.json                # Métricas consolidadas
├── requirements.txt             # Dependências
│
├── dados/
│   ├── historico_ibov.csv       # Base histórica
│   └── backtest_catboost.csv    # Backtest do modelo
│
└── README.md                    # Documentação executiva

⚙️ Stack Tecnológica

Python

CatBoost

Scikit-learn

Pandas / NumPy

Streamlit

Plotly

▶️ Execução Local
pip install -r requirements.txt
streamlit run app.py

🌐 Disponibilização

A solução foi publicada via Streamlit Cloud, permitindo acesso ao produto sem necessidade de instalação local, facilitando demonstração e avaliação.

⚠️ Considerações Importantes

Este sistema não constitui recomendação de investimento

Resultados devem ser interpretados como apoio analítico

O desempenho passado não garante resultados futuros

👩‍💻 Autoria

Projeto desenvolvido por:

Leonardo Chaves Noronha da Silva
Glaucia Cristina Slompo
Ariceny da Silva Huguenin
Flavia Helena de Almeida
Marcelo Soares de Albuquerque

Pós-Tech – Data Analytics
FIAP

🏁 Conclusão

Este projeto demonstra a evolução de um modelo analítico para um produto funcional, integrando:

Machine Learning
Validação temporal
Visualização interativa
Comunicação orientada a decisão
