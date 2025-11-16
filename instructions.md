Instruções para GitHub Copilot – Projeto de previsão do S&P 500​
Objetivo do projeto
O objetivo deste repositório é construir um sistema em Python capaz de prever o comportamento do S&P 500 a partir de dados históricos, incluindo preços, indicadores macroeconómicos, inflação e métricas como PE10.​
O projeto combina aprendizagem supervisionada, não supervisionada e modelos de séries temporais (incluindo LSTM) e expõe os resultados através de um dashboard web em Flask.​

Contexto e dados
O projeto baseia‑se num dataset histórico do S&P 500 e de outros ETFs, obtido tipicamente de fontes como Kaggle, com aproximadamente 1 800 registos em frequência baixa (mensal ou semelhante).​
Os dados incluem preços nominais e reais, indicadores de valorização (como PE10), medidas de inflação e outras variáveis úteis para caracterizar regimes de mercado e ciclos económicos.​
A análise anterior já explorou tendências e variações de preço, bem como padrões de swings e possíveis regimes de mercado, e o objetivo agora é estruturar essa lógica num pipeline limpo e reprodutível.​

Ambiente de desenvolvimento
O projeto foi inicialmente desenvolvido em Spyder durante as aulas, mas agora deve ser continuado e melhorado em VSCode, preferencialmente com um ambiente virtual dedicado.​
O Copilot deve assumir o uso de Python 3.x com bibliotecas como pandas, numpy, scikit‑learn, matplotlib, seaborn, plotly, keras ou tensorflow, flask e outras necessárias.​
A organização sugerida é separar o código em pastas como src/data, src/models, src/api e notebooks/ para experiências exploratórias.​

Estilo de colaboração com o Copilot
O Copilot deve propor código modular, legível e fácil de testar, privilegiando funções e classes com responsabilidades bem definidas.​
Sempre que sugerir um bloco de código relevante (modelo, pipeline, métrica), o Copilot deve incluir comentários claros e, quando fizer sentido, docstrings explicando inputs, outputs e decisões de design.​
O Copilot deve evitar “chutes”, ou seja, não deve inventar campos, colunas ou ficheiros que não existam e deve basear o código em nomes explícitos definidos no próprio projeto.​
Quando existirem várias abordagens possíveis, o Copilot deve justificar brevemente nas docstrings ou comentários porque escolheu uma determinada técnica ou hiperparâmetros (por exemplo, escolha de janelas temporais ou número de neurónios na LSTM).​

Instruções gerais de workflow
O Copilot deve seguir um workflow lógico em etapas: carregar dados, preparar features, dividir treino/teste, treinar modelos, avaliar resultados, guardar modelos e integrar com Flask.​
Cada etapa deve corresponder a funções ou módulos bem separados, para que seja simples reutilizar o pipeline e executar experiências adicionais.​
Sempre que for criada nova funcionalidade, o Copilot deve tentar escrever o código de forma idempotente (repetir a execução não deve corromper ficheiros nem duplicar dados).​

Passo 1 – Carregar e validar dados
O Copilot deve criar funções em src/data para carregar o dataset bruto a partir de data/raw, garantindo conversão correta da coluna de data para datetime e definição de índice temporal adequado.​
Deve incluir verificações básicas de qualidade, como detetar valores em falta, tipos incoerentes e dimensão do dataset (número de linhas e intervalo temporal).​

Passo 2 – Limpeza e pré‑processamento
O Copilot deve implementar funções para tratar valores em falta (por exemplo, forward fill, interpolação ou remoção controlada) e lidar com outliers relevantes.​
Deve incluir normalização ou padronização das variáveis numéricas e garantir que qualquer transformação é aplicada de forma consistente (fit no treino, transform no teste).​

Passo 3 – Feature engineering e criação do alvo
O Copilot deve criar funções para gerar lags temporais do S&P 500 e de indicadores críticos (lag_1, lag_3, lag_12, etc.) adequados à frequência do dataset.​
Deve também criar a variável alvo para regressão (preço futuro num horizonte definido) e para classificação (direção ±1 com base na variação de preço).​

Passo 4 – Divisão treino/teste apropriada a séries temporais
O Copilot deve implementar uma divisão treino/teste baseada em ordem temporal, sem embaralhar as linhas nem usar dados futuros no treino.​
Idealmente, o Copilot deve estruturar uma função que permita facilmente experimentar diferentes pontos de corte ou esquemas de validação walk‑forward.​

Passo 5 – Modelos de regressão supervisionada
O Copilot deve criar um notebook ou módulo (por exemplo, 02_regression_models.ipynb ou src/models/regression.py) que treine pelo menos Regressão Linear, Ridge e Lasso para prever preços futuros.​
O Copilot deve calcular métricas como MAE, MSE e RMSE e devolver um resumo comparativo para facilitar a escolha do modelo baseline.​

Passo 6 – Modelos de classificação (Random Forest e Gradient Boosting)
O Copilot deve implementar modelos de Random Forest e Gradient Boosting para classificar a direção do mercado (subida ou descida) com base nas features construídas.​
Deve calcular métricas de classificação como Accuracy, Precision, Recall e F1, bem como apresentar uma matriz de confusão.​
O Copilot deve preparar funções para guardar os modelos treinados em ficheiros rf.pkl e gbm.pkl, juntamente com os scalers e parâmetros necessários.​

Passo 7 – Modelos não supervisionados (PCA, K‑Means, DBSCAN)
O Copilot deve criar um notebook ou módulo para aplicar PCA ao dataset processado, mostrar a variância explicada e produzir gráficos 2D coloridos por cluster ou regime.​
Deve implementar K‑Means para identificar regimes de mercado ou clusters de volatilidade, adicionando os labels ao dataframe principal.​
O Copilot deve também implementar DBSCAN para deteção de anomalias e produzir gráficos que destaquem crashes ou picos anómalos de mercado ao longo do tempo.​

Passo 8 – Modelo LSTM para séries temporais
O Copilot deve preparar funções para transformar o dataset em janelas (X, y) adequadas a uma rede LSTM, respeitando a ordem temporal.​
Deve propor uma arquitetura LSTM razoável em Keras ou TensorFlow, com documentação sobre número de camadas, neurónios e funções de ativação.​
O Copilot deve treinar a LSTM, comparar o seu desempenho com os modelos de regressão clássica e guardar o modelo em lstm_model.h5.​

Passo 9 – Avaliação, comparação e visualização
O Copilot deve centralizar as métricas de todos os modelos (regressão, classificação, LSTM) numa função ou notebook que gere tabelas comparativas claras.​
Deve sugerir gráficos de previsões versus valores reais, curvas de erro ao longo do tempo e visualizações de regimes de mercado definidos pelos clusters.​

Passo 10 – Integração com Flask
O Copilot deve ajudar a criar um módulo src/api/app.py com rotas Flask para homepage, forecasting, clustering e endpoints JSON (/predict_rf, /predict_lstm, /clusters, /metrics).​
Deve sugerir integração com Plotly ou outra biblioteca de gráficos interativos para mostrar tendências, previsões e clusters diretamente no browser.​
O Copilot deve garantir que o código da API está organizado, com funções separadas para carregar modelos, preparar dados de input e formatar as respostas.​