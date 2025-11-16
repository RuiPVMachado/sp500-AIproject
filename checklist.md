Checklist – Projeto S&P 500 ML & Dashboard Flask​
Visão geral do projeto
Criar projeto em Python para análise e previsão do S&P 500 e ETFs com foco em séries temporais e regimes de mercado.​

Implementar protótipo de website Flask para apresentar resultados da análise e previsões.​

Consolidar projeto como solução completa com modelos supervisionados, não supervisionados, LSTM e dashboard interativo final.​

Etapa 0 – Setup & organização
X Criar repositório GitHub com estrutura básica de pastas (src/, data/, notebooks/, docs/).​

X Configurar ambiente Python e ficheiro de dependências (requirements.txt ou environment.yml).​

X Adicionar .gitignore adequado (ficheiros temporários, virtual env, data bruta, modelos treinados).​

X Configurar GitHub Projects ou equivalente para gerir tarefas em estilo Kanban.​

X Configurar workflow de CI (testes básicos e lint) em GitHub Actions.​

Etapa 1 – Dados & EDA
X Obter dataset histórico do S&P 500 e variáveis associadas (ex.: inflação, indicadores macro, PE10) a partir de fontes como Kaggle.​

X Carregar o dataset em Python, converter a coluna de data para datetime e verificar tipos de dados e valores em falta.​

X Caracterizar o dataset (número de registos, intervalo temporal, descrição básica dos atributos).​

X Produzir análise exploratória inicial com gráficos de linha para SP500, preço real e PE10.​

X Completar EDA com histogramas, boxplots e mapa de correlação entre variáveis.​

X Criar ficheiro docs/eda.md a resumir principais insights e gráficos relevantes.​

Etapa 2 – Pré‑processamento & feature engineering
X Tratar valores em falta e outliers nas principais variáveis de preço e indicadores.​

X Normalizar ou padronizar features numéricas para uso em modelos de ML.​

X Criar variáveis de atraso (lags) para o S&P 500 e indicadores relacionados (lag_1, lag_3, lag_12, etc.).​

X Definir variável alvo para regressão (preço futuro) e para classificação (direção ±1).​

X Implementar pipeline reutilizável em src/data/pipeline.py para carregar, preprocessar, criar lags e alvo.​

X Guardar dataset processado em data/processed/processed_data.csv pronto para modelação.​

Etapa 3 – Modelos supervisionados
X Implementar modelo de Regressão Linear como baseline para previsão de preço futuro do S&P 500.​

X Adicionar modelos de Regressão Ridge e Lasso e comparar métricas de erro (MAE, MSE, RMSE).​

X Implementar modelo de Random Forest para classificação da direção do mercado (subida ou descida).​

X Implementar modelo de Gradient Boosting para classificação da direção do mercado e comparar com Random Forest.​

X Realizar tuning simples de hiperparâmetros (ex.: número de árvores, profundidade) usando GridSearchCV ou RandomizedSearchCV.​

X Guardar modelos supervisionados treinados em ficheiros rf.pkl, gbm.pkl e regressao_baseline.pkl.​

Etapa 4 – Modelos não supervisionados
X Implementar PCA para redução de dimensionalidade e gerar gráfico de variância explicada.​

X Criar gráfico 2D com os dois primeiros componentes principais e pontos coloridos por regime ou cluster.​

X Implementar K-Means para identificar regimes de mercado ou clusters de volatilidade.​

X Ligar labels de clusters ao dataset principal para análise temporal dos regimes.​

X Implementar DBSCAN para deteção de anomalias (crashes, spikes de volatilidade, episódios extremos).​

X Criar gráficos que destaquem anomalias detetadas em função do tempo.​

X Documentar resultados de PCA, K-Means e DBSCAN no ficheiro docs/anomalies.md com breve análise.​

Etapa 5 – LSTM e séries temporais
X Preparar dados em formato de janelas temporais (sequências) para alimentar uma rede LSTM.​

X Definir arquitetura LSTM em Keras ou TensorFlow (camadas LSTM, Dropout, Dense final).​

X Treinar LSTM para prever o preço do S&P 500 num horizonte temporal escolhido (ex.: próximo mês).​

X Comparar desempenho da LSTM com os modelos de regressão clássica usando RMSE ou outra métrica.​

X Guardar modelo LSTM treinado em lstm_model.h5 para utilização posterior na API Flask.​

Etapa 6 – Divisão treino/teste & avaliação
X Criar divisão temporal treino/teste adequada a séries temporais, sem mistura de dados futuros no treino.​

X Implementar validação cruzada temporal ou walk‑forward para avaliar robustez das previsões.​

X Calcular métricas de regressão (MAE, MSE, RMSE) para modelos de previsão de preço.​

X Calcular métricas de classificação (Accuracy, Precision, Recall, F1) para modelos de direção do mercado.​

X Criar tabela comparativa das métricas de todos os modelos supervisionados.​

Etapa 7 – Flask & dashboard
X Criar aplicação Flask mínima com rota para a homepage a resumir o projeto e o objetivo da previsão.​

X Integrar pelo menos um gráfico de tendência do S&P 500 na homepage (ex.: Matplotlib, Plotly ou Seaborn).​

X Criar endpoints de API para previsões com Random Forest, LSTM e outras variantes (/predict_rf, /predict_lstm, /metrics, /clusters).​

X Implementar página de forecasting que mostre previsões dos diferentes modelos e respetivas métricas.​

X Implementar página de clustering com visualização de PCA, clusters K-Means e anomalias de DBSCAN.​

X Implementar página de exploração de dados com filtros básicos e gráficos dinâmicos.​

X Garantir que o dashboard corre localmente sem erros críticos e com carregamento fluido de gráficos.​

Etapa 8 – Deploy & DevOps
X Criar Dockerfile para a aplicação Flask com todas as dependências necessárias.​

X Criar docker-compose.yml para orquestrar aplicação e outros serviços, se necessário.​

X Validar que a aplicação corre com docker-compose up em localhost:5000.​

X Opcionalmente, preparar configuração para deploy em servidor remoto ou serviço de cloud.​

Etapa 9 – Relatório & apresentação
X Escrever relatório completo com secções: resumo, introdução, estado da arte, metodologia, caracterização do dataset, resultados, comparação de modelos, limitações, trabalho futuro, bibliografia.​

X Incluir gráficos e tabelas no relatório para suportar a análise dos resultados.​

X Exportar relatório final em PDF para a pasta docs/.​

X Criar apresentação em PowerPoint com 10–15 slides sobre motivação, dataset, melhores resultados, clusters, anomalias e conclusões.​

X Incluir screenshots do dashboard Flask e exemplos de previsões na apresentação.​

Etapa 10 – Organização, comentários & revisão final
X Garantir que todo o código está organizado por módulos lógicos (data, models, api, utils).​

X Adicionar docstrings e comentários explicativos às funções e classes principais.​

X Verificar que notebooks executam do início ao fim sem erros e com outputs atualizados.​

X Remover ficheiros temporários, versões antigas e código morto do repositório.​

X Fazer revisão final do código, testar demo completa e preparar pacote final de entrega (código, relatório, slides).​