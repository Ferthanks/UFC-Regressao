# 🥊 Previsão de Resultados no UFC: Uma Análise de Regressão Logística e Concept Drift

## 📌 Visão Geral do Projeto
Este projeto utiliza **Machine Learning (Regressão Logística)** para prever os vencedores de combates do UFC com base exclusivamente nas estatísticas de desempenho do 1º round. 
O estudo vai além da simples previsão ao identificar e provar a existência de **Concept Drift** no MMA: uma mudança estrutural na forma como as lutas são julgadas e vencidas após as alterações de regras de 2016 e a entrada da USADA.

## ⚙️ Metodologia e Limpeza de Dados
*   **Base de Dados:** Histórico completo de lutas e estatísticas golpe a golpe do UFC.
*   **Engenharia de Recursos:** Criação de variáveis diferenciais (Red - Blue) para capturar a vantagem relativa entre os lutadores (ex: `diff_sig_landed`, `diff_td_success`, `diff_ctrl_sec`).
*   **Tratamento de Colinearidade:** Remoção de variáveis redundantes (como total de golpes tentados vs. acertados) para garantir a estabilidade dos coeficientes (Odds Ratios).

## 📊 Principais Descobertas: A Era Moderna (2016+) vs Histórico
A comparação entre um modelo treinado com todos os dados do UFC e um focado apenas na "Era Moderna" (2016 em diante, n=4.196) revelou mudanças drásticas no esporte:
1. **O Declínio do Grappling Passivo:** O impacto de quedas bem-sucedidas caiu de 16,5% para 15,2%, e a precisão dos golpes (`diff_sig_acc`) deixou de ser estatisticamente significativa.
2. **A Era do Dano e Agressividade:** Tentativas de finalização e volume bruto de golpes conectados ganharam relevância, alinhando-se aos novos critérios de julgamento focados em impacto real.
3. **Queda do Viés do Favorito:** A vantagem inerente de lutar no corner Vermelho (frequentemente o campeão/favorito) despencou de 62,6% (histórico) para apenas 24,7% (2016+), mostrando um esporte muito mais competitivo e imprevisível.

## 📈 Desempenho do Modelo
O modelo final (Era Moderna) focado nas estatísticas do 1º round atingiu os seguintes resultados:
*   **Acurácia:** ~68,5% de acerto na previsão do vencedor.
*   **Curva ROC (AUC):** 0.733, demonstrando forte capacidade de separar vencedores de perdedores precocemente na luta, mesmo com a atual imprevisibilidade do desporto.

## 🛠️ Tecnologias Utilizadas
*   **Python** (Pandas, Numpy) para manipulação de dados.
*   **Statsmodels & Scikit-Learn** para regressão logística, cálculo de Odds Ratios e métricas de avaliação.
*   **Matplotlib & Seaborn** para visualização de dados (Matriz de Confusão, ROC).