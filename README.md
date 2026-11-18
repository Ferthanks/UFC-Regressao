# 🥊 Predição de Resultados no UFC com Regressão Logística

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Econometrics%20%26%20Stats-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)

## 📌 Visão Geral do Projeto

Este projeto utiliza **Regressão Logística (Logit)** para modelar e prever a probabilidade de vitória de um lutador do UFC com base no seu desempenho técnico durante o **1º round** aliado a variáveis de contexto biológico.

Em vez de tratar o modelo apenas como uma "caixa-preta" de Machine Learning, o foco desta análise é a **inferência estatística**, a **seleção rigorosa de variáveis** e a **interpretabilidade** dos coeficientes através de *Odds Ratios* (Razão de Chances).

### 🎯 Escopo Temporal (Prevenção de *Concept Drift*)
O MMA moderno evoluiu drasticamente em termos de regras, arbitragem e preparação física. Para evitar distorções históricas (*Concept Drift*), a base foi filtrada para contemplar exclusivamente lutas da **Era Moderna (a partir de 2016)** que avançaram além do 1º round, totalizando **4.196 combates** analisados.

---

## 💡 Principais Descobertas

1. **O Fator Idade é Determinante:** Modelos puramente técnicos ignoram o desgaste biológico. A inclusão do diferencial de idade (`diff_age`) provou-se altamente significativa ($p < 0.001$). Mantendo o desempenho técnico constante, **cada ano a mais de idade em relação ao oponente reduz as chances de vitória em ~5,6%**.
2. **Volume e Agressividade no 1º Round:** Cada golpe significativo conectado a mais (`diff_sig_landed`) aumenta a chance de vitória em **~6,2%**, enquanto cada tentativa de finalização (`diff_sub_att`) eleva a chance em **~17,4%**.
3. **Domínio Posicional:** Quedas bem-sucedidas (`diff_td_success`, **+15,2%** por queda) e tempo de controle no solo/clinch (`diff_ctrl_sec`) ditam o ritmo e consolidam a vantagem estatística.
4. **O que NÃO importa estatisticamente:** Reversões de posição no solo (`diff_rev`) apresentaram $p = 0.536$, não agregando poder preditivo isolado e sendo removidas na versão final.

---

## 🔄 Evolução e Seleção dos Modelos

O desenvolvimento seguiu uma abordagem iterativa baseada no **Princípio da Parsimónia (Navalha de Ockham)**:

* **Modelo V1 (Baseline Técnico):** Focado apenas em métricas brutas de volume (golpes, quedas, controle e finalizações). Bom ponto de partida, mas descontextualizado fisicamente.
* **Modelo V2 (Contexto + Solo):** Adição da diferença de idade (`diff_age`) e reversões (`diff_rev`). Comprovou a relevância da idade, mas expôs a insignificância estatística das reversões.
* **Modelo Final V3 (Enxuto / Padrão-Ouro):** Remoção de ruído (`diff_rev`), mantendo exclusivamente as 5 variáveis preditoras com significância estatística comprovada ($p < 0.05$).

---

## 📊 Resultados do Modelo Final (V3)

### Sumário Estatístico e *Odds Ratios*
* **Observações ($N$):** 4.196 lutas (2016+)
* **Log-Likelihood:** -2491.1 ($LLR\ p\text{-value} = 1.83 \times 10^{-161}$)
* **Pseudo $R^2$ (McFadden):** 0.1320

| Variável (Diferencial Red - Blue) | Coeficiente ($\beta$) | *Odds Ratio* ($e^\beta$) | P-valor ($P>\Vert{}z\Vert{}$) | Impacto Prático na Chance de Vitória |
| :--- | :---: | :---: | :---: | :--- |
| **`const` (Intercepto)** | `0.2686` | `1.3081` | `0.000` | Vantagem basal histórica do corner Vermelho |
| **`diff_sig_landed`** | `0.0599` | `1.0617` | `0.000` | **+6,17%** por golpe significativo a mais |
| **`diff_td_success`** | `0.1414` | `1.1519` | `0.000` | **+15,19%** por queda aplicada a mais |
| **`diff_ctrl_sec`** | `0.0026` | `1.0026` | `0.000` | **+0,26%** por segundo extra de controle |
| **`diff_sub_att`** | `0.1600` | `1.1735` | `0.015` | **+17,35%** por tentativa de submissão a mais |
| **`diff_age`** | `-0.0576` | `0.9440` | `0.000` | **-5,60%** por cada ano a mais que o oponente |

### Performance Preditiva
* **AUC-ROC:** **0.743** (Boa capacidade de discriminação entre vencedores e perdedores apenas com dados do R1).
* **Acurácia Global:** **68,8%** (2.885 previsões corretas em 4.196 lutas).

---

## 🗂️ Estrutura do Repositório

```text
UFC_Regressao/
├── data/
│   └── processed/
│       └── ufc_r1_otimizado.csv       # Base limpa e tratada (2016+) utilizada no modelo final
├── notebooks/
│   ├── 01_eda_e_colinearidade.ipynb   # Análise Exploratória (EDA) e diagnóstico de VIF/Colinearidade
│   └── 03_modelo_otimizado.ipynb      # Treinamento V2/V3, Odds Ratios, Matriz de Confusão e Curva ROC
├── outputs/
│   └── figures/                       # Gráficos exportados das avaliações do modelo
├── src/
│   ├── preprocessing.py               # Pipeline de limpeza inicial
│   ├── preprocessing_v1.py            # Versão preliminar de extração
│   └── preprocessing_otimizado.py     # Pipeline final (cruzamento de DOB, cálculo de idade e diferenciais)
├── .gitignore
├── requirements.txt
└── README.md
