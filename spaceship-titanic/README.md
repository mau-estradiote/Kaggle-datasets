# Projeto Spaceship Titanic - Previsão de Transporte (Kaggle)

## 1. Introdução e Objetivo

Este projeto aborda a competição [Spaceship Titanic do Kaggle](https://www.kaggle.com/competitions/spaceship-titanic). O objetivo é construir um modelo de Machine Learning para prever quais passageiros (variável `Transported`) foram transportados para uma dimensão alternativa durante uma colisão da nave, com base em dados recuperados do sistema danificado.

Esta documentação reflete a Versão 1 do projeto, focando em estabelecer um pipeline funcional desde a análise exploratória até a avaliação de modelos de classificação robustos. O trabalho foi estruturado em dois notebooks Jupyter principais:

1.  `EDA_Preprocessing.ipynb`: Contém a Análise Exploratória dos Dados (EDA) e todo o Pré-processamento, gerando arquivos `.csv` com os dados limpos e preparados.
2.  `Modeling.ipynb`: Carrega os dados processados e foca no treinamento, ajuste de hiperparâmetros e avaliação dos modelos de Machine Learning.

## 2. O Dataset

Os dados foram fornecidos pelo Kaggle (`train.csv`, `test.csv`) e incluem informações como:

* Dados demográficos (`PassengerId`, `HomePlanet`, `Name`, `Age`)
* Detalhes da viagem (`CryoSleep`, `Cabin`, `Destination`, `VIP`)
* Registros de gastos a bordo (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`)
* A variável alvo (`Transported` - True/False) presente apenas no conjunto de treino.

## 3. Workflow e Metodologia

O desenvolvimento seguiu as seguintes etapas principais, detalhadas nos notebooks:

### 3.1. Análise Exploratória de Dados (EDA)

* Análise inicial de tipos, valores únicos e dados ausentes (NaNs), com destaque para colunas de gastos e `Age`.
* Investigação das distribuições das variáveis numéricas e categóricas.
* Análise da correlação entre features numéricas e a variável alvo, mostrando relações lineares fracas em geral, mas indicando tendências (ex: correlação negativa entre gastos e `Transported`).
* Identificação de padrões importantes:
    * A feature `CryoSleep` mostrou forte correlação com `Transported`.
    * `HomePlanet` indicou diferenças significativas (passageiros da Terra menos transportados, Europa mais).
    * Um grande número de passageiros (~45%) não possuía gastos registrados (`TotalSpending == 0`), e este grupo tinha maior probabilidade de ser transportado.
* Engenharia de Features inicial baseada na EDA:
    * Criação de `TotalSpending` (soma dos gastos).
    * Criação de `ZeroSpending` (flag binária para gasto total nulo).
    * Extração de `GroupSize` a partir da estrutura do `PassengerId`.
    * Decomposição da `Cabin` em `Deck`, `Num` (descartado nesta versão) e `Side`.

### 3.2. Pré-processamento

Aplicado de forma consistente aos dados de treino e teste:

* **Tratamento de Missing Values:** Utilizadas funções customizadas (via notebook `EDA_Preprocessing.ipynb`) para imputar NaNs, usando mediana para features numéricas e moda para categóricas/binárias. *Nota: Colunas indicadoras de ausência explícitas não foram mantidas na versão final, pois análise de importância prévia mostrou baixo impacto para o Random Forest.*
* **Engenharia de Features:** Adicionadas as colunas `TotalSpending`, `ZeroSpending`, `GroupSize`, `Deck`, `Side`.
* **Codificação de Features:**
    * Categóricas (`HomePlanet`, `Destination`, `Deck`): One-Hot Encoding.
    * Binárias (`CryoSleep`, `VIP`, `Side`): Conversão para 0/1.
* **Features Excluídas:** `PassengerId`, `Name`, `Cabin` (original), `Cabin_Num`.
* **Escalonamento:** Não aplicado, devido à escolha de modelos baseados em árvore.
* **Divisão Treino/Validação:** O conjunto de treino original foi dividido (80% treino / 20% validação) usando `train_test_split` com `stratify=y` e `random_state` para reprodutibilidade.

### 3.3. Modelagem e Tuning

Foram testados três modelos de classificação baseados em árvore: Random Forest, LightGBM e XGBoost. O foco inicial de tuning e análise detalhada foi no Random Forest.

* **Modelo Principal:** `RandomForestClassifier` (Scikit-learn).
* **Ajuste de Hiperparâmetros (RF):** Utilizou-se `RandomizedSearchCV` com validação cruzada (5 folds, 25 iterações) para otimizar os hiperparâmetros. Os melhores parâmetros encontrados para o RF foram:
    ```python
    {'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 15, 'n_estimators': 666}
    ```
* **(LGBM e XGBoost):** Também foram ajustados usando `RandomizedSearchCV` (detalhes no notebook `Modeling.ipynb`).

## 4. Resultados da Avaliação

Os modelos foram avaliados no conjunto de validação local e submetidos ao Kaggle. Os resultados foram consistentes entre os três modelos testados:

* **Random Forest:** Acurácia Validação ≈ **80.0%** | Score Kaggle ≈ **0.791**
* **LightGBM:** Acurácia Validação ≈ **79.4%** | Score Kaggle ≈ **0.791**
* **XGBoost:** Acurácia Validação ≈ **79.3%** | Score Kaggle ≈ **0.791**

**Detalhes do Melhor Modelo (Random Forest na Validação):**

* **Relatório de Classificação:**
    ```
                         precision    recall  f1-score   support

    Não Transportado (0)       0.81      0.77      0.79       863
      Transportado (1)       0.78      0.82      0.80       876

              accuracy                           0.80      1739
             macro avg       0.80      0.80      0.80      1739
          weighted avg       0.80      0.80      0.80      1739
    ```
* **Importância das Features:** As features mais relevantes para o modelo RF foram `TotalSpending`, `ZeroSpending`, `Spa`, `FoodCourt` e `CryoSleep`. Features derivadas de `Cabin` (`Deck`, `Side`) e `GroupSize` tiveram importância secundária nesta configuração.

## 5. Conclusões e Próximos Passos

O pipeline desenvolvido e o modelo Random Forest otimizado atingiram uma performance sólida e competitiva (Acurácia ~80% validação, ~79% Kaggle) para esta primeira versão, utilizando features criadas a partir da EDA (`TotalSpending`, `ZeroSpending`, `GroupSize`) e da decomposição da `Cabin`. A importância do estado de `CryoSleep` e dos padrões de gastos foi confirmada.

**Próximos Passos Planejados:**

* **Refinar Engenharia de Features:**
    * Investigar o `Cabin_Num` (descartado) através de binning ou outras técnicas.
    * Analisar a feature `Name` para extrair tamanho de família/grupo e comparar com `GroupSize`.
    * Experimentar encodings alternativos para `Deck` (ex: Target Encoding com validação cruzada).
* **Otimizar Modelos:** Realizar tuning mais extenso para LightGBM e XGBoost.
* **Análise de Erros:** Investigar os erros de classificação (Falsos Positivos e Negativos) para gerar novos insights.
* **Testar Ensembling/Stacking:** Combinar as previsões dos diferentes modelos.
