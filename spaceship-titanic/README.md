# Análise e Modelo - Kaggle Spaceship Titanic 

Este projeto aborda a competição [Spaceship Titanic do Kaggle](https://www.kaggle.com/competitions/spaceship-titanic). O objetivo é construir um modelo de Machine Learning para prever quais passageiros foram transportados para uma dimensão alternativa durante uma colisão com uma anomalia do espaço-tempo, com base em dados recuperados do sistema danificado da nave. Esta é a Versão 1 do projeto, focando em estabelecer um *pipeline* funcional e um modelo *baseline* robusto.

Os dados foram fornecidos pelo Kaggle em formato CSV e incluem informações sobre os passageiros, como:

* Dados demográficos (`PassengerId`, `HomePlanet`, `Name`, `Age`)
* Detalhes da viagem (`CryoSleep`, `Cabin`, `Destination`, `VIP`)
* Registros de gastos em serviços a bordo (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`)
* A variável alvo (`Transported` - True/False) presente no conjunto de treino.

  ## 3. *Workflow* e Metodologia

O desenvolvimento seguiu as seguintes etapas:

### 3.1. *Exploratory Data Analysis* (EDA)

* Análise inicial de tipos de dados e valores ausentes (NaNs), identificando colunas com maior percentual de ausência (como `Spa`, `VRDeck`, etc.).
* Criação da *feature* `TotalSpending` (soma dos gastos individuais) e análise de sua distribuição, que se mostrou altamente assimétrica (muitos valores zero). Cerca de 45% dos passageiros não tiveram gastos registrados.
* Análise das distribuições das features numéricas (`Age`, gastos) em relação ao alvo `Transported`. Observou-se que passageiros com gastos totais/individuais iguais a zero tinham maior probabilidade de serem transportados.
* Análise das features categóricas/binárias:
    * `CryoSleep` mostrou ser um forte indicador (quem estava em sono criogênico tinha taxa de transporte muito maior).
    * `HomePlanet` também mostrou relevância (passageiros da Terra eram menos transportados, de Europa mais transportados).
    * `Destination` e `VIP` pareceram ter menos impacto inicial.
* Cálculo da matriz de correlação, confirmando relações lineares fracas entre a maioria das features numéricas e o alvo, mas destacando correlações negativas (fracas) entre os gastos (`RoomService`, `Spa`, `VRDeck`, `TotalSpending`) e `Transported`.

### 3.2. Pré-processamento

* **Tratamento de *Missing Values* **Criei funções customizadas ao invés de usar do scikit-learn, afim de aprimorar algumas habilidades. *Nota: Colunas indicadoras de ausência não foram adicionadas nesta versão, visto que em análise prévia o modelo não usou tais features.*
* **Engenharia de Features:** Criação da coluna `TotalSpending`.
* **Codificação de Features:**
    * Features Categóricas (`HomePlanet`, `Destination`): Codificadas através do método de *One-Hot Encoding*.
    * Features Binárias (`CryoSleep`, `VIP`): Convertidas para formato numérico 0/1.
* **Escalonamento:** Não foi aplicado, pois o modelo escolhido (Random Forest) não é sensível à escala das features numéricas.
* **Features Excluídas:** `PassengerId`, `Name`, `Cabin` (nesta versão).
* **Divisão Treino/Validação:** O conjunto de treino foi dividido em 80% para treino e 20% para validação local, usando `train_test_split` com estratificação pela variável alvo.

### 3.3. Modelagem e *Tuning*

* **Modelo Escolhido:** `RandomForestClassifier` (Scikit-learn), devido à sua robustez a outliers, capacidade de lidar com relações não-lineares, menor chance de *overfitting* e ser recomendado para problemas de classificação.
* **Ajuste de Hiperparâmetros:** Foi utilizado `RandomizedSearchCV` com validação cruzada (5 *folds*) para buscar uma combinação otimizada de hiperparâmetros. Os melhores parâmetros encontrados foram:
    ```python
    {'criterion': 'entropy', 'max_depth': 10, 'max_features': 0.5, 'min_samples_leaf': 3, 'min_samples_split': 20, 'n_estimators': 879}
    ```

## 4. Resultados da Avaliação

O modelo final (*Random Forest* com os hiperparâmetros otimizados) foi avaliado no conjunto de validação (20% dos dados de treino originais).

* **Acurácia:** 79.0%
* **Relatório de Classificação:**
    ```
                         precision    recall  f1-score   support

    Não Transportado (0)       0.80      0.76      0.78       863
      Transportado (1)       0.78      0.81      0.79       876

              accuracy                           0.79      1739
             macro avg       0.79      0.79      0.79      1739
          weighted avg       0.79      0.79      0.79      1739
    ```
* **Matriz de Confusão:**
* **Importância das Features:** As features mais importantes identificadas pelo modelo foram `TotalSpending`, `CryoSleep`, `Spa`, `FoodCourt` e `VRDeck`.

## 5. Conclusões e Próximos Passos

O modelo Random Forest otimizado apresentou uma performance razoável (79.0% de acurácia na validação) para esta primeira versão, confirmando a importância dos gastos totais e do estado de criosono.

**Próximos Passos Planejados:**

* Realizar engenharia de features na coluna `Cabin` (extrair Deck, Lado, talvez Número).
* Investigar a coluna `Name` para possível criação de features de tamanho de família/grupo.
* Experimentar outros modelos, como Gradient Boosting (XGBoost, LightGBM).
* Realizar uma análise de erros mais aprofundada nas previsões do modelo atual.
