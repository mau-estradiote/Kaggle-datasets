# Machine Learning Projects Repository

Welcome to my Data Science and Machine Learning Projects repository!

This repository is dedicated to applying practical Machine Learning techniques to various datasets from Kaggle. The main goal is to improve my knowledge and build a strong portfolio that showcases my abilities throughout the entire data science lifecycle, including:

* Exploratory Data analysis (EDA)
* Data cleaning and preprocessing
* Feature Engineering
* Implementing and Evaluating Predictive Models (e.g., Classification, Regression)
* Interpretation of Metrics and Results

  **About me:** I am a Physicist with a Master's in Applied Physics (USP) now pivoting into Data Science. I leverage my rigorous analytical and problem-solving background to tackle complex challenges and extract value from data. I'm eager to connect, find me on [LinkedIn](https://www.linkedin.com/in/maur%C3%ADcio-estradiote-2096ab9b/)!

  ## Projects Index

All the projects follow the pipeline EDA -> Feature Engineering -> Model Training -> Model Evaluation, there's a fluxogram that illustrates this process in the [Predicting Optimal Fertilizer](./Predicting-Optimal-Fertilizers/) project readme. Here's a list of all the projects I developed so far:

* ### 🚀 [Spaceship Titanic - Previsão de Transporte](./spaceship-titanic/) (*In Portuguese*)
    * **Description:** Analysis and predictive models to predict which passengers were transported to an alternate dimension.
    * **Metric:**: Accuracy.
    * **Status:** EDA and feature engineering was accomplished. `RandomizedSearchCV` was used to fine tuning the hyperparameters. The Random Forest, LGBM, and XGBoost models achieved similar performance: ~80% validation accuracy and a Kaggle score of ~0.791.
    * **Next Step:** Leverage model insights to refine the feature engineering and try other methods such as ensembling.
* ### [Predicting Optimal Fertilizer](./Predicting-Optimal-Fertilizers/) (*In English*)
    * **Description:** Analysis and predictive models to recommend the best fertilizer. MLflows were applied to this project.
    * **Metric:** MAP@3.
    * **Status:** Only EDA was made. Random Forest and LGBM models were trained, where LGBM had the best score of 0.30930.
    * **Next Step:** Apply feature engineering and apply a systematic method to fit the models. 

---

## Tools & Technologies

Python | Pandas | NumPy | Scikit-learn | Seaborn | Matplotlib | Jupyter Notebook | Git | GitHub | MLflow
