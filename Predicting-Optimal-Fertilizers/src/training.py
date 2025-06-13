import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV

def train_and_log_models(X_train: pd.DataFrame, y_train: pd.Series, models_to_run: list, all_models: dict, all_params: dict, n_iter=10, cv=3):

    fitted_models = {}

    for model_name in models_to_run:
        print(f"----- Starting Experiment for: {model_name} -----")
        
        model_obj = all_models[model_name]
        param_grid = all_params[model_name]
        
        with mlflow.start_run(run_name=f"{model_name}_random_search"):
            
            mlflow.log_param("model_name", model_name)

            rscv = RandomizedSearchCV(
                estimator=model_obj,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring='neg_log_loss',
                random_state=42,
                n_jobs=2
            )
            rscv.fit(X_train, y_train)

            mlflow.log_params(rscv.best_params_)

            mlflow.log_metric("best_cv_neg_log_loss", rscv.best_score_)

            mlflow.sklearn.log_model(rscv.best_estimator_, f"model_{model_name}")
            
            print(f"Finished training for {model_name}. Best CV Score: {rscv.best_score_:.4f}")
            
            fitted_models[model_name] = rscv.best_estimator_
            
    return fitted_models