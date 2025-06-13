import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def calculate_map3(y_true, y_pred_proba):

    top_3 = np.argsort(y_pred_proba, axis=1)[:, -3:]
    scores = []
    for i, true_label in enumerate(y_true):
        if true_label in top_3[i]:
            rank = 3 - np.where(np.flip(top_3[i]) == true_label)[0][0]
            if rank == 1:
                scores.append(1.0)
            elif rank == 2:
                scores.append(0.5)
            elif rank == 3:
                scores.append(1/3)
        else:
            scores.append(0.0)
        score = np.mean(scores)
    return np.mean(scores)

def plot_confusion_matrix(model, X_val, y_val, class_names):

    fig, ax = plt.subplots(figsize=(10, 8))
    
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_val,
        y_val,
        ax=ax,
        cmap='Blues',
        display_labels=class_names
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.title("Confusion Matrix", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names):
    
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_importances = importances.sort_values(ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    top_importances.plot(kind='barh', color='skyblue')
    plt.title("Top 20 Feature Importances", fontsize=16)
    plt.gca().invert_yaxis()
    plt.show()