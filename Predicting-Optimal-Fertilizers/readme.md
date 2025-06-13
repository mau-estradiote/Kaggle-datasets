# Predicting Optimal Fertilizer

## 1. Introduction and goal

This project is a solution for the Kaggle Playground Series competition (Season 5, Episode 6). The data was synthetically generated from a real-world dataset concerning fertilizer recommendations.

The primary goal is to build a classification model capable of recommending the optimal fertilizer based on soil and environmental data, such as temperature, humidity, and NPK values. The model's performance is evaluated using the **Mean Average Precision at 3 (MAP@3)** metric, which rewards ranked predictions. This means the model must recommend up to three fertilizers, with higher scores given for correct predictions ranked higher in the list. More information is available at the competition's [website](https://www.kaggle.com/competitions/playground-series-s5e6/overview). 

# 2. Project Workflow and Methodology

This project was structured to follow some best practices for data science, including: emphasizing reproducibility, modularity, and iterative improvement. The complete workflow is visualized in the diagram below.

![Project Workflow Diagram](images/Project%20workflow.png)

The key strategies employed include:

* **Separation of Concerns:** The project is organized into two main notebooks (`EDA_pof.ipynb` and `Models.ipynb`) to separate exploratory work from final model training and evaluation.
* **Reproducible Preprocessing:** A preprocessing pipeline was built using Scikit-learn and saved as a `.joblib` artifact. This pipeline handles one-hot encoding for categorical features and standard scaling for numerical features, ensuring consistent data transformation.
* **Experiment Tracking with MLflow:** All model training experiments were logged using **MLflow**. This created a systematic record of parameters, performance metrics (MAP@3), and model artifacts for reproducibility.
* **Iterative Modeling:** A baseline performance was first established. Insights from evaluating this baseline (e.g., feature importances, confusion matrices) were then used to inform the next cycle of feature engineering, creating a loop of continuous improvement.

## 3. Exploratory Data Analysis (EDA)

The initial analysis revealed a clean dataset with 750,000 training samples and no missing values. A key insight from the EDA was the **low cardinality** of all predictive features. Even numerical features like 'Temperature' had a small number of unique values (14), suggesting they could be treated as discrete categories. This insight guided the visualization strategy, favoring bar plots and heatmaps of "percentage lift" to uncover relationships between the features and the 7 target fertilizer classes.

**For a complete and detailed analysis with all visualizations, please see the `notebooks/EDA_pof.ipynb` notebook.**


### 3.1 Data overview

| Features        |Unique Values|
|:----------------|----:|
| Temperature     |  14 |
| Humidity        |  23 |
| Moisture        |  41 |
| Soil Type       |   5 |
| Crop Type       |  11 |
| Nitrogen        |  39 |
| Potassium       |  20 |
| Phosphorous     |  43 |
| Fertilizer Name |   7 |

---------------------------------------------------------------------------------------------------------

|       |   Temperature |    Humidity |    Moisture |    Nitrogen |    Potassium |   Phosphorous |
|:------|--------------:|------------:|------------:|------------:|-------------:|--------------:|
| count |  750000       | 750000      | 750000      | 750000      | 750000       |   750000      |
| mean  |      31.5036  |     61.0389 |     45.1841 |     23.0938 |      9.4783  |       21.0732 |
| std   |       4.02557 |      6.6477 |     11.7946 |     11.2161 |      5.76562 |       12.3468 |
| min   |      25       |     50      |     25      |      4      |      0       |        0      |
| 25%   |      28       |     55      |     35      |     13      |      4       |       10      |
| 50%   |      32       |     61      |     45      |     23      |      9       |       21      |
| 75%   |      35       |     67      |     55      |     33      |     14       |       32      |
| max   |      38       |     72      |     65      |     42      |     19       |       42      |

## 4. Modeling & Experiment Log

The modeling process was iterative. A baseline was established first, followed by experiments with feature engineering. The progress was tracked using MLflow.

| Experiment ID | Key Change / Hypothesis                                | Validation MAP@3 | Key Learning                                                                 |
| :------------ | :----------------------------------------------------- | :--------------- | :--------------------------------------------------------------------------- |
| **Run 1** | **Baseline Model** (No Feature Engineering)            | 0.30930            | LGBM outperforms RF. `Soil Type` and nutrient levels are the top features. Model confidence is low. |

## 5. How to Use This Repository

1.  **Clone the Repository:**
    ```bash
    git clone [your-repository-url]
    cd Predicting-Optimal-Fertilizer
    ```

2.  **Download the Data:**
    The dataset is not included in this repository as per version control best practices for large files.
    - Download the data from the [official Kaggle competition page](https://www.kaggle.com/competitions/playground-series-s5e6/data).
    - Place `train.csv`, `test.csv`, and `sample_submission.csv` inside the `data/` directory. The `data/processed/` subfolder will be created automatically when running the EDA notebook.

3.  **Set Up the Environment:**
    Create a virtual environment and install the required packages (a `requirements.txt` file will be provided).
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Analysis:**
    To review the analysis and run the models, execute the notebooks in the `notebooks/` folder in order, starting with `EDA_pof.ipynb`.

5.  **Track Experiments:**
    To launch the MLflow UI and view experiment results, run the following command from the project's root directory:
    ```bash
    python -m mlflow ui
    ```