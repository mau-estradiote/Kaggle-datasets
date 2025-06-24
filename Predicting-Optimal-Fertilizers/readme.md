# Predicting Optimal Fertilizer

## 🎯 Introduction and Goal

This project is a solution for the Kaggle Playground Series competition (Season 5, Episode 6). The data was synthetically generated from a real-world dataset concerning fertilizer recommendations.

The primary goal is to build a classification model capable of recommending the optimal fertilizer based on soil and environmental data, such as temperature, humidity, and NPK values. The model's performance is evaluated using the **Mean Average Precision at 3 (MAP@3)** metric, which rewards ranked predictions. This means the model must recommend up to three fertilizers, with higher scores given for correct predictions ranked higher in the list. More information is available at the competition's [website](https://www.kaggle.com/competitions/playground-series-s5e6/overview). 

## 📁 Project Structure & Workflow

<details>
<summary><strong>📁 Project Structure</strong> (click to expand)</summary>

<ul>
    <li>📄 .gitignore</li>
    <li>📄 README.md</li>
    <li>📄 requirements.txt</li>
    <li>
        <details>
            <summary><strong>📂 data/</strong></summary>
            <ul>
                <li>📄 sample_submission.csv</li>
                <li>📄 test.csv</li>
                <li>📄 train.csv</li>
                <li>
                    <details>
                        <summary><strong>📂 processed/</strong></summary>
                        <ul>
                            <li>📄 pof_df_proc.csv</li>
                            <li>📄 pof_df_test_proc.csv</li>
                        </ul>
                    </details>
                </li>
            </ul>
        </details>
    </li>
    <li>
        <details>
            <summary><strong>📂 images/</strong></summary>
            <ul>
                <li>📄 Project workflow.png</li>
            </ul>
        </details>
    </li>
    <li>
        <details>
            <summary><strong>📂 models/</strong></summary>
            <ul>
                <li>
                    <details>
                        <summary><strong>📂 preprocessors/</strong></summary>
                        <ul>
                            <li>📄 preproc_artifacts.joblib</li>
                        </ul>
                    </details>
                </li>
            </ul>
        </details>
    </li>
    <li>
        <details>
            <summary><strong>📂 notebooks/</strong></summary>
            <ul>
                <li>📓 EDA_pof.ipynb</li>
                <li>📓 Models.ipynb</li>
            </ul>
        </details>
    </li>
    <li>
        <details>
            <summary><strong>📂 src/</strong></summary>
            <ul>
                <li>📄 __init__.py</li>
                <li>📄 evaluation.py</li>
                <li>📄 feature_engineering.py</li>
                <li>📄 plotting.py</li>
                <li>📄 preprocessing_data.py</li>
                <li>📄 submission.py</li>
                <li>📄 training.py</li>
            </ul>
        </details>
    </li>
</ul>
</details>
<img src="images/Project workflow.png" alt="Project Workflow Diagram" width="750"/>

### Key Methodologies
* **Modular Design:** The project is organized into separate notebooks (`EDA_pof.ipynb`, `Models.ipynb`) for exploration and modeling, with reusable functions stored in the `src/` directory.
* **Reproducible Preprocessing:** A Scikit-learn pipeline, saved as a `.joblib` artifact, ensures consistent data transformation (one-hot encoding, scaling) across all experiments.
* **Experiment Tracking:** **MLflow** was used to systematically log all training runs, including parameters, performance metrics (MAP@3), and model artifacts.
* **Iterative Modeling:** A baseline model was established first. Insights from its evaluation (e.g., feature importances) were then used to guide subsequent feature engineering cycles.

---

## 📊 Exploratory Data Analysis (EDA)

The initial analysis revealed a clean dataset with 750,000 training samples and no missing values. A key insight was the **low cardinality** of all predictive features. Even numerical features like `Temperature` had only 14 unique values, suggesting they could be treated as discrete. This guided the visualization strategy, which favored bar plots and heatmaps to uncover relationships between the features and the seven target fertilizer classes.

**For a complete, detailed analysis, please see the `notebooks/EDA_pof.ipynb` notebook.**

#### Data Overview

| Feature | Unique Values |
|:---|---:|
| Temperature | 14 |
| Humidity | 23 |
| Moisture | 41 |
| Soil Type | 5 |
| Crop Type | 11 |
| Nitrogen | 39 |
| Potassium | 20 |
| Phosphorous | 43 |
| Fertilizer Name | 7 |

-----------------------------------------------------------

| | Temperature | Humidity | Moisture | Nitrogen | Potassium | Phosphorous |
|:---|---:|---:|---:|---:|---:|---:|
| **count** | 750000 | 750000 | 750000 | 750000 | 750000 | 750000 |
| **mean** | 31.50 | 61.04 | 45.18 | 23.09 | 9.48 | 21.07 |
| **std** | 4.03 | 6.65 | 11.79 | 11.22 | 5.77 | 12.35 |
| **min** | 25 | 50 | 25 | 4 | 0 | 0 |
| **25%** | 28 | 55 | 35 | 13 | 4 | 10 |
| **50%** | 32 | 61 | 45 | 23 | 9 | 21 |
| **75%** | 35 | 67 | 55 | 33 | 14 | 32 |
| **max** | 38 | 72 | 65 | 42 | 19 | 42 |

---

## 🧪 Modeling & Results

The modeling process was iterative and tracked with MLflow. A baseline was established first, followed by experiments with feature engineering to improve performance.

| Experiment ID | Key Change / Hypothesis | Validation MAP@3 | Key Learning |
| :--- | :--- | :--- | :--- |
| **Run 1** | **Baseline Model** (No Feature Engineering) | 0.30930 | LGBM outperforms RF. `Soil Type` and nutrient levels are the top features. Model confidence is low. |

---

## 🚀 Getting Started

Follow these steps to set up the project locally.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/mau-estradiote/Kaggle-datasets/tree/master/Predicting-Optimal-Fertilizers
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