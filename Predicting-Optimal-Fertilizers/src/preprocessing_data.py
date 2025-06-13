import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

def preprocess(df: pd.DataFrame, df_test: pd.DataFrame, targ_col: str, fname: str):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols.remove(targ_col)

    preproc = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
        ]
    )

    label_encoder = LabelEncoder()
    fert_name = label_encoder.fit_transform(df[targ_col])
    fert_name = pd.Series(fert_name, name=targ_col+'_encoded')
    df_proc = preproc.fit_transform(df)
    df_test_proc = preproc.transform(df_test)
    cols = preproc.get_feature_names_out()
    df_train_final = pd.DataFrame(df_proc, columns=cols)
    df_test_final = pd.DataFrame(df_test_proc, columns=cols)
    df_train_final[targ_col] = fert_name

    processed_data_dir = '../data/processed/'
    os.makedirs(processed_data_dir, exist_ok=True)
    df_train_final.to_csv(os.path.join(processed_data_dir, fname+'_df_proc.csv'), index=False)
    df_test_final.to_csv(os.path.join(processed_data_dir, fname+'_df_test_proc.csv'), index=False)

    preproc_artifacts = {
        'feature_transformer': preproc,
        'target_encoder': label_encoder,
        'feature_names': cols 
    }

    artifacts_dir = '../models/preprocessors/'
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(preproc_artifacts, os.path.join(artifacts_dir, 'preproc_artifacts.joblib'))