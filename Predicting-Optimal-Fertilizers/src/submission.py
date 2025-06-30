import pandas as pd
import numpy as np

def submission(model, id, x_test, fname):
    y_pred_proba = model.predict_proba(x_test)
    top_3 = np.argsort(y_pred_proba, axis=1)[:, -3:]
    pd.DataFrame({
        'id': id['id'],
        'Fertilizer Name': top_3['Fertilizer Name']
    }).to_csv(fname+'.csv', index=False)