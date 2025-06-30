import pandas as pd
import numpy as np

def nutrients_ratio(df: pd.DataFrame):
    df['N/P'] = df['Nitrogen'] / (df['Phosphorous']+1)
    df['P/K'] = df['Phosphorous'] / (df['Potassium']+1)
    df['N/K'] = df['Nitrogen'] / (df['Potassium']+1)
    df['P/N'] = df['Phosphorous'] / (df['Nitrogen']+1)
    df['K/N'] = df['Potassium'] / (df['Nitrogen']+1)
    df['K/P'] = df['Potassium'] / (df['Phosphorous']+1)
    df['npk_sum'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    return df

def moisture_nutrient(df: pd.DataFrame):
    df['moisture_x_nitrogen'] = df['Moisture'] * df['Nitrogen']
    df['moisture_x_phosphorous'] = df['Moisture'] * df['Phosphorous']
    df['moisture_x_potassium'] = df['Moisture'] * df['Potassium']
    return df

def delta_nutrient(df: pd.DataFrame):
    df['nitrogen_avg_for_soil'] = df.groupby('Soil Type')['Nitrogen'].transform('mean')
    df['nitrogen_delta'] = df['Nitrogen'] - df['nitrogen_avg_for_soil']
    df = df.drop(columns=['nitrogen_avg_for_soil'])

    df['phosphorous_avg_for_soil'] = df.groupby('Soil Type')['Phosphorous'].transform('mean')
    df['phosphorous_delta'] = df['Phosphorous'] - df['phosphorous_avg_for_soil']
    df = df.drop(columns=['phosphorous_avg_for_soil'])

    df['potassium_avg_for_soil'] = df.groupby('Soil Type')['Potassium'].transform('mean')
    df['potassium_delta'] = df['Potassium'] - df['potassium_avg_for_soil']
    df = df.drop(columns=['potassium_avg_for_soil'])
    
    return df