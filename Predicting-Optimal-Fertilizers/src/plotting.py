import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pct_dist(df: pd.DataFrame, col_list: list ):
    for col in col_list:
        pcts = (df[col].value_counts(normalize=True) * 100)
        ax = pcts.plot(kind='bar', figsize=(12, 8))
        for patch in ax.patches:
            # Get the height of the bar (which is the percentage)
            height = patch.get_height()
            if len(ax.patches)>20:
                p = f'{height:.0f}%'
            else:
                p = f'{height:.1f}%'
            ax.text(
                x=patch.get_x() + patch.get_width() / 2,  # Horizontally center the text
                y=height+.04,                          # Place text slightly above the bar
                s=p,                      # The text to display, formatted to .0f or 1.f
                ha='center'                              # Horizontal alignment
            )
        ax.set_xlabel(col)
        ax.set_ylabel('Percentage')
        ax.set_title(f'Percentage Distribution of {col}')
        plt.tight_layout()
        plt.show()

def plot_bar_chart(df: pd.DataFrame, col_list: list, targ_col: str, stacked: bool = True):
    col_list2 = col_list.copy()
    col_list2.remove(targ_col)
    for col in col_list2:
        crosstab = pd.crosstab(df[col], df[targ_col], normalize='index')

        fig, ax = plt.subplots(figsize=(12, 8))
        crosstab.plot(kind='bar', stacked=stacked, ax=ax)
        for container in ax.containers:
            for patch in container:
                width = patch.get_width()
                height = patch.get_height()
                x, y = patch.get_xy()
                
                if height > 0.01: 
                    ax.text(x + width / 2.,
                            y + height / 2.,
                            f'{height*100:.0f}',
                            ha='center',
                            va='center',
                            fontsize=11,    
                            color='black') 
        ax.legend(title='Fertilizer Name', bbox_to_anchor=(1.01,1), fontsize=15)
        ax.set_title(f'Proportion of Fertilizer Types by {col}', fontsize = 15)
        ax.set_ylabel('Proportion', fontsize=15)
        ax.set_xlabel(f'{col}', fontsize=15)
        plt.show()
    del col_list2

def plot_Plift_heatmap(df: pd.DataFrame, col_list: list, targ_col: str, Plog: bool = False):
    col_list2 = col_list.copy()
    col_list2.remove(targ_col)
    fert_counts = df[targ_col].value_counts(normalize=True)
    for col in col_list2:
        crosstab = pd.crosstab(df[col], df[targ_col], normalize='index')
        ax, fig = plt.subplots(figsize=(12, 8))
        if Plog:
            P_lift = np.log(crosstab/fert_counts)*100
            title = plt.title(f'Log-transformed lift for {col}')
        else:
            P_lift = (crosstab - fert_counts)/fert_counts
            title = plt.title(f'Percent Lift for {col}')
        sns.heatmap(P_lift,
                    cmap='coolwarm',
                    annot=True,
                    fmt='.1f',
                    center=0)
    del col_list2