import matplotlib.pyplot as plt
import pandas as pd

def create_df(cols):
    '''
    take columns as arguments
    '''
    df = pd.DataFrame(columns=cols)
    return df

def create_col(col, df):
    '''
    insert a new column
    '''
    
