import pandas as pd     #need version 1.4.0 ! 
import joblib

def make_matrix(df):

    copy_df = df 
    matrixCWE = copy_df.groupby(['product', 'cwe']).size().unstack(fill_value=0)
    matrixCWE = matrixCWE.applymap(lambda x: 1 if x != 0 else 0)
    
    return matrixCWE

def make_matix2(df):
    copy_df = df 
    matrixCWE = copy_df.groupby(['user', 'item']).size().unstack(fill_value=0)
    matrixCWE = matrixCWE.applymap(lambda x: 1 if x != 0 else 0)
    
    return matrixCWE


def save_matrix(df):
    matrix = make_matrix(df)
    joblib.dump(matrix, 'matrix.pkl')


def load_matrix():
    loaded_matrix = joblib.load('matrix.pkl')
    return loaded_matrix