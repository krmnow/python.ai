import os
import numpy as np
import pandas as pd
import matplotlib as plt 
import seaborn as sns

path = 'C:\\Users\\knowak7\\Desktop\\footballAI\\football-striker-performance'
path = path.replace("\\","/")
os.chdir(path)

def load_data(csv):
    return pd.read_csv(csv)

df = load_data('StrikerPerformance.csv')
print(df.head(5))
