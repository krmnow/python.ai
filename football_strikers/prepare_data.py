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
df = df.fillna(0)
df = df.drop('nationality', axis = 1)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
position = df['position']
position_encoded = encoder.fit_transform(position.astype(str))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

pos = position.values.reshape(-1,1)
position = scaler.fit_transform(pos)
print(position.head(5))
