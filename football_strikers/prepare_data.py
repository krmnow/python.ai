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
df = df.drop('current league', axis = 1)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


#pozycja
position = df['position']
position_encoded = encoder.fit_transform(position.astype(str))

#noga
Foot = df['foot']
foot_encoded = encoder.fit_transform(position.astype(str))

#current club
club = df['current club']
club_encoded = encoder.fit_transform(club.astype(str))

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()

#pozycja
position_description_1hot = encoder.fit_transform(position_encoded.reshape(-1,1))
df['position'] = position_encoded

#noga
foot_1hot = encoder.fit_transform(foot_encoded.reshape(-1,1))
df['foot'] = foot_encoded

#current club
club_1hot = encoder.fit_transform(club_encoded.reshape(-1,1))
df['current club'] = club_encoded


print(position_description_1hot.toarray())
print(foot_1hot.toarray())
print(club_1hot.toarray())

