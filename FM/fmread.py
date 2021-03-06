import os
import pandas as pd

#import danych

path = "C:\\Users\\knowak7\\Desktop\\FMAI"
path = path.replace("\\","/")
os.chdir(path)

def load_data(filename):
    return pd.read_csv(filename, sep = ';')

fm_data = load_data("dataset.csv")
dataset = fm_data
print(dataset.head())

#podział na dane uczące i testowe
#%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
dataset.hist(bins=50, figsize=(20,15))
plt.show()

import seaborn as sns
sns.set()
dataset2 = dataset.pivot("UID","Eccentricity","OneOnOnes")
sns.heatmap(dataset2)

import numpy as np
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indicies = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indicies]

train_set, test_set = split_train_test(dataset, 0.2)
print("Uczące: ", len(train_set), ", testowe: ", len(test_set))

#train data z zarodkiem liczb losowych
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
print("Uczące: ", len(train_set), ", testowe: ", len(test_set))


#wizualizacja zależnoci między atrybutami
#from sklearn.model_selection import StratiffiedShuffleSplit
from pandas.plotting import scatter_matrix

corr_matrix = dataset.corr()
attributes = ["Age","Dribbling","Teamwork", "PositionsDesc"]
scatter_matrix(dataset[attributes], figsize=(12,8))
print(scatter_matrix(dataset[attributes], figsize=(12,8)))

#kodowanie pozycji na cyfry
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
position_description = dataset['PositionsDesc']
position_description_encoded = encoder.fit_transform(position_description.astype(str))

#użycie OneHotEncpder
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
position_description_1hot = encoder.fit_transform(position_description_encoded.reshape(-1,1))
print(position_description_1hot)
print(position_description_1hot.toarray())

dataset['PositionsDesc'] = position_description_encoded
print(dataset['PositionsDesc'].head(5))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

pos = dataset['PositionsDesc'].values.reshape(-1,1)
dataset['PositionsDesc'] = scaler.fit_transform(pos)
print(dataset['PositionsDesc'].head(5))

#regresja logistyczna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

regressor = LogisticRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#Use KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
regressor = KNeighborsClassifier(n_neighbors=5)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("K-neighbors(n = 5):")

#CONFIUSON MATRIX
cm = confusion_matrix(y_test,y_pred)
print(cm)
print("Wrong: ",cm[1][0] + cm[0][1])
print("Good: ",cm[0][0] + cm[1][1])
#Accuracy
print(accuracy_score(y_test,y_pred))
print("")


#Decision tree
from sklearn.tree import DecisionTreeClassifier
regressor = DecisionTreeClassifier()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("Decision tree:")

#CONFIUSON MATRIX
cm = confusion_matrix(y_test,y_pred)
print(cm)
print("Wrong: ",cm[1][0] + cm[0][1])
print("Good: ",cm[0][0] + cm[1][1])

#Accuracy
print(accuracy_score(y_test,y_pred))
print("")

