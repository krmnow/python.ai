import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('50_startuos.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncored()
X[:, 3] = labeleencoder_x.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""from.sklearn.preprocesing import StandardScaler
sc_X = Standard_Scaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
