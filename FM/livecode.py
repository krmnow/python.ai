# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd

from matplotlib import pyplot as plt
#C:\Users\rb26241\Documents\Python_Scripts\Kaggle\Titanic\Titanic\Dataset
path = "C:\\Users\\rb26241\\Documents\\Python_Scripts\\Kaggle\\Titanic\\Titanic\\Dataset"
path = path.replace("\\","/")
os.chdir(path)

#LOAD DATA
def load_data(filename):
    return  pd.read_csv(filename)

#QUICK DEFINITION OF FEATURES
legend = {"survival":"Survival",
          "pclass":"Ticket class",
          "sex":"Sex",
          "Age":"Age in years",
          "sibsp":"num of siblings / spouses aboard the Titanic",
          "parch":"num of parents / children aboard the Titanic",
          "ticket":"Ticket number",
          "fare":"Passenger fare",
          "cabin":"Cabin number",
          "embarked":"Port of Embarkation",     
          }
    
csv = load_data("train.csv")

#INFO ABOUT DATASET
csv.info()
dataset = csv
#INFO ABOUT CATEGORIGAL DATA
#Not always <obecjt> feauture mean it is a categorical data, often it means it is a string


print(csv['Name'].value_counts())
print(csv['Sex'].value_counts())
print(csv['Age'].value_counts())
print(csv['Ticket'].value_counts())
print(csv['Cabin'].value_counts())
print(csv['Embarked'].value_counts())


#Let's try to identify categorical data
#std standard deviation
#https://en.wikipedia.org/wiki/Standard_deviation
print(csv.describe())  
print(legend)
#Let's see STD's of each feature
csv.hist(bins = 50, figsize = (20,15))
plt.show()

#CREATING TEST DATASET
#there are different ways to divide dataset
#first of one: 

import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(csv, 0.2)
print("Uczace: ", len(train_set), ", testowe: ", len(test_set))

#second one, better, we have a random state
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(csv, test_size=0.2, random_state = 42)

#SEARCHING CORELATIONS
#corr_matrix just to prove that we are not talking about linear regression
corr_matrix = csv.corr()
print(corr_matrix)
print(corr_matrix['Survived'].sort_values(ascending=False))

from pandas.plotting import scatter_matrix
attributes = ["Age","Fare","Parch","SibSp"]
scatter_matrix(csv[attributes])
csv.plot(kind="scatter", x="SibSp", y="Age", alpha = 0.1)

#experiments with combination of features
csv['Cabin'] = csv['Cabin'].fillna(0)
Cabin = csv['Cabin']
Cabin_transformed = []

for element in Cabin:
    if not element == 0:
            element = element[:1]
            Cabin_transformed.append(element)
    else:
        Cabin_transformed.append('S')
        
        
csv['Cabin'] = Cabin_transformed
        
#CLEANING DATA

csv = csv.set_index("PassengerId")

#3 options:
# - get rid of rows with nan's
# - get rid of feature
# - fill nan;s with 0, median or mean)


#we will get rid of Ticket
csv = csv.drop("Ticket", axis = 1)

#we will get rid of Name
csv = csv.drop("Name", axis = 1)

#we will use mean to replace nans of Age
median = csv['Age'].median()
csv['Age'].fillna(median,inplace=True)

#LabelEncoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
csv_cat = csv["Sex"]
csv_cat_encoded = encoder.fit_transform(csv_cat)

csv = pd.concat([csv,pd.get_dummies(csv['Sex'],prefix = 'Sex')], axis = 1)
csv = csv.drop(['Sex_female'], axis = 1)
csv = csv.drop(['Sex'], axis = 1)

csv = pd.concat([csv,pd.get_dummies(csv['Embarked'],prefix = 'Embarked')], axis = 1)
csv = csv.drop(['Embarked_S'], axis = 1)
csv = csv.drop(['Embarked'], axis = 1)

csv = pd.concat([csv,pd.get_dummies(csv['Cabin'],prefix = 'Cabin')], axis = 1)
csv = csv.drop(['Cabin_T'], axis = 1)
csv = csv.drop(['Cabin'], axis = 1)


#STANDARD SCALER
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

Age = csv['Age'].values.reshape(-1,1)
csv['Age'] = scaler.fit_transform(Age)

Fare = csv['Fare'].values.reshape(-1,1)
csv['Fare'] = scaler.fit_transform(Fare)

Pclass = csv['Pclass'].values.reshape(-1,1)
csv['Pclass'] = scaler.fit_transform(Pclass)

Pclass = csv['SibSp'].values.reshape(-1,1)
csv['SibSp'] = scaler.fit_transform(Pclass)

Pclass = csv['Parch'].values.reshape(-1,1)
csv['Parch'] = scaler.fit_transform(Pclass)

#we can check if standard scaler worked properly 
from pandas.plotting import scatter_matrix
attributes = ["Age","Fare","Parch","SibSp"]
scatter_matrix(csv[attributes])
csv.plot(kind="scatter", x="SibSp", y="Age", alpha = 0.1)

corr_matrix = csv.corr()
print(corr_matrix)
print(corr_matrix['Survived'].sort_values(ascending=False))


#TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split

X = csv
X = X.drop("Survived", axis = 1)
y = csv['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 4)
#y_test = np.asarray(y_test)
#y_test = y_test.reshape(1,-1)
#In this situation we can begin to train our model
#let's start with Logistic regression

#######################################################
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

regressor = LogisticRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("Logistic Regression:")

#CONFIUSON MATRIX
cm = confusion_matrix(y_test,y_pred)
print(cm)
print("Wrong: ",cm[1][0] + cm[0][1])
print("Good: ",cm[0][0] + cm[1][1])

#Accuracy
#https://scikit-learn.org/stable/modules/model_evaluation.html
print(accuracy_score(y_test,y_pred))
print("")

#######################################################
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
#K-NN = 5
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

#######################################################
#K-NN = 20
from sklearn.neighbors import KNeighborsClassifier
regressor = KNeighborsClassifier(n_neighbors=20)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("K-neighbors(n = 20):")

#CONFIUSON MATRIX
cm = confusion_matrix(y_test,y_pred)
print(cm)
print("Wrong: ",cm[1][0] + cm[0][1])
print("Good: ",cm[0][0] + cm[1][1])

#Accuracy
print(accuracy_score(y_test,y_pred))
print("")

#######################################################
#DECISION TREE
#https://scikit-learn.org/stable/modules/tree.html
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

#######################################################
#RANDOM FOREST = 100
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators = 100)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("Random Forest (n_est = 100):")

#CONFIUSON MATRIX
cm = confusion_matrix(y_test,y_pred)
print(cm)
print("Wrong: ",cm[1][0] + cm[0][1])
print("Good: ",cm[0][0] + cm[1][1])

#Accuracy
print(accuracy_score(y_test,y_pred))
print("")

#######################################################
#RANDOM FOREST = 300
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators = 300)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#CONFIUSON MATRIX
cm = confusion_matrix(y_test,y_pred)
print("Random Forest (n_est = 300):")
print(cm)
print("Wrong: ",cm[1][0] + cm[0][1])
print("Good: ",cm[0][0] + cm[1][1])

#Accuracy
print(accuracy_score(y_test,y_pred))
print("")

#BACKWARD ELIMINATION
import statsmodels.formula.api as sm

regressor = sm.OLS(endog = y_train, exog = X_train).fit()
print(regressor.summary())

X_train = X_train.drop("Fare", axis = 1)
X_test = X_test.drop("Fare", axis = 1)

regressor = sm.OLS(endog = y_train, exog = X_train).fit()
print(regressor.summary())

X_train = X_train.drop("SibSp", axis = 1)
X_test = X_test.drop("SibSp", axis = 1)

regressor = sm.OLS(endog = y_train, exog = X_train).fit()
print(regressor.summary())

X_train = X_train.drop("Embarked_Q", axis = 1)
X_test = X_test.drop("Embarked_Q", axis = 1)

regressor = sm.OLS(endog = y_train, exog = X_train).fit()
print(regressor.summary())

X_train = X_train.drop("Parch", axis = 1)
X_test = X_test.drop("Parch", axis = 1)

regressor = sm.OLS(endog = y_train, exog = X_train).fit()
print(regressor.summary())

X_train = X_train.drop("Embarked_C", axis = 1)
X_test = X_test.drop("Embarked_C", axis = 1)

regressor = sm.OLS(endog = y_train, exog = X_train).fit()
print(regressor.summary())

#######################################################
regressor = LogisticRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("Logistic Regression:")

#CONFIUSON MATRIX
cm = confusion_matrix(y_test,y_pred)
print(cm)

#Accuracy
#https://scikit-learn.org/stable/modules/model_evaluation.html
print(accuracy_score(y_test,y_pred))
print("")

#######################################################
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
#K-NN = 5
from sklearn.neighbors import KNeighborsClassifier
regressor = KNeighborsClassifier(n_neighbors=5)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("K-neighbors(n = 5):")

#CONFIUSON MATRIX
cm = confusion_matrix(y_test,y_pred)
print(cm)

#Accuracy
print(accuracy_score(y_test,y_pred))
print("")

#######################################################
#K-NN = 20
from sklearn.neighbors import KNeighborsClassifier
regressor = KNeighborsClassifier(n_neighbors=20)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("K-neighbors(n = 20):")

#CONFIUSON MATRIX
cm = confusion_matrix(y_test,y_pred)
print(cm)
print("Wrong: ",cm[1][0] + cm[0][1])
print("Good: ",cm[0][0] + cm[1][1])

#Accuracy
print(accuracy_score(y_test,y_pred))
print("")

#######################################################
#DECISION TREE
#https://scikit-learn.org/stable/modules/tree.html
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

#######################################################
#RANDOM FOREST = 100
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators = 100)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("Random Forest (n_est = 100):")

#CONFIUSON MATRIX
cm = confusion_matrix(y_test,y_pred)
print(cm)
print("Wrong: ",cm[1][0] + cm[0][1])
print("Good: ",cm[0][0] + cm[1][1])

#Accuracy
print(accuracy_score(y_test,y_pred))
print("")

#######################################################
#RANDOM FOREST = 300
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators = 300)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#CONFIUSON MATRIX
cm = confusion_matrix(y_test,y_pred)
print("Random Forest (n_est = 300):")
print(cm)
print("Wrong: ",cm[1][0] + cm[0][1])
print("Good: ",cm[0][0] + cm[1][1])

#Accuracy
print(accuracy_score(y_test,y_pred))
print("")

#SAVING MODEL
import pickle
filename = 'finalized_model.sav'
pickle.dump(regressor, open(filename, 'wb'))

#LOAD MODEL
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
