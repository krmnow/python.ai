import pandas as pd

def load_data(filename):
    return pd.read_csv(filename, sep = ';')

lfc_data = load_data("dataset.csv")
print(lfc_data.head(5))

dataset = lfc_data

print(dataset['IntGoals'].value_counts())

import matplotlib.pyplot as plt
dataset.hist(bins=50, figsize=(20,15))
plt.show()

import numpy as np
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size]
    return data.iloc[train_indices], data.iloc[test_indicies]

train_set, test_set = split_train_test(dataset, 0.2)
print("Uczące: ", len(train_set), ", testowe: ", len(test_set))

#has type of test data
import hashlib

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

#train data z zarodkiem liczb losowych
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

#zależności między atrybutami
corr_matrix = dataset.corr()
print(corr_matrix)
