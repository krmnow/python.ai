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
