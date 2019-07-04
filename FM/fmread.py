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
