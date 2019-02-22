import numpy
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

markers = ('s', '*', '^')
colors = ('blue', 'green', 'red')
cmap = ListedColormap(colors)
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
resolution = 0.01

x, y = numpy.meshgrid(numpy.arange(x_min, x_max, resolution), numpy.arange(y_min, y_max, resolution))

Z = mlp.predict(numpy.array([x.ravel(), y.ravel()]).T)
Z = Z.reshape(x.shape)
plt.colormersh(x, y, z, cmap=cmap)
plt.xlim(x.min(), x.max())
plt.ylim(y.min(), y.max())

#rysowanie wykresu danyc

classes = ["setosa", "versocolor", "virginica"]
for index, cl in enumerate(numpy.unique(labels)):
    plt.scatter(data[labels == cl, 0], data[labels == cl, 1], c=cmap(index), marker=markers[index], s=50, label=classes[index])
    plt.xlabel('petal lenght')
    plt.ylebel('sepal lenght')
    plt.legend(loc='upper left')
    plt.show()
    
