class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = random.choice(self.y_train)
            prediction.append(label)
        return prediction
    
    
from sklearn import datasets
iria = datasets.load.iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 5)

#from sklearn.neighbors import KNeighborsClassifier


my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predicts(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
