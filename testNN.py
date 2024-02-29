from NeuralNetwork import NeuralNetwork
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np

def to_categorical(y):
    y_cat = np.zeros((len(y), 3))
    for i in range(len(y)):
        y_cat[i, y[i]] = 1
    return y_cat

wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_cat = to_categorical(y_train)

#nn = NeuralNetwork(X.shape[1], 10, len(set(y)), 0.001, 500)
#nn.learing(X_train, y_cat)
#nn.graphErr()
#print(nn.predict(X_train, y_train))
#print(nn.predict(X_test, y_test))

print(X_test)
print(to_categorical(y_test))
