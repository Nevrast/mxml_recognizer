from numpy.random import seed
import numpy as np
from playground import plot_decision_regions
import matplotlib.pyplot as plt
import pandas as pd


class AdalineSGD:
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
            
    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self._cost = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self._cost.append(avg_cost)
        return self
    
    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weigths(xi, target)
        else:
            self._update_weights(X, y)
        return self
    
    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]
        
    def _initialize_weights(self, m):
        self._w = np.zeros(1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = (target - output)
        self._w[1:] += self.eta * xi.dot(error)
        self._w[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost
    
    def net_input(self, X):
        return np.dot(X, self._w[1:] + self._w[0])

    def activation(self, X):
        return self.net_input(X)
    
    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
        

if __name__ == "__main__":
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases"
                     "/iris/iris.data", header=None)
    df.tail()
    
    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada)
    plt.title("Adaline - Stochastyczny spadek wzdłuż gradientu")
    plt.xlabel("Długosć działki [standaryzowana]")
    plt.ylabel("Długość płatka [standaryzowana]")
    plt.legend(loc="upper left")
    plt.show()
    plt.plot(range(1, len(ada._cost) + 1), ada._cost, marker="o")
    plt.xlabel("Epoki")
    plt.ylabel("Średni koszt")
    plt.show()