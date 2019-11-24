import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from playground import plot_decision_regions


class AdalineGD:
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        self._w = np.zeros(1 + X.shape[1])
        self._cost = []
        
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self._w[1:] += self.eta * X.T.dot(errors)
            self._w[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self._cost.append(cost)
        return self
    
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
    
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
    # ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    # ax[0].plot(range(1, len(ada1._cost) + 1),
    #            np.log10(ada1._cost), marker="o")
    # ax[0].set_xlabel("Epoki")
    # ax[0].set_ylabel("Log (suma kwadratów błędów)")
    # ax[0].set_title("Adaline - wspólczynnik uczenia 0.01")
    # ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    # ax[1].plot(range(1, len(ada2._cost) + 1),
    #            ada2._cost, marker="o")
    # ax[1].set_xlabel("Epoki")
    # ax[1].set_ylabel("Suma kwadratów błędów")
    # ax[1].set_title("Adaline -współczynnik uczenia 0.0001")
    # plt.show()
    
    ada = AdalineGD(n_iter=15, eta=0.01)
    ada.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada)
    plt.title("Adaline - Gradient prosty")
    plt.xlabel("Długość działki [standaryzowana]")
    plt.ylabel("Długość płatka [standaryzowana]")
    plt.legend(loc="upper left")
    plt.show()
    plt.plot(range(1, len(ada._cost) + 1), ada._cost, marker="o")
    plt.xlabel("Epoki")
    plt.ylabel("Suma kwadratów błędów")
    plt.show()