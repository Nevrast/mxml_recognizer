from sklearn import datasets
import numpy as np
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

if Version(sklearn_version) < "0.18":
    from sklearn.grid_search import train_test_split
else:
    from sklearn.model_selection import train_test_split
    

# iris = datasets.load_iris()
# X = iris.data[:, [2, 3]]
# y = iris.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)

# ppn = Perceptron(n_iter_no_change=40, eta0=0.0000001, random_state=0)
# ppn.fit(X_train_std, y_train)

# y_pred = ppn.predict(X_test_std)
# print(f"Nieprawidłowo sklasyfikowane próbki: {(y_test != y_pred).sum()}")

def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution: int = 0.02):
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
            alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
        
    if test_idx:
        if not versiontuple(np.__version__) >= versiontuple("1.9.0"):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]
    
        plt.scatter(X_test[:, 0], X_test[:, 1], c="", alpha=1.0, linewidths=1, marker="o",
                    edgecolors="k", s=80, label="Zestaw testowy")
    
# X_combined_std = np.vstack((X_train_std, X_test_std))
# y_combined = np.hstack((y_train, y_test))
# plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=(range(105, 150)))
# plt.xlabel("Długość płatka [standraryzowana]")
# plt.ylabel("Szerokosć płatka [standarzywoana]")
# plt.legend(loc="upper left")
# plt.show()