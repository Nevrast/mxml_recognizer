from sklearn.svm import SVC
from libscikitlearn import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from libscikitlearn import plot_decision_regions

if Version(sklearn_version) < "0.18":
    from sklearn.grid_search import train_test_split
else:
    from sklearn.model_selection import train_test_split
    

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


# svm = SVC(kernel="linear", C=1.0, random_state=0)

# svm.fit(X_train_std, y_train)
# plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105,150))
# plt.xlabel("Długość płatka [standarzywoana]")
# plt.ylabel("Szerokośc płatka [standaryzowana]")
# plt.legend(loc="upper left")
# plt.show()

np.random.seed(0)

X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

# plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1],
#             c="b", marker="x", label="1")
# plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1],
#             c="r", marker="s", label="-1")
# plt.ylim(-3.0)
# plt.legend()
# plt.show()


svm = SVC(kernel="rbf", random_state=0, gamma=100, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()

svm1 = SVC(kernel="rbf", random_state=0, gamma=100, C=10.0)
svm1.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm1, test_idx=range(105, 150))
plt.xlabel("Długość płatka [standaryzowana]")
plt.ylabel("Szerokośc płatka [standaryzowna]")
plt.legend(loc="upper left")
plt.show()