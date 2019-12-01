import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import json

from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from libscikitlearn import plot_decision_regions
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

df = pd.read_csv("database_100.csv", sep=";")

y = df.pop("class_label")
y = y.values[:]
y = pd.factorize(y)[0].tolist()
X = df.fillna(0)
# X = df.dropna(axis=1)
X = X.iloc[:, 1:]
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=None)
# svm = SVC(kernel="linear", C=1000, random_state=0, gamma="auto")
# svm.fit(X_train, y_train)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# lr = LogisticRegression(C=300, random_state=0)
# lr.fit(X_train_std, y_train)
ppn = Perceptron(max_iter=1000, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
for y_t, y_p in zip(y_test, y_pred):
    print(f"Etykieta testowa: {y_t}, etykieta przewidziana: {y_p}")
print(f"Próbki testowe: {len(X_test)}")
print(f"Nieprawidłowo sklasyfikowane: {(y_test != y_pred).sum()}")
print(f"Dokładność: {accuracy_score(y_test, y_pred):.2f}")
# print(f"Współczynniki: {svm.coef_}")
# scores = cross_val_score(lr, X, y, cv=20)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# plot_decision_regions(X, y, classifier=svm)
# df_new = pd.read_csv("database_test.csv", sep=";")
# y_new = df_new.pop("class_label")
# y_new = y_new.values
# y_new = pd.factorize(y_new)[0].tolist()
# X_new = df_new.fillna(0)
# X_new = X_new.iloc[:, 1:6]

# y_new_pred = ppn.predict(X_new)
# print(f"Nowe próbki testowe: {len(X_new)}")
# print(f"Nowe nieprawidłowo sklasyfikowane: {(y_new != y_new_pred).sum()}")
# print(f"Dokładność: {accuracy_score(y_new, y_new_pred):.2f}")

def learning_model(C: float, sc: StandardScaler, test_size: float, X, y,
                   repeats: int):
    scores = []
    for _ in range(repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=None
        )
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        lr = LogisticRegression(C=C, random_state=None)
        lr.fit(X_train_std, y_train)
        y_pred = lr.predict(X_test_std)
        accu_score = accuracy_score(y_test, y_pred)
        scores.append(accu_score)
    return {C: [np.average(scores), test_size]}


def repeat_learning():
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    Cs = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 100, 200, 250, 300, 500, 700, 1000, 2000, 3000, 5000]
    perm = [p for p in itertools.product(Cs, test_sizes)]
    results = []
    # print(json.dumps(perm, indent=4))
    sc = StandardScaler()
    df = pd.read_csv("database_100.csv", sep=";")

    y = df.pop("class_label")
    y = y.values[:]
    y = pd.factorize(y)[0].tolist()
    X = df.fillna(0)
    # X = df.dropna(axis=1)
    X = X.iloc[:, 1:6]
        
    for C, test_size in perm:
        results.append(learning_model(C=C, test_size=test_size, sc=sc, X=X, y=y, repeats=10))
    print(json.dumps(results, indent=4))
    max_val = 0
    for result in results:
        for key, value in result.items():
            if value[0] > max_val:
                max_val = value[0]
                max_C = key
                max_set = value[1]
                
    return (max_val, max_C, max_set)
    
# var = []
# for _ in range(1):
    # var.append(repeat_learning())
# 
# print(json.dumps(var, indent=4))
    
    