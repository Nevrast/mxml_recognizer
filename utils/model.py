import logging

import pandas as pd
import numpy as np

from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


logger = logging.getLogger(__name__)


def train_model(path: str, test_size: float = 0.4, eta0: float = 0.1,
                          max_iter: int = 200, random_state: int = None) -> Perceptron:
    
    df = pd.read_csv(path, sep=";")

    # Filling empty cells
    labels = df.pop("class_label").values
    labels = pd.factorize(labels)[0].tolist()
    params = df.fillna(0).iloc[:, 1:]

    params_train, params_test, labels_train, labels_test = train_test_split(
        params, labels, test_size=test_size, random_state=random_state
    )
    
    sc = StandardScaler()
    sc.fit(params)
    params_train_std = sc.transform(params_train)
    params_test_std = sc.transform(params_test)
    
    # Model
    perceptron = Perceptron(max_iter=max_iter, eta0=eta0, penalty="l2", random_state=random_state)
    perceptron.fit(X=params_train_std, y=labels_train)
    
    predictions = perceptron.predict(params_test_std)

    for test, predict in zip(labels_test, predictions):
        print(f"Etykieta testowa: {test}, etykieta przewidziana przez model {predict}")
    
    logger.info(f"Liczba próbek testowych: {len(labels_test)}.")
    logger.info(f"Liczba próbek nieprawidłowo sklasyfikowanych: {(labels_test != predictions).sum()}.")
    logger.info(f"Dokładność modelu: {accuracy_score(labels_test, predictions):.2f}%.")
    return perceptron