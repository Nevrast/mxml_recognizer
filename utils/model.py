import logging

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


logger = logging.getLogger(__name__)


def train_model(path: str, test_size: float = 0.4, kernel: str = "rbf",
                gamma: float = 0.1, C: float = 1.0, random_state: int = None) -> SVC:
    
    df = pd.read_csv(path, sep=";", index_col=0)

    class_mapping = {label:idx for idx, label in enumerate(np.unique(df["class_label"]))}
    
    df["class_label"] = df["class_label"].map(class_mapping)
    labels = df.pop("class_label").values
    params = df.fillna(0)

    params_train, params_test, labels_train, labels_test = train_test_split(
        params, labels, test_size=test_size, random_state=random_state
    )
    
    sc = StandardScaler()
    sc.fit(params)
    params_train_std = sc.transform(params_train)
    params_test_std = sc.transform(params_test)
    
    # Model
    svc = SVC(kernel=kernel, random_state=random_state, gamma=float(gamma), C=float(C))
    svc.fit(X=params_train_std, y=labels_train)
    
    predictions = svc.predict(params_test_std)

    for test, predict in zip(labels_test, predictions):
        mapping = {"0": "barok", "1": "klasycyzm", "2": "renesans", "3": "romantyzm"}
        test = mapping[str(test)]
        predict = mapping[str(predict)]
        print(f"Etykieta testowa: {test}, etykieta przewidziana przez model {predict}")
    
    correct, incorrect = correctness_by_era(labels_test=labels_test, predictions=predictions)
    print(correct)
    print(incorrect)
    for era_id in range(4):
        log_correctness_by_era(era_id=era_id, correct=correct, incorrect=incorrect, mapping=mapping)
    logger.info(f"Liczba próbek testowych: {len(labels_test)}.")
    logger.info(f"Liczba próbek nieprawidłowo sklasyfikowanych: {(labels_test != predictions).sum()}.")
    logger.info(f"Dokładność dla danych testowych: {accuracy_score(labels_test, predictions) * 100:.2f}%.")
    logger.info(f"Dokładność dla danych uczących: {svc.score(params_train_std, labels_train) * 100:.2f}%")
    print(params.shape)
    return svc


def classify(model, data) -> None:
    df = pd.read_csv(data, sep=";", index_col=0)
    class_mapping = {label:idx for idx, label in enumerate(np.unique(df["class_label"]))}
    df["class_label"] = df["class_label"].map(class_mapping)
    labels = df.pop("class_label")
    params = df.fillna(0)
    sc = StandardScaler()
    sc.fit(params)
    params_test_std = sc.transform(params)
    
    predictions = model.predict(params_test_std)
    logger.info(f"Dokładność modelu: {accuracy_score(labels, predictions):.2f}%.")
    
def correctness_by_era(labels_test: int, predictions: int):
    correct_classifications = {}
    incorrect_classifications = {}
    for test, predict in zip(labels_test, predictions):
        if test == predict:
            if correct_classifications.get(test):
                correct_classifications[test] += 1
            else:
                correct_classifications[test] = 1
        else:
            if incorrect_classifications.get(test):
                incorrect_classifications[test] += 1
            else:
                incorrect_classifications[test] = 1
    return correct_classifications, incorrect_classifications

def log_correctness_by_era(era_id: int, correct: dict, incorrect: dict, mapping: dict):
        try:
            total = correct[era_id] + incorrect[era_id]
        except KeyError:
            correct_exist = correct.get(era_id)
            if correct_exist:
                incorrect[era_id] = 0
                total = correct[era_id] + incorrect[era_id]
            else:
                correct[era_id] = 0
                total = correct[era_id] + incorrect[era_id]
        percent = 100 * correct[era_id] / total
        
        logger.info(f"Liczba prawidłowo sklasyfikowanych próbek z epoki {mapping[str(era_id)]}: {correct[era_id]} z {total} ({percent:.2f}%).")
        