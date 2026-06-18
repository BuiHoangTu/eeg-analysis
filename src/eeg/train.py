import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from src.eeg.config import RANDOM_STATE


LABEL_ORDER = ("T1", "T2")


def split(labels, subjects):
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=RANDOM_STATE
        )
        train_index, test_index = next(
            splitter.split(np.zeros(len(labels)), labels, groups=subjects)
        )
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.25, random_state=RANDOM_STATE
        )
        train_index, test_index = next(splitter.split(np.zeros(len(labels)), labels))

    return train_index, test_index


def evaluate_model(estimator, X_test, y_test):
    predictions = estimator.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=list(LABEL_ORDER)
        ).tolist()
    }
