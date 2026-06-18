from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.eeg.config import RANDOM_STATE


def logisticRegression():
    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    )


def randomForest():
    return RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
