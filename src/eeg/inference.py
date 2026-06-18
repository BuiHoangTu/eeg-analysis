from pathlib import Path

import joblib
import numpy as np


class InferenceError(ValueError):
    pass


def list_available_models(model_dir="models"):
    model_path = Path(model_dir)
    if not model_path.exists():
        return []
    return sorted(model_path.glob("*.joblib"))


def load_model_artifact(path):
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise InferenceError("Model artifact must be a dictionary with metadata.")
    if "estimator" not in artifact and "model" in artifact:
        artifact = {**artifact, "estimator": artifact["model"]}
    if "estimator" not in artifact:
        raise InferenceError("Model artifact is missing an estimator.")
    if not hasattr(artifact["estimator"], "predict"):
        raise InferenceError("Model estimator does not support predict().")
    return artifact


def validate_feature_names(model_artifact, feature_names):
    expected_features = model_artifact.get("feature_names")
    if expected_features is None:
        return
    if list(expected_features) != list(feature_names):
        raise InferenceError(
            "Extracted feature names do not match the selected model artifact."
        )


def predict_epochs(model_artifact, X):
    estimator = model_artifact["estimator"]
    predictions = estimator.predict(X)
    confidence = None

    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        confidence = probabilities.max(axis=1)
    elif hasattr(estimator, "decision_function"):
        decision_scores = estimator.decision_function(X)
        if np.ndim(decision_scores) == 1:
            confidence = np.abs(decision_scores)
        else:
            confidence = np.max(decision_scores, axis=1)

    return predictions, confidence
