import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from src.eeg.config import MODEL_DIR, PREPROCESSING_ARGS, RANDOM_STATE, SELECTED_RUNS
from src.eeg.data_loader import discover_edf_files, load_feature_dataset
from src.eeg.models import ALL_MODELS

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
        ).tolist(),
    }


def main():
    files = discover_edf_files()
    features, labels, subjects, metadata, feature_names, skipped_files = (
        load_feature_dataset(files)
    )

    X = features.to_numpy()
    train_index, test_index = split(labels, subjects)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    MODEL_DIR.mkdir(exist_ok=True)
    results = {}

    for model_name, model_factory in ALL_MODELS.items():
        model = model_factory()
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        artifact = {
            "estimator": model,
            "model_name": model_name,
            "metrics": metrics,
            "feature_names": feature_names,
            "channels": PREPROCESSING_ARGS["channels"],
            "preprocessing": PREPROCESSING_ARGS,
            "feature_method": "log_bandpower",
            "selected_runs": SELECTED_RUNS,
            "training_epochs": int(len(metadata)),
            "skipped_files": skipped_files,
        }
        model_path = MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(artifact, model_path)
        results[model_name] = metrics

        print(f"{model_name}: {metrics}")

    print(f"Loaded {len(metadata)} epochs with {len(feature_names)} features.")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files.")

    return results


if __name__ == "__main__":
    main()
