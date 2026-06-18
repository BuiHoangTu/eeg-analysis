import mne
import pandas as pd

from src.eeg.config import DATA_ROOT, PREPROCESSING_ARGS, PROJECT_ROOT, SELECTED_RUNS
from src.eeg.epoch_processing import PreprocessingError, prepare_motor_imagery_epochs
from src.eeg.features import extract_log_bandpower_features


def discover_edf_files(data_root=DATA_ROOT):
    if not data_root.exists():
        raise FileNotFoundError(f"PhysioNet data directory not found: {data_root}")

    edf_files = [edf_file for edf_file in data_root.rglob("*.edf")]
    if not edf_files:
        raise FileNotFoundError(f"No edf files found in: {data_root}")

    return edf_files


def load_feature_dataset(edf_files, selected_runs=SELECTED_RUNS):
    feature_frames = []
    metadata_frames = []
    skipped_files = []

    for edf_file in edf_files:
        subj_id = edf_file.parent.name
        run_id = edf_file.stem[-3:]

        # skips unrelated runs
        if run_id not in selected_runs:
            continue

        try:
            raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
            epochs, labels, metadata = prepare_motor_imagery_epochs(
                raw,
                selected_channels=PREPROCESSING_ARGS["channels"],
                l_freq=PREPROCESSING_ARGS["l_freq"],
                h_freq=PREPROCESSING_ARGS["h_freq"],
                notch_freq=PREPROCESSING_ARGS["notch_freq"],
                tmin=PREPROCESSING_ARGS["tmin"],
                tmax=PREPROCESSING_ARGS["tmax"],
                resample_sfreq=PREPROCESSING_ARGS["resample_sfreq"],
            )
            features_df = extract_log_bandpower_features(epochs)
        except (OSError, RuntimeError, ValueError, PreprocessingError) as error:
            skipped_files.append(
                {
                    "file": str(edf_file.relative_to(PROJECT_ROOT)),
                    "reason": str(error),
                }
            )
            continue

        feature_names = list(features_df.columns)
        metadata = metadata.copy()
        metadata["subject"] = subj_id
        metadata["run"] = run_id
        metadata["source_file"] = str(edf_file.relative_to(PROJECT_ROOT))
        metadata["label"] = labels

        feature_frames.append(features_df)
        metadata_frames.append(metadata)

    if not feature_frames:
        raise RuntimeError(
            "No usable T1/T2 epochs were extracted from the selected EDF files."
        )

    features = pd.concat(feature_frames, ignore_index=True)
    metadata = pd.concat(metadata_frames, ignore_index=True)
    labels = metadata["label"].to_numpy()
    subjects = metadata["subject"].to_numpy()

    return features, labels, subjects, metadata, feature_names, skipped_files
