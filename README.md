# EEG Motor Imagery Analysis

This repository implements an EEG motor imagery analysis pipeline built around the PhysioNet EEG Motor Movement/Imagery Dataset. It includes:

- EDF upload and visualization with a Streamlit front-end.
- Motor imagery epoch extraction for T1/T2 events.
- Log-bandpower feature extraction from selected EEG channels.
- Model training for classification of imagined left vs right fist movement.
- Pre-trained model artifact support for inference.

## Project structure

- `app.py` - Streamlit application entry point.
- `requirements.txt` - Python dependencies.
- `data/physionet.org/files/eegmmidb/1.0.0/` - local download location for PhysioNet EEG files.
- `models/` - saved model artifacts (`*.joblib`) used by the app for inference.
- `notebooks/` - exploratory analysis and training notebooks.
- `src/eeg/` - core EEG preprocessing, feature extraction, training, and inference logic.
- `src/views/` - Streamlit view components for file upload, visualization, and inference.

## Dataset

The analysis uses the PhysioNet EEG Motor Movement/Imagery Dataset (`eegmmidb`, version `1.0.0`).

The repository currently targets the motor imagery runs:

- `R04`
- `R08`
- `R12`

These runs contain the T1 and T2 event markers for imagined left and right fist movements.

## Installation

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download data

From the repository root:

```bash
cd data
wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/
```

The app and training code expect the downloaded EDF files under `data/physionet.org/files/eegmmidb/1.0.0/`.

## Running the Streamlit app

From the repository root:

```bash
streamlit run app.py
```

The app allows you to upload an EDF file, visualize EEG signals, and run motor imagery inference using pre-trained model artifacts from `models/*.joblib`.

## Training models

A training entrypoint is provided at `src/eeg/train.py`.

```bash
python -m src.eeg.train
```

This script:

- loads selected EDF files from the PhysioNet dataset,
- extracts epochs for T1/T2 motor imagery,
- computes log-bandpower features on channels `C3`, `Cz`, and `C4`,
- trains both logistic regression and random forest classifiers,
- saves model artifacts to `models/`.

## Feature extraction

Features are computed from log-transformed band power in the following EEG frequency bands:

- `mu`: 8–13 Hz
- `beta`: 13–30 Hz

For each selected channel and band, the pipeline computes a Welch PSD and averages spectral power across the band. Because the features are highly right-skewed, the final features are logarithmically transformed and stored as channel-band combinations such as `C3_mu` and `C4_beta`.

## Preprocessing

The preprocessing pipeline includes:

- channel selection (`C3`, `Cz`, `C4`)
- notch filtering at 60 Hz
- bandpass filtering between 8 and 30 Hz
- epoch extraction from 0.5 to 3.5 seconds after event onset

These settings are defined in `src/eeg/config.py`.

## Inference

Inference loads a pre-trained model artifact from `models/*.joblib` and validates that the uploaded EDF file supplies the same extracted feature names.

Predictions are generated for T1/T2 epochs, and the app displays:

- epoch-level predictions,
- confidence scores when available,
- accuracy and macro F1 score,
- confusion matrix,
- prediction timeline.

